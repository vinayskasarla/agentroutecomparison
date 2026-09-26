"""LLM Path Lab — compare ways of routing an agent's LLM call.

One FastAPI process plays every role so the comparison needs no extra infra:
  * the "agent" (the /api/run orchestrator), which can consult Jev (TypeSafe's System One model),
  * an AI Gateway and an Agent Runtime (the /svc/* endpoints),
    reached over real HTTP hops so their serialization + network cost is measured,
  * Redis (real, if reachable; otherwise an in-process stand-in, labeled as such).

Run:  uvicorn app:app --port 8088   then open http://localhost:8088
"""
import asyncio
import collections
import hashlib
import json
import os
import random
import re
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import envfile  # noqa: F401  (loads .env before anything reads the environment)
import catalog
import advisor
import constraints
import grading
import audit
from pii import redact
import news
import policy
import pricing_sync
from providers import ROUTE_QUESTION, call_jev, call_llm, has_jev_key, has_key, jev_answer_question

app = FastAPI(title="LLM Path Lab")


@app.on_event("startup")
async def refresh_prices():
    """Pull current prices from the machine-readable sources in the background; failures keep reviewed values."""
    if os.environ.get("PRICE_SYNC", "on") != "off":
        async def run():
            try:
                report = await asyncio.wait_for(pricing_sync.sync(), 120)
                audit.log("prices_synced", None, sources=report["sources"], changes=report["changes"])
            except Exception as exc:
                audit.log("prices_sync_failed", None, error=f"{type(exc).__name__}: {exc}")
        asyncio.create_task(run())
PAGES = {"/": "advisor", "/ask": "ask", "/benchmark": "benchmark", "/news": "news"}


APP_PASSWORD = os.environ.get("APP_PASSWORD")  # optional shared password until SSO sits in front of the app


@app.middleware("http")
async def password_gate(request: Request, call_next):
    """HTTP basic auth when APP_PASSWORD is set (any username). Off by default; SSO replaces it in production."""
    if APP_PASSWORD:
        import base64, secrets
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.startswith("Basic "):
            try:
                ok = secrets.compare_digest(base64.b64decode(auth[6:]).decode().split(":", 1)[1], APP_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response("Password required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Architecture Advisor"'})
    return await call_next(request)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Request and session IDs for the audit trail, plus a page_view event per page load."""
    request.state.request_id = uuid.uuid4().hex[:12]
    request.state.session_id = request.cookies.get("plab_sid") or uuid.uuid4().hex[:16]
    if request.method == "GET" and request.url.path in PAGES:
        audit.log("page_view", request, page=PAGES[request.url.path])
    response = await call_next(request)
    if "plab_sid" not in request.cookies:
        response.set_cookie("plab_sid", request.state.session_id, httponly=True, samesite="lax",
                            secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https")
    response.headers["X-Request-Id"] = request.state.request_id
    return response
SELF_URL = os.environ.get("SELF_URL", "http://127.0.0.1:8088")
_http = httpx.AsyncClient(timeout=180)

# ---------------------------------------------------------------- cache backend
_redis = None
_mem_kv: dict = {}
_semantic_index: dict = {}  # namespace -> [(token_set, payload)]


async def cache_backend() -> str:
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"),
                                       socket_connect_timeout=0.5)
            await client.ping()
            _redis = client
        except Exception:
            _redis = False
    return "redis" if _redis else "in-memory"


async def kv_get(key):
    if await cache_backend() == "redis":
        raw = await _redis.get(key)
        return json.loads(raw) if raw else None
    return _mem_kv.get(key)


async def kv_set(key, value):
    if await cache_backend() == "redis":
        await _redis.set(key, json.dumps(value), ex=3600)
    else:
        _mem_kv[key] = value


def exact_key(ns, model, prompt):
    norm = re.sub(r"\s+", " ", prompt.strip().lower())
    return f"lab:{ns}:{hashlib.sha256(f'{model}|{norm}'.encode()).hexdigest()[:24]}"


def tokens(text):
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def semantic_lookup(ns, prompt, threshold):
    """Token-overlap (Jaccard) similarity: a transparent stand-in for embedding search."""
    toks = tokens(prompt)
    best, best_sim = None, 0.0
    for other, payload in _semantic_index.get(ns, []):
        sim = len(toks & other) / max(1, len(toks | other))
        if sim > best_sim:
            best, best_sim = payload, sim
    return (best, best_sim) if best_sim >= threshold else (None, best_sim)


# ------------------------------------------------------------ middleware logic
INJECTION = re.compile(r"ignore (all|previous|prior) instructions|reveal (the )?system prompt", re.I)


def is_complex(prompt):
    """Ground truth for simulated routing of free-form prompts: arithmetic, counting, comparison or long prompts."""
    return len(prompt.split()) > 16 or bool(
        re.search(r"\d+\s*[*x+\-/]\s*\d+|how many|larger|smaller|time|calculate", prompt, re.I))


_buckets: dict = {}


def rate_limit_ok(app_id, rate=50.0, burst=100.0):
    now = time.monotonic()
    tokens_left, last = _buckets.get(app_id, (burst, now))
    tokens_left = min(burst, tokens_left + (now - last) * rate)
    ok = tokens_left >= 1
    _buckets[app_id] = (tokens_left - 1 if ok else tokens_left, now)
    return ok


def fault_hit(ctx, attempt):
    # the outage hits the primary provider; the failover provider is rarely down at the same time
    rate = ctx.get("fault_rate", 0) * (1 if attempt == 0 else 0.1)
    if not rate:
        return False
    rng = random.Random(f"{ctx['run_id']}|{ctx['qid']}|{ctx['pass']}|{attempt}")
    return rng.random() < rate


async def llm_stage(model, prompt, ctx, extra_tokens=0, failover=False):
    """Call the model (with optional provider failover). Returns (result, stages, wall_ms)."""
    stages, t0 = [], time.perf_counter()
    attempt_model = model
    for attempt in range(2 if failover else 1):
        if fault_hit(ctx, attempt):
            stages.append({"name": f"Provider error ({attempt_model})", "kind": "error", "ms": 250.0})
            if failover:
                attempt_model = policy.safe_fallback(model)  # only ever fail over to an approved model
                if attempt_model:
                    continue
                break
            return ({"error": "provider 503 (injected fault)", "answer": "", "confidence": None,
                     "in_tokens": 0, "out_tokens": 0, "model": model, "simulated": True},
                    stages, (time.perf_counter() - t0) * 1000)
        res = await call_llm(attempt_model, prompt, item=graded_item(ctx), extra_context_tokens=extra_tokens,
                             force_sim=ctx["force_sim"], seed=ctx["pass"])
        stages.append({"name": f"LLM · {attempt_model}", "kind": "llm", "ms": res["llm_ms"],
                       "simulated": res["simulated"]})
        res["failover"] = attempt > 0
        return res, stages, (time.perf_counter() - t0) * 1000 if not res["simulated"] else 0.0
    return ({"error": "all providers failed", "answer": "", "confidence": None, "in_tokens": 0,
             "out_tokens": 0, "model": model, "simulated": True}, stages, 0.0)


async def hop(url, body, rtt_ms):
    """Real HTTP call to another service + configured network RTT. Returns (payload, hop_ms)."""
    t0 = time.perf_counter()
    await asyncio.sleep(rtt_ms / 1000)
    resp = await _http.post(f"{SELF_URL}{url}", json=body)
    payload = resp.json()
    wall = (time.perf_counter() - t0) * 1000
    return payload, max(0.0, wall - payload["internal_ms"])


def graded_item(ctx):
    """The test case this call belongs to, or None for an ungraded free-form prompt."""
    item = ctx.get("item")
    return item if item and item.get("accept") else None


def jev_stage(label, r):
    return {"name": label, "kind": "jev", "ms": r["jev_ms"], "simulated": r["simulated"]}


async def jev_route(model, prompt, ctx):
    """Ask Jev whether the provider's small model is enough. Returns (model_to_use, stage, jev_tokens)."""
    if not policy.service_allowed("jev"):
        return model, {"name": "Jev router off (vendor not approved)", "kind": "jev", "ms": 0.0, "simulated": True}, 0
    item = graded_item(ctx)
    needs_large = item["difficulty"] >= 2 if item else is_complex(prompt)
    r = await call_jev(prompt, ROUTE_QUESTION, truth={"route": "large" if needs_large else "small"},
                       force_sim=ctx["force_sim"], seed=ctx["pass"])
    if r["error"]:  # fail safe: keep the model the app asked for
        return model, jev_stage("Jev router failed → keep model", r), 0
    a = r["answers"]["route"]
    small = catalog.PROVIDERS[catalog.MODEL_BY_ID[model]["provider"]]["small"]
    if not policy.model_allowed(small):
        small = model
    return (small if a["choice"] == "small" else model), jev_stage(
        f"Jev router → {a['choice']} ({a['confidence']:.2f})", r), r["in_tokens"]


# ---------------------------------------------------------------- the services
class SvcReq(BaseModel):
    model: str
    prompt: str
    ctx: dict
    features: dict = {}


@app.post("/svc/gateway")
async def svc_gateway(req: SvcReq):
    t0 = time.perf_counter()
    f, ctx = req.features, req.ctx
    ns = f"{ctx['cache_ns']}:{ctx['path']}"
    if not rate_limit_ok(ctx["app_id"]):
        return {"result": {"error": "429 quota exceeded"}, "stages": [], "internal_ms": 0}
    prompt, pii = redact(req.prompt)
    blocked = bool(INJECTION.search(prompt))
    pre_ms = (time.perf_counter() - t0) * 1000
    stages = [{"name": "Gateway: key vault, quota, guardrails", "kind": "gateway", "ms": pre_ms}]
    info = {"pii_redacted": pii, "cache": None, "routed_to": None, "jev_in_tokens": 0}
    if blocked:
        return {"result": {"error": "blocked by prompt-injection guard"}, "stages": stages,
                "internal_ms": pre_ms, **info}

    model = req.model
    if f.get("cache"):
        c0 = time.perf_counter()
        hit = await kv_get(exact_key(ns, model, prompt))
        kind = "exact" if hit else None
        if not hit and f.get("semantic"):
            hit, sim = semantic_lookup(ns, prompt, ctx["semantic_threshold"])
            kind = f"semantic {sim:.2f}" if hit else None
        stages.append({"name": "Cache lookup" + (f" (HIT: {kind})" if hit else " (miss)"), "kind": "cache",
                       "ms": (time.perf_counter() - c0) * 1000})
        if hit:
            info["cache"] = kind
            return {"result": {**hit, "in_tokens": 0, "out_tokens": 0}, "stages": stages,
                    "internal_ms": (time.perf_counter() - t0) * 1000, **info}

    if f.get("jev_first"):
        item = graded_item(ctx)
        if item and item.get("options"):  # a decision with fixed answers: let Jev take it if it's confident
            r = await call_jev(prompt, jev_answer_question(item["options"]), truth={"answer": item["sim_right"]},
                               difficulty=item["difficulty"], force_sim=ctx["force_sim"], seed=ctx["pass"])
            info["jev_in_tokens"] = r["in_tokens"]
            if r["error"]:
                stages.append(jev_stage("Jev error → LLM", r))
            else:
                a = r["answers"]["answer"]
                stages.append(jev_stage(f"Jev · choice {a['choice']} ({a['confidence']:.2f})", r))
                if a["confidence"] >= ctx["jev_threshold"]:
                    return {"result": {"answer": a["choice"], "confidence": a["confidence"] * 100, "in_tokens": 0,
                                       "out_tokens": 0, "model": r["model"], "simulated": r["simulated"]},
                            "stages": stages, "internal_ms": (time.perf_counter() - t0) * 1000, **info,
                            "decided_by": "jev"}
                info["escalated"] = True
        else:  # free text: Jev can't answer it, so it picks the model tier instead
            f = {**f, "router": True}

    if f.get("router"):
        model, stage, jev_tokens = await jev_route(req.model, prompt, ctx)
        info["jev_in_tokens"] += jev_tokens
        stages.append(stage)
        info["routed_to"] = model if model != req.model else None

    res, llm_stages, _ = await llm_stage(model, prompt, ctx, extra_tokens=f.get("extra_tokens", 0),
                                            failover=True)
    stages += llm_stages
    p0 = time.perf_counter()
    if f.get("cache") and not res.get("error"):
        payload = {k: res[k] for k in ("answer", "confidence", "model", "simulated")}
        await kv_set(exact_key(ns, req.model, prompt), payload)
        index = _semantic_index.setdefault(ns, [])
        index.append((tokens(prompt), payload))
        del index[:-500]  # keep long-lived (Ask page) namespaces bounded
    # output guardrail + usage metering for charge-back
    redact(res.get("answer", ""))
    stages.append({"name": "Gateway: output guard + metering", "kind": "gateway",
                   "ms": (time.perf_counter() - p0) * 1000})
    return {"result": res, "stages": stages, "internal_ms": (time.perf_counter() - t0) * 1000, **info}


@app.post("/svc/runtime")
async def svc_runtime(req: SvcReq):
    t0 = time.perf_counter()
    # load session memory, check tool/data policy, open a trace span
    memory = json.dumps({"session": req.ctx["app_id"], "facts": ["user prefers concise answers"] * 8})
    json.loads(memory)
    features = {**req.features, "extra_tokens": req.ctx["runtime_context_tokens"]}
    pre = (time.perf_counter() - t0) * 1000
    payload, hop_ms = await hop("/svc/gateway", {**req.model_dump(), "features": features}, req.ctx["hop_rtt_ms"])
    p0 = time.perf_counter()
    trace = json.dumps(payload["stages"])  # flush trace span
    post = (time.perf_counter() - p0) * 1000 if trace else 0.0
    stages = ([{"name": "Runtime: memory + policy + trace", "kind": "runtime", "ms": pre + post},
               {"name": "Network hop → gateway", "kind": "hop", "ms": hop_ms}] + payload["stages"])
    return {**payload, "stages": stages, "internal_ms": (time.perf_counter() - t0) * 1000}


# -------------------------------------------------------------- orchestration
async def run_path(path_id, model, prompt, ctx):
    rtt = ctx["hop_rtt_ms"]
    body = {"model": model, "prompt": prompt, "ctx": ctx}
    if path_id in ("direct", "direct_small"):
        use = catalog.PROVIDERS[catalog.MODEL_BY_ID[model]["provider"]]["small"] if path_id == "direct_small" else model
        res, stages, _ = await llm_stage(use, prompt, ctx)
        return {"result": res, "stages": stages, "pii_redacted": 0, "pii_sent": bool(redact(prompt)[1]),
                "routed_to": use if use != model else None}
    if path_id == "redis":
        c0 = time.perf_counter()
        key = exact_key(f"{ctx['cache_ns']}:redis", model, prompt)
        hit = await kv_get(key)
        stages = [{"name": "Redis GET" + (" (HIT)" if hit else " (miss)"), "kind": "cache",
                   "ms": (time.perf_counter() - c0) * 1000}]
        if hit:
            return {"result": {**hit, "in_tokens": 0, "out_tokens": 0}, "stages": stages, "cache": "exact",
                    "pii_sent": False}
        res, llm_stages, _ = await llm_stage(model, prompt, ctx)
        s0 = time.perf_counter()
        if not res.get("error"):
            await kv_set(key, {k: res[k] for k in ("answer", "confidence", "model", "simulated")})
        stages += llm_stages + [{"name": "Redis SET", "kind": "cache", "ms": (time.perf_counter() - s0) * 1000}]
        return {"result": res, "stages": stages, "pii_sent": bool(redact(prompt)[1])}
    if path_id in ("jev", "jev_llm"):
        return await run_jev_path(path_id, model, prompt, ctx)
    url, features = {
        "gateway": ("/svc/gateway", {}),
        "gateway_jev": ("/svc/gateway", {"jev_first": True}),
        "platform": ("/svc/runtime", {"cache": True, "semantic": True, "router": True}),
    }[path_id]
    payload, hop_ms = await hop(url, {**body, "features": features}, rtt)
    payload["stages"] = [{"name": "Network hop", "kind": "hop", "ms": hop_ms}] + payload["stages"]
    payload["pii_sent"] = False
    return payload


async def run_jev_path(path_id, model, prompt, ctx):
    pii_sent = bool(redact(prompt)[1])  # Jev is an external API too
    item = graded_item(ctx)
    if not item or not item.get("options"):  # Jev can't write text, so it picks the model tier instead
        use, stage, jev_tokens = await jev_route(model, prompt, ctx)
        res, llm_stages, _ = await llm_stage(use, prompt, ctx)
        return {"result": res, "stages": [stage] + llm_stages, "jev_in_tokens": jev_tokens,
                "routed_to": use if use != model else None, "pii_sent": pii_sent}
    r = await call_jev(prompt, jev_answer_question(item["options"]),
                       truth={"answer": item["sim_right"]}, difficulty=item["difficulty"],
                       force_sim=ctx["force_sim"], seed=ctx["pass"])
    if r["error"]:
        stages = [jev_stage("Jev error", r)]
        if path_id == "jev":
            return {"result": {"error": r["error"]}, "stages": stages, "pii_sent": pii_sent}
        conf = None
    else:
        a = r["answers"]["answer"]
        conf = a["confidence"]
        stages = [jev_stage(f"Jev · choice {a['choice']} ({conf:.2f})", r)]
        if path_id == "jev" or conf >= ctx["jev_threshold"]:
            return {"result": {"answer": a["choice"], "confidence": conf * 100, "in_tokens": 0, "out_tokens": 0,
                               "model": r["model"], "simulated": r["simulated"]},
                    "stages": stages, "jev_in_tokens": r["in_tokens"], "decided_by": "jev", "pii_sent": pii_sent}
    res, llm_stages, _ = await llm_stage(model, prompt, ctx)  # System 2: escalate to the LLM
    return {"result": res, "stages": stages + llm_stages, "jev_in_tokens": r["in_tokens"],
            "escalated": True, "jev_confidence": conf, "pii_sent": pii_sent}


def grade(answer, accept):
    return grading.grade(answer, accept)


def hallucinated(item, answer, confidence, correct):
    """A wrong answer given as if it were right: made up for an unanswerable question, outside the allowed
    labels, or stated with at least 70% confidence."""
    if not item or not item.get("accept") or correct is None or correct:
        return None if correct is None else False
    if item.get("unanswerable"):
        return True
    if item.get("options") and grading.first_label(answer, item["options"]) is None:
        return True
    return confidence is not None and confidence >= 70


def normalize_items(raw):
    """Turn user test cases into graded items the simulator and Jev can use."""
    items = []
    for i, r in enumerate(raw[:60]):
        text = str(r.get("input", "")).strip()
        expected = str(r.get("expected", "")).strip()
        if not text:
            continue
        options = [str(o).strip() for o in (r.get("options") or []) if str(o).strip()] or None
        unanswerable = expected.lower() in catalog.UNKNOWN_ANSWERS
        # "|" separates alternative answers, unless the expected answer is a field list ("a: x | b: y"),
        # which older specs wrote with "|": then it's one answer with several fields.
        alts = [a.strip() for a in expected.split("|") if a.strip()]
        if len(alts) > 1 and all(re.match(r"^[\w .-]{1,40}?\s*[:=]", a) for a in alts):
            alts = ["; ".join(alts)]
        accept = (catalog.UNKNOWN_ANSWERS if unanswerable else alts) or None
        prompt = text + (f"\n\nAnswer with exactly one of: {', '.join(options)}." if options else "")
        if unanswerable or (accept and r.get("grounded")):
            difficulty = 3 if unanswerable else 2
        else:
            difficulty = 1 if options and len(text.split()) < 60 else (2 if is_complex(text) else 1)
        if options and accept:
            right = next((o for o in options if o.lower() == accept[0].lower()), accept[0])
            wrong = next((o for o in options if o.lower() != right.lower()), "other")
        else:
            right = "unknown" if unanswerable else (accept[0] if accept else None)
            wrong = "(simulated made-up answer)" if unanswerable else "(simulated wrong answer)"
        items.append({"id": f"e{i + 1}", "q": prompt, "input": text, "accept": accept, "options": options,
                      "difficulty": difficulty, "sim_right": right, "sim_wrong": wrong, "unanswerable": unanswerable})
    return items


class RunReq(BaseModel):
    model: str
    workload: str = "suite"
    prompt: str = ""
    passes: int = 2
    paths: list = [p["id"] for p in catalog.PATHS]
    force_sim: bool = False
    fault_rate: float = 0.0
    assumptions: dict = {}
    cache_scope: str = ""  # set by the Ask page so caches persist across questions in one browser session
    items: list = []  # workload "examples": [{input, expected, options?, grounded?}] from the Architecture Advisor


@app.post("/api/run")
async def api_run(req: RunReq, request: Request):
    allowed, why = policy.check_model(req.model) if req.model in catalog.MODEL_BY_ID else (False, "unknown model")
    if not allowed:
        audit.log("policy_blocked", request, model=req.model, reason=why)
        return JSONResponse({"error": f"{req.model} is blocked by company policy: {why}"}, status_code=403)
    audit.log("ask_question" if req.workload == "custom" and req.cache_scope else "benchmark_run", request,
              model=req.model, workload=req.workload, prompt=req.prompt if req.workload == "custom" else None,
              routes=req.paths, passes=req.passes, force_sim=req.force_sim, outage_test=bool(req.fault_rate))
    a = {**catalog.DEFAULT_ASSUMPTIONS, **req.assumptions}
    run_id = uuid.uuid4().hex[:8]
    scope = re.sub(r"[^a-zA-Z0-9]", "", req.cache_scope)[:32]
    cache_ns = f"s{scope}" if scope else run_id
    if req.workload == "suite":
        items = catalog.suite_items()
    elif req.workload == "examples":
        items = normalize_items(req.items)
    else:
        items = [{"id": "custom", "q": req.prompt, "accept": None}]
    if not items:
        items = [{"id": "custom", "q": req.prompt or "Hello", "accept": None}]
    queue: asyncio.Queue = asyncio.Queue()
    live = has_key(catalog.MODEL_BY_ID[req.model]["provider"]) and not req.force_sim
    # Jev alone can only pick from options, so it's a candidate only when every test case has them
    paths = [p for p in req.paths if not (p == "jev" and not all(i.get("options") and i.get("accept") for i in items))]
    if not policy.service_allowed("jev"):  # Jev-based routes need an approved Jev vendor
        paths = [p for p in paths if p not in ("jev", "jev_llm", "gateway_jev")]

    async def worker(path_id):
        for pas in range(1, max(1, min(req.passes, 5)) + 1):
            for item in items:
                ctx = {"run_id": run_id, "cache_ns": cache_ns, "path": path_id, "qid": item["id"], "pass": pas,
                       "item": item,
                       "app_id": "demo-app", "force_sim": req.force_sim, "fault_rate": req.fault_rate,
                       "hop_rtt_ms": float(a["hop_rtt_ms"]), "semantic_threshold": float(a["semantic_threshold"]),
                       "runtime_context_tokens": int(a["runtime_context_tokens"]),
                       "jev_threshold": float(a["jev_threshold"])}
                try:
                    out = await run_path(path_id, req.model, item["q"], ctx)
                except Exception as exc:
                    out = {"result": {"error": f"{type(exc).__name__}: {exc}"[:200]}, "stages": []}
                res = out["result"]
                correct = (grade(res.get("answer"), item["accept"])
                           if item.get("accept") and not res.get("error") else None)
                await queue.put({
                    "type": "call", "path": path_id, "qid": item["id"], "pass": pas,
                    "answer": res.get("answer", ""), "confidence": res.get("confidence"),
                    "correct": correct,
                    "hallucinated": hallucinated(item, res.get("answer"), res.get("confidence"), correct),
                    "error": res.get("error"), "model": res.get("model", req.model),
                    "in_tokens": res.get("in_tokens", 0), "out_tokens": res.get("out_tokens", 0),
                    "simulated": res.get("simulated", True), "failover": res.get("failover", False),
                    "cache": out.get("cache"), "routed_to": out.get("routed_to"),
                    "pii_redacted": out.get("pii_redacted", 0), "pii_sent": out.get("pii_sent", False),
                    "jev_in_tokens": out.get("jev_in_tokens", 0), "decided_by": out.get("decided_by"),
                    "escalated": out.get("escalated", False),
                    "stages": out["stages"], "latency_ms": sum(s["ms"] for s in out["stages"]),
                })

    async def stream():
        yield json.dumps({"type": "meta", "run_id": run_id, "live": live, "cache_backend": await cache_backend(),
                          "jev_live": has_jev_key() and not req.force_sim, "paths": paths,
                          "items": items, "assumptions": a,
                          "total": len(items) * max(1, min(req.passes, 5)) * len(paths)}) + "\n"
        tasks = [asyncio.create_task(worker(p)) for p in paths]
        done = asyncio.gather(*tasks)
        while not (done.done() and queue.empty()):
            try:
                yield json.dumps(await asyncio.wait_for(queue.get(), 0.2)) + "\n"
            except asyncio.TimeoutError:
                continue
        if not scope:
            for ns in [k for k in _semantic_index if k.startswith(run_id)]:
                del _semantic_index[ns]
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class AdviseReq(BaseModel):
    goal: str = ""
    spec: dict | None = None  # an edited spec from the page; skips the understanding step
    force_sim: bool = False
    harness: bool = False  # wrap each architecture in production controls, priced into the numbers
    constraints: dict | None = None  # data class, region, images, existing platform pieces, commitment, engineer-week rate
    fresh: bool = False  # skip the result cache
    multi_model: bool = True  # allow a different model per role (router, workers, escalation)
    compare: bool = False  # test every eligible model (rate-limited), instead of the first-run shortlist
    priorities: list[str] = []  # what matters most: cost, accuracy, speed, simplicity (any combination)


# Advice is computed on the fly and never stored per user. Finished results are kept in memory for a while, keyed
# by the question and the knowledge they were computed with, so asking again (or reopening the link) is instant.
# Nothing is written to disk; a restart clears it.
ADVICE_TTL = int(os.environ.get("ADVICE_CACHE_TTL", 24 * 3600))
ADVICE_MAX = int(os.environ.get("ADVICE_CACHE_MAX", 200))
_advice_cache: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
CONSTRAINT_KEYS = ("data_class", "residency", "needs_images", "existing", "commitment", "engineer_week_usd", "peak_factor",
                   "tool_count", "integration", "mcp_servers", "tool_calls_per_request", "api_latency_ms", "steps_known",
                   "request_types", "specialists", "multi_hop", "docs_change_often", "structured_data")


def _knowledge_version():
    blob = json.dumps([[m["id"], m["in"], m["out"], m.get("status")] for m in catalog.MODELS]
                      + [policy.POLICY, advisor.HARNESS, constraints.CAPS, constraints.TCO, constraints.QUOTAS], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _advice_key(req, session_id=None):
    # Scoped to the browser session: a new browser session always gets a fresh run; the same session reuses it.
    body = {"sid": session_id, "cmp": req.compare, "goal": " ".join(req.goal.lower().split()), "spec": req.spec, "harness": req.harness, "sim": req.force_sim, "mm": req.multi_model,
            "pr": sorted(set(req.priorities)), "constraints": req.constraints or {}, "k": _knowledge_version()}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:32]


# Stability within a browser session: the same goal keeps the same spec and test cases, and a model's answer to
# a test case is measured once and reused (toggles, what-ifs and the full comparison re-rank the same evidence).
_spec_pins: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
_measured: "collections.OrderedDict[str, dict]" = collections.OrderedDict()


def _pin_key(session_id, goal, force_sim):
    return hashlib.sha256(json.dumps([session_id, " ".join(goal.lower().split()), force_sim]).encode()).hexdigest()


def _lru_put(store, key, value, limit):
    store[key] = {"at": time.time(), "v": value}
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)


def _lru_get(store, key):
    hit = store.get(key)
    if hit and time.time() - hit["at"] < ADVICE_TTL:
        return hit["v"]
    store.pop(key, None)
    return None


# Full model comparisons are limited per client IP in a rolling 24 hours (in memory; a restart resets it).
COMPARE_LIMIT = int(os.environ.get("COMPARE_LIMIT_PER_DAY", 5))
TRUSTED_PROXY_HOPS = int(os.environ.get("TRUSTED_PROXY_HOPS", 1))  # App Runner's load balancer appends the real client IP
_compare_log: dict = {}
_provider_down: dict = {}  # provider -> time its live calls last all failed; skipped in shortlists for 10 minutes


def client_ip(request: Request) -> str:
    """The caller's IP: the entry our own proxy appended to X-Forwarded-For (clients can forge the rest)."""
    xff = [x.strip() for x in request.headers.get("x-forwarded-for", "").split(",") if x.strip()]
    if TRUSTED_PROXY_HOPS and len(xff) >= TRUSTED_PROXY_HOPS:
        return xff[-TRUSTED_PROXY_HOPS]
    return request.client.host if request.client else "unknown"


def compare_quota(ip: str) -> dict:
    now = time.time()
    runs = [t for t in _compare_log.get(ip, []) if now - t < 24 * 3600]
    _compare_log[ip] = runs
    return {"limit": COMPARE_LIMIT, "used": len(runs), "left": max(0, COMPARE_LIMIT - len(runs)),
            "resets_in_s": int(24 * 3600 - (now - runs[0])) if runs else None}


@app.get("/api/compare-quota")
async def api_compare_quota(request: Request):
    return compare_quota(client_ip(request))


def _cache_get(key):
    hit = _advice_cache.get(key)
    if hit and time.time() - hit["at"] < ADVICE_TTL:
        return hit
    _advice_cache.pop(key, None)
    return None


def _cache_put(key, spec_event, advice):
    _advice_cache[key] = {"at": time.time(), "spec": spec_event, "advice": advice}
    _advice_cache.move_to_end(key)
    while len(_advice_cache) > ADVICE_MAX:
        _advice_cache.popitem(last=False)


@app.get("/api/advice/{key}")
async def api_advice_cached(key: str, request: Request):
    hit = _cache_get(key)
    if not hit:
        return JSONResponse({"error": "This result has expired. Run the advice again."}, status_code=404)
    audit.log("advice_reopened", request, key=key)
    return {**hit["spec"], "advice": {**hit["advice"], "cached_at": hit["at"]}}


@app.post("/api/advise")
async def api_advise(req: AdviseReq, request: Request):
    """Understand the goal, write test cases, test the models on them, and rank architectures."""
    queue: asyncio.Queue = asyncio.Queue()
    started = time.perf_counter()
    audit.log("advise_requested", request, goal=req.goal, edited=bool(req.spec), force_sim=req.force_sim, harness=req.harness)
    key = _advice_key(req, request.state.session_id)
    hit = None if req.fresh else _cache_get(key)
    ip = client_ip(request)
    if req.compare and not hit:  # a cached comparison doesn't use up the quota
        q = compare_quota(ip)
        if not q["left"]:
            audit.log("compare_limited", request, goal=req.goal, ip=ip)
            return JSONResponse({"error": f"You've used all {COMPARE_LIMIT} full comparisons for today. More become available in "
                                          f"{round(q['resets_in_s'] / 3600, 1)} hours.", "quota": q}, status_code=429)
        _compare_log.setdefault(ip, []).append(time.time())
    if hit:
        audit.log("advise_cache_hit", request, goal=req.goal, key=key)

        async def cached():
            yield json.dumps(hit["spec"]) + "\n"
            yield json.dumps({**hit["advice"], "cached_at": hit["at"]}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        return StreamingResponse(cached(), media_type="application/x-ndjson")

    async def work():
        await queue.put({"type": "stage", "stage": "understand"})
        note = None
        pin = _pin_key(request.state.session_id, req.goal, req.force_sim) if req.goal and not req.spec else None
        pinned = _lru_get(_spec_pins, pin) if pin else None
        if req.spec:
            spec, source = req.spec, "edited"
        elif pinned:  # same goal earlier in this browser session: same spec and test cases, no new architect call
            spec, source = json.loads(json.dumps(pinned)), "pinned"
        elif advisor.claude_available() and not req.force_sim:
            try:
                spec, source = await advisor.spec_from_claude(req.goal), "claude"
            except Exception as exc:  # fall back to rules rather than failing the page
                spec, source = advisor.spec_from_rules(req.goal), "rules"
                note = f"The Claude architect failed ({type(exc).__name__}: {str(exc)[:160]}); used built-in rules instead."
        else:
            spec, source = advisor.spec_from_rules(req.goal), "rules"
        arch_usage = spec.pop("_usage", None) if isinstance(spec, dict) else None
        if pin and not pinned:
            _lru_put(_spec_pins, pin, spec, 2000)
        spec = {**spec, **{k: v for k, v in (req.constraints or {}).items() if k in CONSTRAINT_KEYS and v not in (None, "")}}
        spec = advisor.finalize_spec({**spec, "multi_model": req.multi_model, "priorities": req.priorities})
        spec["harness"] = req.harness
        items = normalize_items(advisor.items_for(spec))
        confirm = advisor.cross_check(spec, req.goal) if source in ("claude", "pinned") and req.goal else []
        use_judge = grading.judge_allowed(spec) and not req.force_sim
        judge_usage = []
        spec_event = {"type": "spec", "spec": spec, "goal": req.goal, "source": "compare" if req.compare else source, "note": note, "confirm": confirm,
                      "architect_model": advisor.ARCHITECT_MODEL if source == "claude" else None}
        await queue.put(spec_event)

        live_providers = [p for p in catalog.PROVIDERS if has_key(p)] if not req.force_sim else []
        # Every callable offering is tested: live where this server has access, simulated elsewhere (labelled per row).
        # Only approved models are tested, so test cases never reach an unapproved vendor.
        # Models that can't take this data class or region, or lack a capability the agent needs, aren't tested at all.
        models, excluded = constraints.screen([m["id"] for m in catalog.CALLABLE if policy.model_allowed(m["id"])], spec)
        if not models:
            raise ValueError("No approved model can take this data class and region. Relax the data constraints.")
        eligible, all_eligible = len(models), list(models)
        if not req.compare:  # first run: a small representative shortlist; the full comparison is on request
            live_ok = lambda m: (not req.force_sim and has_key(catalog.MODEL_BY_ID[m]["provider"])  # noqa: E731
                                 and time.time() - _provider_down.get(catalog.MODEL_BY_ID[m]["provider"], 0) > 600)
            models = constraints.shortlist(models, spec, live_ok)
        use_jev = (policy.service_allowed("jev") and bool(spec["labels"])
                   and all(i.get("options") and i.get("accept") for i in items))
        jev_blocked = bool(spec["labels"]) and not policy.service_allowed("jev")
        total = len(items) * (len(models) + (1 if use_jev else 0))
        await queue.put({"type": "stage", "stage": "test", "models": models, "cases": len(items), "total": total,
                         "live": bool(live_providers), "jev": use_jev, "jev_live": has_jev_key() and not req.force_sim})
        sem, done = asyncio.Semaphore(8), [0]

        async def one_llm(model, item, force_sim=req.force_sim):
            mkey = hashlib.sha256(json.dumps([request.state.session_id, model, item["q"], item.get("accept"), force_sim]).encode()).hexdigest()
            prior = _lru_get(_measured, mkey)
            if prior:  # measured earlier in this session: reuse it, so re-runs rank the same evidence
                done[0] += 1
                await queue.put({"type": "progress", "done": done[0], "total": total})
                return {**prior, "reused": True}
            async with sem:
                r = await call_llm(model, item["q"], item=item if item.get("accept") else None, force_sim=force_sim)
            correct, method = (None, None)
            if item.get("accept"):
                if r["error"]:
                    correct, method = False, "error"
                else:
                    correct, method, needs_judge = grading.grade_item(r["answer"], item)
                    if needs_judge and use_judge and not r["simulated"]:
                        try:
                            correct, usage = await grading.judge(item["input"], item["accept"], r["answer"])
                            method = "judge"
                            if usage:
                                judge_usage.append(usage)
                        except Exception:  # keep the keyword grade if the judge is unavailable
                            pass
            done[0] += 1
            await queue.put({"type": "progress", "done": done[0], "total": total})
            row = {"answer": r["answer"], "confidence": r["confidence"], "correct": correct, "graded_by": method,
                   "hallucinated": hallucinated(item, r["answer"], r["confidence"], correct) if not r["error"] else False,
                   "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"], "latency_ms": r["llm_ms"],
                   "error": r["error"], "simulated": r["simulated"]}
            if not r["error"]:
                _lru_put(_measured, mkey, row, 50000)
            return row

        async def one_jev(item):
            async with sem:
                j = await call_jev(item["q"], jev_answer_question(item["options"]), truth={"answer": item["sim_right"]},
                                   difficulty=item["difficulty"], force_sim=req.force_sim)
            done[0] += 1
            await queue.put({"type": "progress", "done": done[0], "total": total})
            if j["error"]:
                return {"answer": "", "confidence": None, "correct": False, "hallucinated": False, "in_tokens": 0,
                        "latency_ms": j["jev_ms"], "error": j["error"]}
            a = j["answers"]["answer"]
            correct = grade(a["choice"], item["accept"])
            return {"answer": a["choice"], "confidence": a["confidence"] * 100, "correct": correct,
                    "hallucinated": hallucinated(item, a["choice"], a["confidence"] * 100, correct),
                    "in_tokens": j["in_tokens"], "latency_ms": j["jev_ms"], "error": None, "simulated": j["simulated"]}

        per_model = await asyncio.gather(*[asyncio.gather(*[one_llm(m, it) for it in items]) for m in models])
        res = dict(zip(models, per_model))
        # A model whose live calls all failed can't be reached from here: it's simulated instead, and labelled.
        unavailable = {m: next(r["error"] for r in rs) for m, rs in res.items() if rs and all(r["error"] for r in rs)}
        for m in unavailable:
            res[m] = list(await asyncio.gather(*[one_llm(m, it, force_sim=True) for it in items]))
        mode = {m: "simulated" if all(r["simulated"] for r in rs) else "live" for m, rs in res.items()}
        for m in unavailable:
            _provider_down[catalog.MODEL_BY_ID[m]["provider"]] = time.time()
        jev = list(await asyncio.gather(*[one_jev(it) for it in items])) if use_jev else None
        await queue.put({"type": "stage", "stage": "rank"})
        result = advisor.rank_designs(spec, res, jev, mode, jev_blocked=jev_blocked)
        result["harness"] = advisor.harness_advice(spec)
        result["run_cost"] = advisor.run_cost(arch_usage, res, mode, "full" if req.compare else "shortlist", judge_usage)
        graded = [r.get("graded_by") for rs in res.values() for r in rs if r.get("graded_by")]
        result["grading"] = {m: graded.count(m) for m in ("exact", "keyword", "judge", "error") if graded.count(m)}
        result["grading"]["judge_on"] = use_judge
        result["stability"] = {"spec": source, "reused": sum(1 for rs in res.values() for r in rs if r.get("reused")),
                               "measured": sum(len(rs) for rs in res.values())}
        if not req.compare:
            result["run_cost"]["compare_estimate"] = advisor.compare_estimate(res, all_eligible)
        result["robustness"] = advisor.robustness(spec, res, jev, mode, jev_blocked, result)
        result["excluded_models"] = excluded
        result["policy"] = {"name": policy.POLICY["name"], "approved_platforms": sorted(policy.APPROVED_PLATFORMS),
                            "excluded": [e for e in policy.summary()["excluded"]]}
        for row in result["model_table"]:
            if row["id"] in unavailable:
                row["mode"], row["live_error"] = "simulated", str(unavailable[row["id"]])[:160]
        result["unavailable"] = [{"id": m, "label": catalog.MODEL_BY_ID[m]["label"], "platform": catalog.MODEL_BY_ID[m]["platform"],
                                  "error": str(e)[:160]} for m, e in unavailable.items()]
        status = {m["id"]: m["status"] for m in catalog.MODELS}
        for d in result["all"]:
            d["unverified_models"] = sorted({m for m in d["models"].values() if m and status.get(m) != "verified"})
        audit.log("advise_completed", request, goal=req.goal, spec_source=source, task_type=spec["task_type"],
                  labels=spec["labels"], latency=spec["latency"], requests_per_day=spec["requests_per_day"],
                  risk=spec["risk"], harness=req.harness, cases=len(items), models_tested=list(res), models_live=[m for m, v in mode.items() if v == "live"],
                  models_unavailable=list(unavailable),
                  needs_confirmation=[c["field"] for c in confirm],
                  top3=[{"rank": i + 1, "architecture": d["arch"], "model": d["models"]["primary"],
                         "fallback": d["models"]["fallback"], "small": d["models"]["small"],
                         "accuracy": d["accuracy"], "p95_ms": round(d["p95_ms"]), "monthly_usd": round(d["monthly"], 2),
                         "meets_requirements": d["passes"]} for i, d in enumerate(result["top"])],
                  duration_ms=round((time.perf_counter() - started) * 1000))
        advice = {"type": "advice", **result, "evaluated_models": models, "key": key,
                  "scope": "full" if req.compare else "shortlist", "eligible_models": eligible, "compare_quota": compare_quota(ip),
                         "simulated": any(v == "simulated" for v in mode.values()), "mode": mode,
                         "live_count": sum(v == "live" for v in mode.values()),
                         "cases": [{"input": c["input"], "expected": c["expected"]} for c in spec["test_cases"]],
                         "per_case": {m: [{k: r.get(k) for k in ("answer", "correct", "hallucinated", "latency_ms", "graded_by")} for r in rs]
                                      for m, rs in res.items()},
                         "jev_per_case": [{k: r[k] for k in ("answer", "correct", "confidence")} for r in jev] if jev else None}
        _cache_put(key, spec_event, advice)
        await queue.put(advice)

    async def stream():
        task = asyncio.create_task(work())
        while not (task.done() and queue.empty()):
            try:
                yield json.dumps(await asyncio.wait_for(queue.get(), 0.2)) + "\n"
            except asyncio.TimeoutError:
                continue
        if task.exception():
            audit.log("advise_failed", request, goal=req.goal, error=f"{type(task.exception()).__name__}: {task.exception()}")
            yield json.dumps({"type": "error", "error": f"{type(task.exception()).__name__}: {task.exception()}"}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class ClientEvent(BaseModel):
    event: str
    data: dict = {}


@app.post("/api/events")
async def api_events(ev: ClientEvent, request: Request):
    """Browser-side actions worth auditing (downloads, re-runs). Unknown event names are ignored."""
    if ev.event in audit.ALLOWED_CLIENT_EVENTS:
        audit.log(ev.event, request, **{k: v for k, v in list(ev.data.items())[:20]})
    return {"ok": True}


@app.get("/api/policy")
async def api_policy():
    return policy.summary()


@app.get("/api/me")
async def api_me(request: Request):
    """The signed-in user as the SSO layer reports them (anonymous until SSO is in front of the app)."""
    who = audit.identity(request.headers)
    return {"signed_in": who["id"] != "anonymous", "name": who.get("name"), "email": who.get("email"), "id": who["id"],
            "logout_url": os.environ.get("LOGOUT_URL")}


@app.get("/api/news")
async def api_news(request: Request, refresh: bool = False):
    data = await news.get_news(refresh)
    audit.log("news_viewed", request, refresh=refresh, items=len(data["items"]), not_in_catalog=data["not_in_catalog"])
    return data


@app.get("/api/knowledge")
async def api_knowledge():
    return {**advisor.knowledge_status(), "last_price_sync": pricing_sync.LAST_SYNC or None}


@app.post("/api/knowledge/sync-prices")
async def api_sync_prices(request: Request):
    report = await pricing_sync.sync()
    audit.log("prices_synced", request, sources=report["sources"], changes=report["changes"])
    return report


NON_CHAT = re.compile(r"embed|tts|whisper|dall-e|image|audio|moderation|realtime|transcri|search|computer|veo|imagen|aqa|speech|video", re.I)


@app.post("/api/knowledge/check")
async def api_knowledge_check(request: Request):
    """Compare the catalog with each provider's live model list: flags retired IDs and new ones to review."""
    out = {}
    for provider, info in catalog.PROVIDERS.items():
        ours = [m["id"] for m in catalog.MODELS if m["provider"] == provider and m.get("callable", True)]
        if provider in ("bedrock", "azure", "vertex"):
            out[provider] = {"checked": False, "reason": "prices come from the price sync; model IDs are resolved when called"}
            continue
        if not has_key(provider):
            out[provider] = {"checked": False, "reason": f"{info['env']} not set"}
            continue
        try:
            key = os.environ[info["env"]]
            if provider == "anthropic":
                client = __import__("anthropic").AsyncAnthropic()
                live = [m.id async for m in client.models.list(limit=1000)]
            elif provider == "google":
                r = await _http.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
                                    headers={"x-goog-api-key": key})
                r.raise_for_status()
                live = [m["name"].split("/", 1)[-1] for m in r.json().get("models", [])]
            else:
                base = "https://api.openai.com/v1" if provider == "openai" else "https://api.x.ai/v1"
                r = await _http.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                live = [m["id"] for m in r.json().get("data", [])]
        except Exception as exc:
            out[provider] = {"checked": False, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
            continue
        out[provider] = {"checked": True, "missing": [m for m in ours if m not in live],
                         "new": sorted(m for m in live if m not in ours and not NON_CHAT.search(m))[:40]}
    audit.log("knowledge_checked", request, result={p: {k: v for k, v in r.items() if k != "new"} for p, r in out.items()})
    return out


@app.get("/api/config")
async def api_config():
    return {
        "models": catalog.MODELS, "providers": catalog.PROVIDERS, "paths": catalog.PATHS,
        "capabilities": catalog.CAPABILITIES, "path_caps": catalog.PATH_CAPS,
        "assumptions": catalog.DEFAULT_ASSUMPTIONS, "suite_size": len(catalog.SUITE),
        "keys": {p: has_key(p) for p in catalog.PROVIDERS}, "cache_backend": await cache_backend(),
        "jev": {**catalog.JEV, "has_key": has_jev_key()}, "options": catalog.OPTIONS,
        "architect": {"claude": advisor.claude_available(), "model": advisor.ARCHITECT_MODEL},
        "policy": policy.summary(), "approved_models": [m["id"] for m in catalog.MODELS if policy.model_allowed(m["id"])],
        "jev_allowed": policy.service_allowed("jev"),
    }


STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def advisor_page():
    return FileResponse(os.path.join(STATIC, "advisor.html"))


@app.get("/benchmark")
async def benchmark_page():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/news")
async def news_page():
    return FileResponse(os.path.join(STATIC, "news.html"))


@app.get("/ask")
async def ask_page():
    return FileResponse(os.path.join(STATIC, "ask.html"))
