"""LLM Path Lab — compare ways of routing an agent's LLM call.

One FastAPI process plays every role so the comparison needs no extra infra:
  * the "agent" (the /api/run orchestrator), which can consult Jev (TypeSafe's System One model),
  * an AI Gateway and an Agent Runtime (the /svc/* endpoints),
    reached over real HTTP hops so their serialization + network cost is measured,
  * Redis (real, if reachable; otherwise an in-process stand-in, labeled as such).

Run:  uvicorn app:app --port 8088   then open http://localhost:8088
"""
import asyncio
import hashlib
import json
import os
import random
import re
import time
import uuid

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import catalog
from providers import ROUTE_QUESTION, call_jev, call_llm, has_jev_key, has_key, jev_answer_question

app = FastAPI(title="LLM Path Lab")
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
PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "<CARD>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
]
INJECTION = re.compile(r"ignore (all|previous|prior) instructions|reveal (the )?system prompt", re.I)


def redact(prompt):
    count = 0
    for pat, repl in PII_PATTERNS:
        prompt, n = pat.subn(repl, prompt)
        count += n
    return prompt, count


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
                attempt_model = catalog.PROVIDERS[catalog.MODEL_BY_ID[model]["provider"]]["fallback"]
                continue
            return ({"error": "provider 503 (injected fault)", "answer": "", "confidence": None,
                     "in_tokens": 0, "out_tokens": 0, "model": model, "simulated": True},
                    stages, (time.perf_counter() - t0) * 1000)
        res = await call_llm(attempt_model, prompt, qid=ctx["qid"], extra_context_tokens=extra_tokens,
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


def suite_item(qid):
    return next((s for s in catalog.SUITE if s["id"] == qid), None)


def jev_stage(label, r):
    return {"name": label, "kind": "jev", "ms": r["jev_ms"], "simulated": r["simulated"]}


async def jev_route(model, prompt, ctx):
    """Ask Jev whether the provider's small model is enough. Returns (model_to_use, stage, jev_tokens)."""
    item = suite_item(ctx["qid"])
    needs_large = item["difficulty"] >= 2 if item else is_complex(prompt)
    r = await call_jev(prompt, ROUTE_QUESTION, truth={"route": "large" if needs_large else "small"},
                       force_sim=ctx["force_sim"], seed=ctx["pass"])
    if r["error"]:  # fail safe: keep the model the app asked for
        return model, jev_stage("Jev router failed → keep model", r), 0
    a = r["answers"]["route"]
    small = catalog.PROVIDERS[catalog.MODEL_BY_ID[model]["provider"]]["small"]
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
    ns = f"{ctx['run_id']}:{ctx['path']}"
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

    if f.get("router"):
        model, stage, info["jev_in_tokens"] = await jev_route(req.model, prompt, ctx)
        stages.append(stage)
        info["routed_to"] = model if model != req.model else None

    res, llm_stages, _ = await llm_stage(model, prompt, ctx, extra_tokens=f.get("extra_tokens", 0),
                                            failover=True)
    stages += llm_stages
    p0 = time.perf_counter()
    if f.get("cache") and not res.get("error"):
        payload = {k: res[k] for k in ("answer", "confidence", "model", "simulated")}
        await kv_set(exact_key(ns, req.model, prompt), payload)
        _semantic_index.setdefault(ns, []).append((tokens(prompt), payload))
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
    if path_id == "direct":
        res, stages, _ = await llm_stage(model, prompt, ctx)
        return {"result": res, "stages": stages, "pii_redacted": 0, "pii_sent": bool(redact(prompt)[1])}
    if path_id == "redis":
        c0 = time.perf_counter()
        key = exact_key(f"{ctx['run_id']}:redis", model, prompt)
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
        "platform": ("/svc/runtime", {"cache": True, "semantic": True, "router": True}),
    }[path_id]
    payload, hop_ms = await hop(url, {**body, "features": features}, rtt)
    payload["stages"] = [{"name": "Network hop", "kind": "hop", "ms": hop_ms}] + payload["stages"]
    payload["pii_sent"] = False
    return payload


async def run_jev_path(path_id, model, prompt, ctx):
    pii_sent = bool(redact(prompt)[1])  # Jev is an external API too
    item = suite_item(ctx["qid"])
    if not item:  # free-form prompt: Jev can't write the answer, so it picks the model tier instead
        use, stage, jev_tokens = await jev_route(model, prompt, ctx)
        res, llm_stages, _ = await llm_stage(use, prompt, ctx)
        return {"result": res, "stages": [stage] + llm_stages, "jev_in_tokens": jev_tokens,
                "routed_to": use if use != model else None, "pii_sent": pii_sent}
    r = await call_jev(prompt, jev_answer_question(catalog.OPTIONS[item["id"]]),
                       truth={"answer": catalog.SIM_RIGHT[item["id"]]}, difficulty=item["difficulty"],
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
    ans = (answer or "").lower()
    return any(re.search(r"(?<![\w.])" + re.escape(a) + r"(?![\w])", ans) for a in accept)


class RunReq(BaseModel):
    model: str
    workload: str = "suite"
    prompt: str = ""
    passes: int = 2
    paths: list = [p["id"] for p in catalog.PATHS]
    force_sim: bool = False
    fault_rate: float = 0.0
    assumptions: dict = {}


@app.post("/api/run")
async def api_run(req: RunReq):
    a = {**catalog.DEFAULT_ASSUMPTIONS, **req.assumptions}
    run_id = uuid.uuid4().hex[:8]
    items = catalog.SUITE if req.workload == "suite" else [{"id": "custom", "q": req.prompt, "accept": None}]
    queue: asyncio.Queue = asyncio.Queue()
    live = has_key(catalog.MODEL_BY_ID[req.model]["provider"]) and not req.force_sim
    paths = [p for p in req.paths if not (p == "jev" and req.workload != "suite")]  # Jev alone can't write free text

    async def worker(path_id):
        for pas in range(1, max(1, min(req.passes, 5)) + 1):
            for item in items:
                ctx = {"run_id": run_id, "path": path_id, "qid": item["id"], "pass": pas,
                       "app_id": "demo-app", "force_sim": req.force_sim, "fault_rate": req.fault_rate,
                       "hop_rtt_ms": float(a["hop_rtt_ms"]), "semantic_threshold": float(a["semantic_threshold"]),
                       "runtime_context_tokens": int(a["runtime_context_tokens"]),
                       "jev_threshold": float(a["jev_threshold"])}
                try:
                    out = await run_path(path_id, req.model, item["q"], ctx)
                except Exception as exc:
                    out = {"result": {"error": f"{type(exc).__name__}: {exc}"[:200]}, "stages": []}
                res = out["result"]
                await queue.put({
                    "type": "call", "path": path_id, "qid": item["id"], "pass": pas,
                    "answer": res.get("answer", ""), "confidence": res.get("confidence"),
                    "correct": (grade(res.get("answer"), item["accept"]) if item["accept"] and not res.get("error") else None),
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
        for ns in [k for k in _semantic_index if k.startswith(run_id)]:
            del _semantic_index[ns]
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/config")
async def api_config():
    return {
        "models": catalog.MODELS, "providers": catalog.PROVIDERS, "paths": catalog.PATHS,
        "capabilities": catalog.CAPABILITIES, "path_caps": catalog.PATH_CAPS,
        "assumptions": catalog.DEFAULT_ASSUMPTIONS, "suite_size": len(catalog.SUITE),
        "keys": {p: has_key(p) for p in catalog.PROVIDERS}, "cache_backend": await cache_backend(),
        "jev": {**catalog.JEV, "has_key": has_jev_key()}, "options": catalog.OPTIONS,
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
