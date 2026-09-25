"""LLM calls: real provider APIs when a key is present, a calibrated simulator otherwise.

Every call returns the same shape so the pipelines don't care which one ran:
    {text, answer, confidence, in_tokens, out_tokens, llm_ms, simulated, model, error}
"""
import hashlib
import json
import os
import random
import re
import time

import anthropic
import httpx

import policy

import asyncio

from catalog import JEV, MODEL_BY_ID, PROVIDERS, base_model, sim_profile

# Secrets created by the Terraform setup start as this placeholder until you store a real value.
# Treat them as unset so that provider runs in simulated mode instead of failing auth.
PLACEHOLDER = "NOT_SET"
for _name in [k for k, v in os.environ.items() if v.strip() == PLACEHOLDER]:
    del os.environ[_name]

SYSTEM = (
    "Answer the user's question. Respond with ONLY a JSON object: "
    '{"answer": "<your answer: just the final value for a quiz question, a few sentences otherwise>", '
    '"confidence": <0-100 integer, how sure you are>}'
)

_http = httpx.AsyncClient(timeout=90)
_anthropic = None


_aws_creds = None


def _aws_available() -> bool:
    global _aws_creds
    if _aws_creds is None:
        try:
            import boto3
            _aws_creds = boto3.Session().get_credentials() is not None
        except Exception:
            _aws_creds = False
    return _aws_creds


def has_key(provider: str) -> bool:
    """Whether this platform can be called live from here."""
    if provider == "bedrock":
        return _aws_available()
    if provider == "azure":
        return bool((os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"))
                    or (os.environ.get("AZURE_FOUNDRY_RESOURCE") and os.environ.get("AZURE_FOUNDRY_API_KEY")))
    if provider == "vertex":
        return bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    return bool(os.environ.get(PROVIDERS[provider]["env"]))


def available(model_id: str) -> bool:
    """Whether this specific model offering can be called live."""
    m = MODEL_BY_ID[model_id]
    if not m.get("callable", True) or not has_key(m["provider"]):
        return False
    if m["provider"] == "azure":
        env = ("AZURE_FOUNDRY_RESOURCE", "AZURE_FOUNDRY_API_KEY") if m["maker"] == "Anthropic" else ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
        return all(os.environ.get(e) for e in env)
    if m["provider"] == "vertex":
        return m["maker"] == "Anthropic"  # Gemini on Vertex isn't wired for live calls yet
    return True


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def parse_answer(text: str):
    """Pull {answer, confidence} out of the model's reply; tolerate prose around it."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            conf = obj.get("confidence")
            return str(obj.get("answer", "")).strip(), (float(conf) if conf is not None else None)
        except (ValueError, TypeError):
            pass
    return (text or "").strip()[:200], None


async def call_llm(model_id: str, prompt: str, *, item=None, extra_context_tokens=0,
                   force_sim=False, seed=0):
    m = MODEL_BY_ID[model_id]
    provider = m["provider"]
    allowed, why = policy.check_model(model_id)
    if not allowed:  # defence in depth: nothing reaches an unapproved vendor, even in simulation
        return {"text": "", "answer": "", "confidence": None, "in_tokens": 0, "out_tokens": 0, "llm_ms": 0.0,
                "simulated": True, "model": model_id, "error": f"blocked by company policy: {why}"}
    if force_sim or not available(model_id):
        return simulate(model_id, prompt, item=item, extra_context_tokens=extra_context_tokens, seed=seed)
    start = time.perf_counter()
    try:
        if m["maker"] == "Anthropic" and provider in ("bedrock", "azure", "vertex"):
            text, tin, tout = await _call_claude_on_cloud(m, prompt)
        elif provider == "bedrock":
            text, tin, tout = await _call_bedrock(m, prompt)
        elif provider == "azure":
            text, tin, tout = await _call_azure_openai(m, prompt)
        elif provider == "anthropic":
            text, tin, tout = await _call_anthropic(model_id, prompt)
        elif provider == "google":
            text, tin, tout = await _call_gemini(model_id, prompt)
        else:
            text, tin, tout = await _call_openai_compatible(provider, model_id, prompt)
    except Exception as exc:  # surface provider errors in the UI instead of crashing the run
        return {"text": "", "answer": "", "confidence": None, "in_tokens": 0, "out_tokens": 0,
                "llm_ms": (time.perf_counter() - start) * 1000, "simulated": False,
                "model": model_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
    answer, conf = parse_answer(text)
    return {"text": text, "answer": answer, "confidence": conf,
            "in_tokens": tin + extra_context_tokens, "out_tokens": tout,
            "llm_ms": (time.perf_counter() - start) * 1000, "simulated": False,
            "model": model_id, "error": None}


async def _call_anthropic(model_id, prompt, client=None, api_model=None):
    global _anthropic
    if client is None:
        if _anthropic is None:
            _anthropic = anthropic.AsyncAnthropic()
        client = _anthropic
    kwargs = {}
    if model_id != "claude-haiku-4-5":
        kwargs["output_config"] = {"effort": "low"}  # short factual answers; keep thinking light
    resp = await client.messages.create(
        model=api_model or model_id, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}], **kwargs,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model refused")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


async def _call_openai_compatible(provider, model_id, prompt):
    base = "https://api.openai.com/v1" if provider == "openai" else "https://api.x.ai/v1"
    body = {"model": model_id, "messages": [{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": prompt}]}
    if provider == "openai":
        body["reasoning_effort"] = "low"
    resp = await _http.post(f"{base}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {os.environ[PROVIDERS[provider]['env']]}"})
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return data["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


async def _call_gemini(model_id, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = await _http.post(url, json=body, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"].get("parts", [])
    usage = data.get("usageMetadata", {})
    out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return "".join(p.get("text", "") for p in parts), usage.get("promptTokenCount", 0), out


_cloud_clients = {}


async def _call_claude_on_cloud(m, prompt):
    """Claude on Bedrock, Azure AI Foundry or Vertex AI, through the Anthropic SDK's platform clients."""
    base = base_model(m["id"])
    if m["provider"] not in _cloud_clients:
        if m["provider"] == "bedrock":
            _cloud_clients["bedrock"] = anthropic.AsyncAnthropicBedrockMantle(aws_region=os.environ.get("AWS_REGION", "us-east-1"))
        elif m["provider"] == "azure":
            _cloud_clients["azure"] = anthropic.AsyncAnthropicFoundry(api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
                                                                      resource=os.environ["AZURE_FOUNDRY_RESOURCE"])
        else:
            _cloud_clients["vertex"] = anthropic.AsyncAnthropicVertex(project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
                                                                      region=os.environ.get("VERTEX_REGION", "global"))
    api_model = m.get("api_model") or (f"anthropic.{base}" if m["provider"] == "bedrock" else base)
    return await _call_anthropic(base, prompt, client=_cloud_clients[m["provider"]], api_model=api_model)


_bedrock_ids = {}


def _resolve_bedrock_id(m):
    """Find the Bedrock model (or inference profile) ID by model name, so IDs aren't hard-coded."""
    import boto3
    region = os.environ.get("AWS_REGION", "us-east-1")
    bedrock = boto3.client("bedrock", region_name=region)
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())  # noqa: E731
    want = norm(m.get("aws_name") or m["label"])
    found = None
    for s in bedrock.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]:
        if norm(s["modelName"]).startswith(want) or want.startswith(norm(s["modelName"])):
            found = s
            break
    if not found:
        raise RuntimeError(f"no Bedrock model named like {m.get('aws_name') or m['label']} in {region}")
    if "ON_DEMAND" in found.get("inferenceTypesSupported", []):
        return found["modelId"]
    for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]:
        if any(found["modelId"] in mm.get("modelArn", "") for mm in p.get("models", [])):
            return p["inferenceProfileId"]
    raise RuntimeError(f"{found['modelId']} needs an inference profile and none was found")


async def _call_bedrock(m, prompt):
    import boto3
    if m["id"] not in _bedrock_ids:
        _bedrock_ids[m["id"]] = m.get("bedrock_id") or await asyncio.to_thread(_resolve_bedrock_id, m)
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    resp = await asyncio.to_thread(client.converse, modelId=_bedrock_ids[m["id"]], system=[{"text": SYSTEM}],
                                   messages=[{"role": "user", "content": [{"text": prompt}]}],
                                   inferenceConfig={"maxTokens": 2000})
    text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
    return text, resp["usage"]["inputTokens"], resp["usage"]["outputTokens"]


async def _call_azure_openai(m, prompt):
    """Azure AI Foundry's OpenAI-compatible v1 endpoint; the deployment name defaults to the base model ID."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    body = {"model": m.get("azure_deployment") or base_model(m["id"]),
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    resp = await _http.post(f"{endpoint}/openai/v1/chat/completions", json=body,
                            headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]})
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return data["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def simulate(model_id, prompt, *, item=None, extra_context_tokens=0, seed=0):
    """Deterministic stand-in for a model call (no sleeping — latency is reported, not waited).

    `item` is a graded test case ({difficulty, sim_right, sim_wrong}); without one the answer is a placeholder.
    """
    prof = sim_profile(model_id)
    # seeded by the underlying model, so the same model on another cloud gives the same answers
    digest = hashlib.sha256(f"{base_model(model_id)}|{prompt}|{seed}".encode()).hexdigest()
    rng = random.Random(int(digest[:12], 16))
    if item and item.get("sim_right") is not None:
        correct = rng.random() < prof["skill"][item["difficulty"]]
        answer = item["sim_right"] if correct else item["sim_wrong"]
        confidence = rng.randint(82, 99) if correct else rng.randint(58, 92)  # models are over-confident when wrong
    else:
        answer = "(simulated answer — add an API key to see real output)"
        confidence = rng.randint(70, 95)
    out_tokens = 18 + rng.randint(0, 12)
    if MODEL_BY_ID[model_id]["tier"] != "small":
        out_tokens += rng.randint(40, 160)  # large models spend some reasoning tokens even at low effort
    in_tokens = est_tokens(SYSTEM + prompt) + extra_context_tokens
    llm_ms = prof["ttft"] * rng.lognormvariate(0, 0.18) + out_tokens / prof["tps"] * 1000
    llm_ms += extra_context_tokens * 0.05  # prefill cost of extra context
    return {"text": json.dumps({"answer": answer, "confidence": confidence}), "answer": answer,
            "confidence": float(confidence), "in_tokens": in_tokens, "out_tokens": out_tokens,
            "llm_ms": llm_ms, "simulated": True, "model": model_id, "error": None}


# ------------------------------------------------------------------ Jev
def has_jev_key() -> bool:
    return bool(os.environ.get(JEV["env"]))


def jev_answer_question(options):
    """A `choice` question asking Jev to pick the correct candidate answer."""
    return {"answer": {"type": "choice", "instructions": "Which option correctly answers the question in the state?",
                       "criteria": {o: None for o in sorted(options)}}}


ROUTE_QUESTION = {"route": {
    "type": "choice",
    "instructions": "Which model tier does this request need?",
    "criteria": {
        "small": "A simple lookup, short fact or formatting task a small fast model handles reliably.",
        "large": "Needs multi-step reasoning, arithmetic, counting, careful comparison or long-form writing.",
    },
}}


async def call_jev(state, questions, *, truth=None, difficulty=1, force_sim=False, seed=0):
    """POST /v1/systemone. `truth` is the correct label per question, used only by the simulator.

    Returns {answers, in_tokens, jev_ms, simulated, model, error}; answers follow the API shape:
    choice -> {"type": "choice", "choice", "confidence", "probabilities"}.
    """
    if not policy.service_allowed("jev"):
        return {"answers": {}, "in_tokens": 0, "jev_ms": 0.0, "simulated": True, "model": JEV["model"],
                "error": "blocked by company policy: Jev (TypeSafe) is not an approved vendor"}
    if force_sim or not has_jev_key():
        return simulate_jev(state, questions, truth or {}, difficulty, seed)
    start = time.perf_counter()
    try:
        resp = await _http.post(
            os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone",
            json={"model": os.environ.get("TYPESAFE_DEFAULT_MODEL", JEV["model"]), "state": state, "questions": questions},
            headers={"Authorization": f"Bearer {os.environ[JEV['env']]}"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {"answers": {}, "in_tokens": 0, "jev_ms": (time.perf_counter() - start) * 1000,
                "simulated": False, "model": JEV["model"], "error": f"Jev {type(exc).__name__}: {exc}"[:300]}
    return {"answers": data["answers"], "in_tokens": data.get("usage", {}).get("input_tokens", 0),
            "jev_ms": (time.perf_counter() - start) * 1000, "simulated": False, "model": data.get("model", JEV["model"]),
            "error": None}


def simulate_jev(state, questions, truth, difficulty, seed):
    """Calibrated stand-in: when Jev is wrong, its confidence is low, so a threshold catches most misses."""
    digest = hashlib.sha256(f"jev|{json.dumps(state)}|{sorted(questions)}|{seed}".encode()).hexdigest()
    rng = random.Random(int(digest[:12], 16))
    answers = {}
    for name, q in questions.items():
        labels = list(q["criteria"])
        right = truth.get(name, labels[0])
        p = JEV["skill"][difficulty] if name == "answer" else 0.93
        if rng.random() < p:
            pick, conf = right, min(0.99, max(0.55, rng.gauss(p, 0.06)))
        else:
            pick, conf = rng.choice([lab for lab in labels if lab != right] or labels), rng.uniform(0.34, 0.66)
        rest = [lab for lab in labels if lab != pick]
        weights = [rng.random() + 0.05 for _ in rest]
        probs = {pick: conf, **{lab: (1 - conf) * w / sum(weights) for lab, w in zip(rest, weights)}}
        answers[name] = {"type": "choice", "choice": pick, "confidence": round(conf, 3),
                         "probabilities": {k: round(v, 3) for k, v in probs.items()}}
    in_tokens = est_tokens(json.dumps(state) + json.dumps(questions))
    return {"answers": answers, "in_tokens": in_tokens, "simulated": True, "model": JEV["model"], "error": None,
            "jev_ms": 70 + 45 * rng.lognormvariate(0, 0.35) + in_tokens * 0.08}
