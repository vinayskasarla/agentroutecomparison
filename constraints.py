"""What narrows or re-prices a design beyond accuracy: real prompt size and prompt caching, model capabilities,
data class and region rules, peak load against quotas, total cost of ownership, and model retirements.

Everything here reads from knowledge/*.json so owners can correct it without touching code. Values that
aren't confirmed stay None and are reported as unchecked, never guessed.
"""
import json
import math
import os
import re

import catalog
import news
import policy

_load = lambda name: json.load(open(os.path.join(catalog.KNOWLEDGE_DIR, name)))  # noqa: E731
CAPS = _load("capabilities.json")
TCO = _load("tco.json")
QUOTAS = _load("quotas.json")

# How many model calls one request makes in each architecture (for peak token load).
CALLS = {"single_call": 1, "rag": 1, "long_context": 1, "batch": 1, "workflow": 2, "evaluator_optimizer": 3,
         "tool_agent": 3, "multi_agent": 5}

DEFAULT_SYSTEM_PROMPT = {"classification": 800, "extraction": 1000, "grounded_qa": 1500, "tool_actions": 3000,
                         "conversation": 1500, "research": 2000, "summarization": 1000, "generation": 1200, "open_qa": 1000}


# ------------------------------------------------------------------ capabilities and prompt caching
def caps(model_id):
    base = catalog.base_model(model_id)
    c = {**CAPS["defaults"], **CAPS["models"].get(base, {})}
    c["known"] = base in CAPS["models"]
    return c


def cache_rule(model_id):
    m = catalog.MODEL_BY_ID[model_id]
    return next((r for r in CAPS["prompt_caching"]["rules"]
                 if r["platform"] == m["platform"] and r["maker"] == m["maker"]), None)


def screen(model_ids, spec):
    """Models that can do this job under the data rules; the rest with the reason they were ruled out."""
    keep, out = [], []
    for m in model_ids:
        ok, why = policy.check_data(m, spec["data_class"], spec["residency"])
        c = caps(m)
        if ok and spec["needs_tools"] and not c["tools"]:
            ok, why = False, "doesn't support tool calling"
        if ok and spec.get("needs_images") and not c["vision"]:
            ok, why = False, "doesn't accept images"
        if ok:
            keep.append(m)
        else:
            info = catalog.MODEL_BY_ID[m]
            out.append({"id": m, "label": info["label"], "platform": info["platform"], "reason": why})
    return keep, out


# ------------------------------------------------------------------ real prompt size
def prompt_parts(spec, arch):
    """Tokens added to every model call on top of the test case: the static part (cacheable) and the rest."""
    static = spec["system_prompt_tokens"]
    dynamic = spec["history_tokens"]
    if spec["needs_documents"]:
        if arch == "long_context":
            static += spec["document_tokens"]  # the same documents every time, so they cache too
        else:
            dynamic += spec["context_tokens"]
    return static, dynamic


def inflate(res, spec, arch, models):
    """Per-case results re-sized to the real prompt: extra input tokens, and the cached share where the
    model's platform discounts cached input."""
    static, dynamic = prompt_parts(spec, arch)
    out = {}
    for m in {x for x in models if x and x in res}:
        rule = cache_rule(m) if spec["prompt_caching"] else None
        cached = static if rule and static >= rule["min_tokens"] else 0
        out[m] = [{**r, "in_tokens": r["in_tokens"] + static + dynamic, "cached_tokens": cached,
                   "cache_mult": rule["cached_input"] if cached else 1.0,
                   "out_tokens": spec["output_tokens"] or r["out_tokens"]} for r in res[m]]
    return out


def required_context(spec, arch):
    static, dynamic = prompt_parts(spec, arch)
    return static + dynamic + 1000 + (spec["output_tokens"] or 500)


# ------------------------------------------------------------------ accuracy uncertainty
def wilson(k, n, z=1.96):
    """95% range for a success rate measured as k of n."""
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------------ peak load
def peak(d, spec, tokens_per_call):
    """Requests and tokens per minute in the busiest minute, and a check against known quotas."""
    calls = CALLS.get(d["arch"]) or (1 + (d.get("escalation_rate") or 0))
    rpm = spec["requests_per_day"] / 1440 * spec["peak_factor"]
    tpm = rpm * calls * tokens_per_call
    if d["arch"] == "batch":
        rpm = tpm = 0.0  # batch jobs are queued by the provider, not rate limited per minute
    m = d["models"]["primary"]
    lim = QUOTAS["limits"].get(m) or {}
    over = [k for k, v in (("rpm", rpm), ("tpm", tpm)) if lim.get(k) and v > lim[k]]
    plat = catalog.MODEL_BY_ID[m]["platform"]
    return {"rpm": rpm, "tpm": tpm, "calls": calls, "quota": lim or None, "over": over,
            "check_url": QUOTAS["where_to_check"].get(plat), "provisioned": QUOTAS["provisioned"].get(plat)}


# ------------------------------------------------------------------ total cost of ownership
def tco(arch, spec, harness):
    infra = []
    for key, x in TCO["infra"].items():
        if arch not in x["applies"]:
            continue
        if "monthly" in x:
            reuse = key in spec["existing"]
            infra.append({"id": key, "name": x["name"], "monthly": 0.0 if reuse else x["monthly"],
                          "basis": "you already run one" if reuse else x["basis"], "source": x["source"]})
    lo, hi = TCO["build_weeks"][arch]
    if harness:
        lo, hi = lo + TCO["harness_build_weeks"][0], hi + TCO["harness_build_weeks"][1]
    rate = spec.get("engineer_week_usd") or TCO.get("engineer_week_usd")
    return {"infra": infra, "infra_monthly": sum(i["monthly"] for i in infra), "build_weeks": [lo, hi],
            "build_usd": [lo * rate, hi * rate] if rate else None}


def embedding_cost_per_req(arch, spec):
    e = TCO["infra"]["embeddings"]
    return 60 * e["per_1m_tokens"] / 1e6 if arch in e["applies"] else 0.0  # ~60-token query, embedded once


# ------------------------------------------------------------------ retirements
def _norm(s):
    return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()


def retirement(model_id):
    """A retirement date from models.json, or a retirement notice in the (already fetched) news feed."""
    m = catalog.MODEL_BY_ID[model_id]
    if m.get("retires_on"):
        return {"date": m["retires_on"], "title": f"{m['label']} retires on {m['retires_on']}", "link": m.get("source")}
    data = news.CACHE.get("data")
    if not data:
        return None
    names = {_norm(m["label"]), _norm(catalog.MODEL_BY_ID[catalog.base_model(model_id)]["label"])}
    for item in data["items"]:
        if item["category"] == "retirement" and any(_norm(x["name"]) in names for x in item["models"]):
            return {"date": item["date"], "title": item["title"], "link": item["link"]}
    return None
