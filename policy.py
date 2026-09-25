"""Company AI vendor policy: which platforms, model makers and extra services may be used.

Enforced on the server at every entry point (evaluation, recommendations, Ask/Benchmark calls, failover and the
Jev router), so a page can't bypass it. Loaded from knowledge/policy.json, or the file in POLICY_FILE.
"""
import json
import os

import catalog

PATH = os.environ.get("POLICY_FILE") or os.path.join(catalog.KNOWLEDGE_DIR, "policy.json")
POLICY = json.load(open(PATH))
APPROVED_PLATFORMS = {p["platform"] for p in POLICY["approved_platforms"]}
BLOCKED_MAKERS = {m.lower() for m in POLICY.get("blocked_makers", [])}
APPROVED_SERVICES = set(POLICY.get("approved_services", []))


def check_model(model_id):
    """(allowed, reason) for one model offering."""
    m = catalog.MODEL_BY_ID.get(model_id)
    if not m:
        return False, "unknown model"
    if m["platform"] not in APPROVED_PLATFORMS:
        return False, f"{m['platform']} is not an approved platform"
    if m["maker"].lower() in BLOCKED_MAKERS:
        return False, f"{m['maker']} models are blocked by policy"
    return True, None


def model_allowed(model_id):
    return check_model(model_id)[0]


def service_allowed(service):
    return service in APPROVED_SERVICES


def safe_fallback(model_id):
    """A failover model the policy allows: the provider's configured fallback if approved, else the first approved
    model on a different platform."""
    m = catalog.MODEL_BY_ID[model_id]
    preferred = catalog.PROVIDERS[m["provider"]]["fallback"]
    if model_allowed(preferred) and catalog.MODEL_BY_ID[preferred]["platform"] != m["platform"]:
        return preferred
    return next((x["id"] for x in catalog.CALLABLE if model_allowed(x["id"]) and x["platform"] != m["platform"]), None)


def summary():
    excluded = [{"id": m["id"], "label": m["label"], "platform": m["platform"], "reason": check_model(m["id"])[1]}
                for m in catalog.MODELS if not model_allowed(m["id"])]
    return {"name": POLICY["name"], "owner": POLICY.get("owner"), "reviewed_on": POLICY.get("reviewed_on"),
            "approved_platforms": POLICY["approved_platforms"], "blocked_makers": sorted(BLOCKED_MAKERS),
            "services": {k: {**v, "approved": k in APPROVED_SERVICES} for k, v in POLICY.get("services", {}).items()},
            "approved_models": sum(1 for m in catalog.MODELS if model_allowed(m["id"])), "excluded": excluded,
            "data_classes": POLICY.get("data_classes", {}), "regions": POLICY.get("regions", {}),
            "data_rules_note": POLICY.get("data_rules_note")}


DATA_CLASSES = POLICY.get("data_classes", {})
REGIONS = POLICY.get("regions", {})


def check_data(model_id, data_class, residency="any"):
    """(allowed, reason) for sending this class of data to this model, in this region."""
    ok, why = check_model(model_id)
    if not ok:
        return ok, why
    plat = catalog.MODEL_BY_ID[model_id]["platform"]
    rule = DATA_CLASSES.get(data_class, {}).get("platforms", "all")
    if rule != "all" and plat not in rule:
        return False, f"{plat} isn't approved for {DATA_CLASSES[data_class]['label'].split(' (')[0].lower()} data"
    if residency and residency != "any" and residency not in REGIONS.get(plat, []):
        return False, f"{plat} can't keep processing in the {residency}"
    return True, None
