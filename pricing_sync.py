"""Keep model prices current from machine-readable sources.

- Anthropic: the official pricing page (Markdown). Covers Claude on the Anthropic API and on Azure AI Foundry,
  which Anthropic bills at standard API rates.
- Amazon Bedrock: the public AWS Price List API (on-demand text tokens, us-east-1 by default).
- Azure AI Foundry (OpenAI models): the public Azure Retail Prices API.

OpenAI, Google and xAI publish no machine-readable price list, so those rows stay 'needs_review' until an owner
confirms them in knowledge/models.json. Anything not matched keeps its reviewed value.
"""
import datetime
import json
import os
import re

import httpx

import catalog

ANTHROPIC_URL = "https://platform.claude.com/docs/en/about-claude/pricing.md"
AWS_INDEX = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/region_index.json"
AZURE_URL = "https://prices.azure.com/api/retail/prices"
LAST_SYNC: dict = {}


def _money(cell):
    m = re.search(r"\$([\d.]+)", cell or "")
    return float(m.group(1)) if m else None


async def anthropic_prices(client) -> dict:
    text = (await client.get(ANTHROPIC_URL)).raise_for_status().text
    section = text.split("## Model pricing", 1)[1].split("\n## ", 1)[0]
    out = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not cells[0].startswith("Claude"):
            continue
        name = re.sub(r"\s*\(.*?\)|<[^>]+>", "", cells[0]).strip()
        pin, pout = _money(cells[1]), _money(cells[-1])
        if pin is not None and pout is not None:
            out[name] = (pin, pout)
    return out


async def bedrock_prices(client, region="us-east-1") -> dict:
    idx = (await client.get(AWS_INDEX)).raise_for_status().json()
    offer = (await client.get("https://pricing.us-east-1.amazonaws.com" + idx["regions"][region]["currentVersionUrl"])).raise_for_status().json()
    terms = offer["terms"]["OnDemand"]
    found = {}
    for sku, p in offer["products"].items():
        a = p["attributes"]
        kind = (a.get("inferenceType") or "").lower()
        usage = a.get("usagetype", "").lower()
        if "model" not in a or a.get("feature") not in (None, "On-demand Inference"):
            continue
        if kind not in ("input tokens", "output tokens", "text input tokens", "text output tokens"):
            continue
        if any(x in usage for x in ("batch", "cache", "latency", "flex", "priority", "long")):
            continue
        for t in terms.get(sku, {}).values():
            for dim in t["priceDimensions"].values():
                per_1k = float(dim["pricePerUnit"]["USD"])
                scope = "global" if "global" in usage else "regional"
                found.setdefault(a["model"], {}).setdefault(scope, {})["in" if "input" in kind else "out"] = per_1k * 1000
    out = {}
    for name, scopes in found.items():
        s = scopes.get("global") if {"in", "out"} <= set(scopes.get("global", {})) else scopes.get("regional", {})
        if {"in", "out"} <= set(s):
            out[name] = (round(s["in"], 4), round(s["out"], 4))
    return out


async def azure_prices(client, names) -> dict:
    """Best-effort match of Azure OpenAI meters by model name (input/output, per 1K or 1M tokens)."""
    out, url, params, pages = {}, AZURE_URL, {"$filter": "serviceName eq 'Foundry Models' and armRegionName eq 'eastus'"}, 0
    items = []
    while url and pages < 30:
        r = (await client.get(url, params=params)).raise_for_status().json()
        items += r.get("Items", [])
        url, params, pages = r.get("NextPageLink"), None, pages + 1
    for name in names:
        want = re.sub(r"[^a-z0-9]", "", name.lower())
        hit = {}
        for it in items:
            meter = re.sub(r"[^a-z0-9]", "", (it.get("meterName", "") + it.get("productName", "")).lower())
            if want not in meter or any(x in meter for x in ("batch", "cached", "ft", "realtime", "audio")):
                continue
            scale = 1000 if it.get("unitOfMeasure", "").upper().startswith("1K") else 1 if it.get("unitOfMeasure", "").upper().startswith("1M") else None
            if scale is None or it.get("type") != "Consumption":
                continue
            side = "in" if re.search(r"inp|input", meter) else "out" if re.search(r"outp|output", meter) else None
            if side and side not in hit:
                hit[side] = it["retailPrice"] * scale
        if {"in", "out"} <= set(hit):
            out[name] = (round(hit["in"], 4), round(hit["out"], 4))
    return out


async def sync(write=False) -> dict:
    """Refresh prices in memory (and in knowledge/models.json when write=True). Returns what changed."""
    today = datetime.date.today().isoformat()
    report = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "sources": {}, "changes": []}
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        sources = {
            "anthropic": (anthropic_prices(client), lambda m: m.get("price_key") if m["maker"] == "Anthropic"
                          and m["platform"] in ("Anthropic API", "Azure AI Foundry") else None, ANTHROPIC_URL),
            "bedrock": (bedrock_prices(client, os.environ.get("AWS_PRICING_REGION", "us-east-1")),
                        lambda m: m.get("aws_name") if m["platform"] == "Amazon Bedrock" else None,
                        "https://pricing.us-east-1.amazonaws.com (AWS Price List API)"),
            "azure": (azure_prices(client, [m["azure_meter"] for m in catalog.MODELS if m.get("azure_meter")]),
                      lambda m: m.get("azure_meter"), AZURE_URL),
        }
        for name, (coro, key_of, src) in sources.items():
            try:
                prices = await coro
            except Exception as exc:
                report["sources"][name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                continue
            matched = 0
            for m in catalog.MODELS:
                key = key_of(m)
                if not key or key not in prices:
                    continue
                matched += 1
                new = prices[key]
                if (m["in"], m["out"]) != new:
                    report["changes"].append({"id": m["id"], "old": [m["in"], m["out"]], "new": list(new)})
                m["in"], m["out"] = new
                m.update(status="verified", verified_on=today, price_source=f"auto: {src}")
            report["sources"][name] = {"ok": True, "prices_found": len(prices), "models_matched": matched}
    if write:
        path = os.path.join(catalog.KNOWLEDGE_DIR, "models.json")
        data = json.load(open(path))
        data["models"] = catalog.MODELS
        json.dump(data, open(path, "w"), indent=1, ensure_ascii=False)
    LAST_SYNC.clear()
    LAST_SYNC.update(report)
    return report
