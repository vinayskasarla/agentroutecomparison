"""Check which vendors this machine can call live, with one tiny real call each (a fraction of a cent in total).

    python check_setup.py

Reads the same .env as the app. Prints, per approved platform, whether credentials are set and whether a real call
works, so you know what will be measured live and what will be simulated before you run the advisor.
"""
import asyncio
import time

import envfile
import catalog
import policy
from providers import call_llm, has_key

# One small, cheap model per platform to prove the credentials work.
PROBE = {"anthropic": "claude-haiku-4-5", "openai": "gpt-5-mini", "xai": "grok-4-fast", "google": "gemini-2.5-flash",
         "bedrock": "bedrock-nova-micro", "azure": "azure-gpt-5-mini"}
EXTRA = {"bedrock": "bedrock-claude-haiku-4-5", "azure": "azure-claude-haiku-4-5"}  # Claude on the same cloud


async def probe(model_id):
    t = time.perf_counter()
    r = await call_llm(model_id, "Reply with the single word: ok", item=None)
    ok = not r["error"] and not r["simulated"]
    cost = (r["in_tokens"] * catalog.MODEL_BY_ID[model_id]["in"] + r["out_tokens"] * catalog.MODEL_BY_ID[model_id]["out"]) / 1e6
    return ok, (r["error"] or "").strip()[:160], (time.perf_counter() - t) * 1000, cost if ok else 0.0


async def main():
    print(f".env: {envfile.PATH} ({'found, ' + str(len(envfile.LOADED)) + ' settings loaded' if envfile.LOADED else 'not found or empty'})\n")
    total, live_models = 0.0, 0
    rows = []
    for prov, model in PROBE.items():
        plat = catalog.MODEL_BY_ID[model]["platform"]
        if not policy.model_allowed(model):
            rows.append((plat, "blocked by policy", "")); continue
        if not has_key(prov):
            rows.append((plat, "not configured", f"set {catalog.PROVIDERS[prov]['env']}")); continue
        checks = [model] + ([EXTRA[prov]] if prov in EXTRA else [])
        for m in checks:
            ok, err, ms, cost = await probe(m)
            total += cost
            n = sum(1 for x in catalog.CALLABLE if x["provider"] == prov and policy.model_allowed(x["id"]))
            if ok and m == model:
                live_models += n
            rows.append((f"{plat} · {catalog.MODEL_BY_ID[m]['label']}", "LIVE ✓" if ok else "FAILED ✗",
                         f"{ms:.0f} ms" if ok else err))
    w = max(len(r[0]) for r in rows)
    for name, status, note in rows:
        print(f"  {name:<{w}}  {status:<16} {note}")
    approved = sum(1 for x in catalog.CALLABLE if policy.model_allowed(x["id"]))
    print(f"\nAbout {live_models} of {approved} approved model offerings can be measured live; the rest will be simulated.")
    print(f"This check cost about ${total:.5f}.")
    if not envfile.LOADED:
        print("\nTip: cp .env.example .env, fill in your keys, and run this again.")

if __name__ == "__main__":
    asyncio.run(main())
