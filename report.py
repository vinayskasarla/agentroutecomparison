"""Run a set of goals through your running advisor and write a report of the real numbers.

    uvicorn app:app --port 8088          # in one terminal (reads .env)
    python report.py                     # first runs only (shortlist), ~$0.05 per goal
    python report.py --compare           # plus a full comparison per goal, ~$0.10-0.40 more each
    python report.py --goals my_goals.txt --out reports/

Writes reports/advisor-report-<timestamp>.md and .csv: per goal, the recommended architecture, model and
fallback, accuracy and its likely range, monthly cost, how many models were measured live, and what the run cost.
"""
import argparse
import csv
import datetime
import json
import os
import sys

import httpx

DEFAULT_GOALS = [
    "Route every incoming support ticket to the right team (billing, technical, account or sales). About 50k tickets a day, in real time.",
    "Answer customer questions from our returns policy and help center docs; never make things up. 20k questions a day.",
    "Extract the order number and the customer's requested action from support emails. About 20k emails a day.",
    "A support agent that answers customers from our help-center docs, looks up orders and issues refunds through our order and "
    "payments APIs (exposed via MCP servers), and hands complex cases to a specialist sub-agent. 20k conversations a day.",
]


def advise(client, body):
    r = client.post("/api/advise", json=body, timeout=900)
    if r.status_code != 200:
        return None, None, r.json().get("error", r.text)
    spec = advice = err = None
    for line in r.text.splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev["type"] == "spec":
            spec = ev["spec"]
        elif ev["type"] == "advice":
            advice = ev
        elif ev["type"] == "error":
            err = ev["error"]
    return spec, advice, err


def summarize(goal, advice, kind):
    d, p, table = advice["top"][0], advice["models"], {r["id"]: r for r in advice["model_table"]}
    label = lambda m: f"{table[m]['label']} ({table[m]['platform']})" if m in table else (m or "—")  # noqa: E731
    live = sum(1 for v in advice["mode"].values() if v == "live")
    return {"goal": goal, "run": kind, "architecture": d["name"], "model": label(p["primary"]), "fallback": label(p["fallback"]),
            "meets_all": "yes" if d["passes"] else "no: " + ", ".join(c["label"].lower() for c in d["checks"] if not c["pass"]),
            "accuracy": f"{round(100 * d['accuracy'])}%" if d["accuracy"] is not None else "—",
            "likely_range": f"{round(100 * d['acc_lo'])}–{round(100 * d['acc_hi'])}%" if d.get("acc_lo") is not None else "—",
            "test_cases": d["n"], "monthly_usd": round(d["monthly"], 2), "fixed_infra_usd": round(d.get("infra_monthly", 0), 2),
            "models_tested": len(advice["evaluated_models"]), "models_live": live,
            "run_cost_usd": round(advice["run_cost"]["total"], 4), "cost_if_all_live_usd": round(advice["run_cost"]["if_all_live"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8088")
    ap.add_argument("--goals", help="text file, one goal per line")
    ap.add_argument("--compare", action="store_true", help="also run a full comparison per goal (uses the daily limit)")
    ap.add_argument("--harness", action="store_true", help="include production controls in every cost")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    goals = [g.strip() for g in open(a.goals)] if a.goals else DEFAULT_GOALS
    goals = [g for g in goals if g]
    rows = []
    with httpx.Client(base_url=a.base_url, auth=("report", os.environ["APP_PASSWORD"]) if os.environ.get("APP_PASSWORD") else None) as c:
        try:
            c.get("/").raise_for_status()  # starts a browser-style session (cookie)
        except Exception as e:
            sys.exit(f"Can't reach the advisor at {a.base_url} ({e}). Start it first: uvicorn app:app --port 8088")
        for i, goal in enumerate(goals, 1):
            print(f"[{i}/{len(goals)}] {goal[:90]}…", flush=True)
            spec, adv, err = advise(c, {"goal": goal, "harness": a.harness})
            if err or not adv:
                print("   failed:", err); continue
            for u in adv.get("unavailable", []):
                print(f"   ! {u['label']} ({u['platform']}) couldn't be called live, simulated instead: {u['error'][:120]}")
            rows.append(summarize(goal, adv, "first run")); r = rows[-1]
            print(f"   → {r['architecture']} on {r['model']} · {r['accuracy']} ({r['likely_range']}) · ${r['monthly_usd']:,}/mo · run cost ${r['run_cost_usd']}")
            if a.compare:
                _, full, err = advise(c, {"goal": goal, "spec": spec, "compare": True, "harness": a.harness})
                if err or not full:
                    print("   comparison skipped:", err); continue
                rows.append(summarize(goal, full, "full comparison")); r = rows[-1]
                print(f"   → compared {r['models_tested']} models ({r['models_live']} live): {r['model']} · ${r['monthly_usd']:,}/mo · run cost ${r['run_cost_usd']}")
    if not rows:
        sys.exit("No results.")
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    base = os.path.join(a.out, f"advisor-report-{stamp}")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    spent = sum(r["run_cost_usd"] for r in rows)
    with open(base + ".md", "w") as f:
        f.write(f"# Architecture Advisor — results {datetime.date.today()}\n\n")
        f.write(f"{len(goals)} goals · total model spend for this report: **${spent:.3f}**\n\n")
        f.write("| Goal | Run | Architecture | Model | Fallback | Accuracy (likely) | Meets all | Per month | Models tested (live) | Run cost |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['goal'][:70]}… | {r['run']} | {r['architecture']} | {r['model']} | {r['fallback']} | {r['accuracy']} ({r['likely_range']}) "
                    f"| {r['meets_all']} | ${r['monthly_usd']:,} | {r['models_tested']} ({r['models_live']}) | ${r['run_cost_usd']} |\n")
        f.write("\nModels not measured live are simulated from profiles and marked in the app; add their vendor keys for real measurements.\n")
    print(f"\nWrote {base}.md and {base}.csv · total spend ${spent:.3f}")


if __name__ == "__main__":
    main()
