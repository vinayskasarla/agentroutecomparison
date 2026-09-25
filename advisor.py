"""The architect: understand a goal, pick candidate architectures, and rank them on measured results.

Pure logic lives here (no I/O besides the optional Claude call), so it can be tested on its own.
app.py runs the model evaluations and hands the per-test-case results to `rank_designs`.
"""
import datetime
import json
import os
import re
import statistics

import anthropic

import catalog

ARCHITECT_MODEL = os.environ.get("ARCHITECT_MODEL", "claude-opus-5")

TASK_TYPES = ["classification", "extraction", "grounded_qa", "open_qa", "summarization", "generation",
              "tool_actions", "research", "conversation"]
LATENCY_BUDGET_MS = {"realtime": 1500, "interactive": 5000, "background": 24 * 3600 * 1000}

# ------------------------------------------------------------------ architecture catalog
# Patterns, their best-practice sources and review date live in knowledge/patterns.json.
PATTERN_KNOWLEDGE = json.load(open(os.path.join(catalog.KNOWLEDGE_DIR, "patterns.json")))
ARCHITECTURES = PATTERN_KNOWLEDGE["patterns"]

# ------------------------------------------------------------------ sample test cases
POLICY = ("Acme Returns Policy\n- Unused items can be returned within 30 days of delivery for a full refund.\n"
          "- Opened electronics can be returned within 14 days with a 15% restocking fee.\n"
          "- Refunds go back to the original payment method within 5 business days of the item arriving at our warehouse.\n"
          "- Gift cards and downloadable software cannot be returned.\n"
          "- Return shipping is free for orders over $50; otherwise a $6 label fee is deducted.")

SAMPLE_CASES = {
    "classification": {"labels": ["billing", "technical", "account", "sales"], "cases": [
        ("I was charged twice for my subscription this month.", "billing"),
        ("The mobile app crashes every time I upload a photo.", "technical"),
        ("How do I change the email address on my profile?", "account"),
        ("Do you offer volume discounts for 200 seats?", "sales"),
        ("My invoice shows the wrong VAT number.", "billing"),
        ("I get a 500 error when I call the orders API.", "technical"),
        ("I'm locked out after too many password attempts.", "account"),
        ("Can someone walk our team through the enterprise plan?", "sales"),
        ("Why was my card declined when renewing?", "billing"),
        ("Sync between desktop and phone stopped working yesterday.", "technical"),
        ("Please delete my account and all my data.", "account"),
        ("We'd like a quote for adding the analytics add-on.", "sales")]},
    "grounded_qa": {"reference": POLICY, "cases": [
        ("How many days do I have to return an unused item?", "30"),
        ("Is there a fee for returning opened headphones?", "15%|restocking"),
        ("How long until my refund shows up?", "5 business days|5"),
        ("Can I return a gift card?", "no|cannot"),
        ("Is return shipping free on a $30 order?", "no|$6"),
        ("Do you ship to Canada?", "unknown"),
        ("Can I exchange an item for a different size instead of a refund?", "unknown"),
        ("Can I return downloadable software?", "no|cannot"),
        ("What is the restocking fee for opened electronics?", "15%"),
        ("Who pays return shipping on a $120 order?", "free|no fee")]},
    "extraction": {"cases": [
        ("Hi, my order A-10492 arrived damaged. Reply with just the order number.", "A-10492"),
        ("Order #B77310 still hasn't shipped, can you check? Reply with just the order number.", "B77310"),
        ("I'm Jane (jane.roe@acme.com). Ref: ORD-5521-X. Where's my parcel? Reply with just the order number.", "ORD-5521-X"),
        ("Tracking says delivered but no box. Order no. 884201. Reply with just the order number.", "884201"),
        ("Please cancel C-3010 (not C-3009, that one's fine). Reply with just the order number to cancel.", "C-3010"),
        ("Hey, where is my stuff?? Reply with just the order number, or unknown if there isn't one.", "unknown"),
        ("Invoice INV-2231 for order A-20931 has the wrong address. Reply with just the order number.", "A-20931"),
        ("My card 4111 1111 1111 1111 was charged for order Z-1180 twice. Reply with just the order number.", "Z-1180")]},
    "open_qa": {"cases": [
        ("What is 17 * 23?", "391"), ("What is the capital city of Australia?", "Canberra"),
        ("How many times does the letter r appear in strawberry?", "3|three"),
        ("Which number is larger: 9.11 or 9.9?", "9.9"),
        ("A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How many cents is the ball?", "5|five"),
        ("Which HTTP status code means Too Many Requests?", "429"), ("What is 15% of 240?", "36"),
        ("Who was the CEO of our company in 1850?", "unknown")]},
}
SAMPLE_FOR = {"classification": "classification", "extraction": "extraction", "grounded_qa": "grounded_qa",
              "summarization": "grounded_qa", "tool_actions": "classification", "conversation": "open_qa",
              "open_qa": "open_qa", "generation": "open_qa", "research": "open_qa"}

# ------------------------------------------------------------------ understanding the goal
SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "task_type", "labels", "needs_documents", "document_size", "needs_tools", "needs_memory", "multi_step",
                 "handles_personal_data", "risk", "latency", "requests_per_day", "repeat_rate", "accuracy_target",
                 "reference_material", "test_cases", "assumptions"],
    "properties": {
        "summary": {"type": "string", "description": "One sentence restating what the agent must do."},
        "task_type": {"type": "string", "enum": TASK_TYPES},
        "labels": {"type": "array", "items": {"type": "string"},
                   "description": "The fixed set of allowed answers if the output is a category or decision; else empty."},
        "needs_documents": {"type": "boolean", "description": "Answers must come from the company's own documents or data."},
        "document_size": {"type": "string", "enum": ["none", "small", "large"],
                          "description": "none if no documents; small if they fit in one prompt (under ~100 pages); large otherwise."},
        "needs_tools": {"type": "boolean", "description": "The agent must call APIs/systems or take actions."},
        "needs_memory": {"type": "boolean", "description": "Needs conversation history or per-user memory."},
        "multi_step": {"type": "boolean", "description": "Needs several distinct reasoning or research steps."},
        "handles_personal_data": {"type": "boolean"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"],
                 "description": "Impact of a wrong answer or action (money, safety, legal, customer trust)."},
        "latency": {"type": "string", "enum": ["realtime", "interactive", "background"],
                    "description": "realtime: <1.5s; interactive: a person waits a few seconds; background: nobody waits."},
        "requests_per_day": {"type": "integer"},
        "repeat_rate": {"type": "number", "description": "Share of requests that repeat an earlier one exactly, 0-0.9."},
        "accuracy_target": {"type": "number", "description": "Minimum acceptable share of correct answers, 0.5-1."},
        "reference_material": {"type": "string",
                               "description": "If needs_documents: a short realistic sample document the test cases are answered from. Else empty."},
        "test_cases": {
            "type": "array",
            "description": "12-16 realistic inputs with short gradable expected answers. Use 'unknown' when the correct behaviour is to decline. Separate alternatives with |.",
            "items": {"type": "object", "additionalProperties": False, "required": ["input", "expected"],
                      "properties": {"input": {"type": "string"}, "expected": {"type": "string"}}},
        },
        "assumptions": {"type": "array", "items": {"type": "string"},
                        "description": "What you assumed where the goal was silent (volume, latency, risk...)."},
    },
}

ARCHITECT_PROMPT = """You are a principal AI architect helping a product team plan an LLM-powered agent.
Read their goal and produce a specification plus test cases that will be used to measure candidate models.

Guidance:
- Pick the single best task_type. If the output is one of a fixed set of categories, list them in labels.
- Test cases must be realistic inputs the agent will see, each with a SHORT expected answer that can be checked
  by keyword match (a label, a number, a name, an ID, or a key phrase). Use | to separate acceptable alternatives.
- Include 2-3 cases where the right behaviour is to decline, with expected answer "unknown" (e.g. the question
  isn't covered by the documents, or the input lacks the needed information).
- For tool_actions or research tasks, write cases that test the decision the model must get right (which
  action/tool, which record, which conclusion), still with short expected answers.
- Where the goal doesn't say, assume sensible values for a mid-size company and list them in assumptions.

Goal:
"""


def claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


async def spec_from_claude(goal: str) -> dict:
    client = anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model=ARCHITECT_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SPEC_SCHEMA}},
        messages=[{"role": "user", "content": ARCHITECT_PROMPT + goal}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("the architect model declined this request")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def _has(text, *words):
    return any(re.search(r"\b" + w, text) for w in words)


def spec_from_rules(goal: str) -> dict:
    """Keyword-based stand-in for the Claude architect when no ANTHROPIC_API_KEY is set."""
    g = goal.lower()
    labels = []
    m = re.search(r"\(([^)]*(?:,| or | and )[^)]*)\)", goal) or re.search(
        r"\b(?:into|as either|as|between)\s+([\w\s/-]+(?:,\s*[\w\s/-]+)+(?:,?\s*(?:or|and)\s+[\w\s/-]+))", goal)
    if m:
        labels = [p.strip(" .").lower() for p in re.split(r",|\bor\b|\band\b", m.group(1)) if p.strip(" .")]
        labels = [lab for lab in labels if 0 < len(lab.split()) <= 3][:12]
    if _has(g, "research", "investigat", "deep dive", "competitive analysis", "due diligence", "literature"):
        task = "research"
    elif _has(g, "book", "refund", "cancel", "update the", "create a ticket", "file a", "send an email", "schedule",
              "place order", "take action", "call our api", "reset password", "provision"):
        task = "tool_actions"
    elif labels or _has(g, "classif", "categori", "route", "routing", "triage", "label", "tag ", "detect", "flag",
                        "spam", "sentiment", "intent", "prioriti"):
        task = "classification"
    elif _has(g, "extract", "pull out", "parse", "fields from", "invoice", "entities"):
        task = "extraction"
    elif _has(g, "policy", "policies", "documentation", "our docs", "knowledge base", "handbook", "faq", "manual",
              "contract", "from our", "internal wiki"):
        task = "grounded_qa"
    elif _has(g, "summar", "tl;dr", "digest", "recap", "meeting notes"):
        task = "summarization"
    elif _has(g, "write", "draft", "generate", "compose", "rewrite", "marketing copy", "reply to"):
        task = "generation"
    elif _has(g, "chat", "conversation", "assistant", "copilot"):
        task = "conversation"
    else:
        task = "open_qa"
    if task == "classification" and not labels:
        labels = (["spam", "not spam"] if "spam" in g else ["positive", "neutral", "negative"] if "sentiment" in g
                  else ["yes", "no"] if _has(g, "detect", "flag", "whether") else [])
    latency = ("background" if _has(g, "nightly", "overnight", "batch", "backfill", "weekly", "offline", "daily report")
               else "realtime" if _has(g, "real-time", "real time", "realtime", "instantly", "instant", "under a second", "live ", "in milliseconds")
               else "interactive")
    vol = 10000
    mv = re.search(r"(\d[\d,.]*)\s*(k|m|thousand|million)?\s*(?:requests|tickets|messages|emails|calls|queries|documents|per|a|/)\s*(?:per\s*)?(day|daily|month|hour)?", g)
    if mv:
        n = float(mv.group(1).replace(",", ""))
        n *= {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(mv.group(2) or "", 1)
        n /= {"month": 30, "hour": 1 / 24}.get(mv.group(3) or "", 1)
        vol = max(1, int(n))
    risk = ("high" if _has(g, "payment", "refund", "medical", "patient", "legal", "financ", "loan", "compliance",
                           "delete", "approve", "insurance claim")
            else "medium" if _has(g, "customer", "billing", "account", "order") else "low")
    sample = SAMPLE_CASES[SAMPLE_FOR[task]]
    cases = [{"input": i, "expected": e} for i, e in sample["cases"]]
    notes = ["Set ANTHROPIC_API_KEY so the architect writes test cases for your exact goal. "
             "These are sample cases from a similar task; replace them with real ones for real evidence."]
    if task == "classification" and labels and set(labels) != set(sample.get("labels", [])):
        cases = [{"input": c["input"], "expected": ""} for c in cases]
        notes.append("Your labels differ from the sample cases, so accuracy isn't graded. Add expected answers to grade it.")
    return {
        "summary": goal.strip().rstrip(".") + ".",
        "task_type": task,
        "labels": labels,
        "needs_documents": task == "grounded_qa" or _has(g, "our docs", "knowledge base", "help cent", "documentation"),
        "document_size": ("large" if _has(g, "knowledge base", "help cent", "documentation", "docs", "wiki", "manuals", "all our")
                          else "small") if task == "grounded_qa" or _has(g, "our docs", "knowledge base", "help cent", "documentation") else "none",
        "needs_tools": task == "tool_actions",
        "needs_memory": task in ("conversation", "tool_actions") or _has(g, "remember", "history", "follow-up", "multi-turn"),
        "multi_step": task in ("research",) or _has(g, "multi-step", "several steps", "then ", "workflow"),
        "handles_personal_data": _has(g, "customer", "patient", "employee", "user", "email", "phone", "account",
                                      "personal", "payment", "hr "),
        "risk": risk,
        "latency": latency,
        "requests_per_day": vol,
        "repeat_rate": 0.3 if task in ("grounded_qa", "open_qa", "conversation") else 0.1,
        "accuracy_target": 0.95 if risk == "high" else 0.9 if task in ("classification", "extraction") else 0.85,
        "reference_material": sample.get("reference", "") if task in ("grounded_qa", "summarization") else "",
        "test_cases": cases,
        "labels_from_sample": bool(labels == [] and "labels" in sample),
        "assumptions": notes,
    }


def finalize_spec(spec: dict) -> dict:
    s = dict(spec)
    s["task_type"] = s.get("task_type") if s.get("task_type") in TASK_TYPES else "open_qa"
    s["labels"] = [str(x).strip() for x in (s.get("labels") or []) if str(x).strip()][:30]
    if s["task_type"] == "classification" and not s["labels"] and s.get("labels_from_sample"):
        s["labels"] = SAMPLE_CASES["classification"]["labels"]
    s["document_size"] = s.get("document_size") if s.get("document_size") in ("none", "small", "large") else (
        "large" if s.get("needs_documents") else "none")
    s["risk"] = s.get("risk") if s.get("risk") in ("low", "medium", "high") else "medium"
    s["latency"] = s.get("latency") if s.get("latency") in LATENCY_BUDGET_MS else "interactive"
    s["requests_per_day"] = max(1, int(s.get("requests_per_day") or 10000))
    s["repeat_rate"] = min(0.9, max(0.0, float(s.get("repeat_rate") or 0)))
    s["accuracy_target"] = min(1.0, max(0.5, float(s.get("accuracy_target") or 0.9)))
    s.setdefault("max_hallucination", 0.02 if s["risk"] == "high" else 0.05)
    s.setdefault("latency_budget_ms", LATENCY_BUDGET_MS[s["latency"]])
    s.setdefault("monthly_budget", None)
    return s


def items_for(spec: dict) -> list:
    """Test cases in the shape app.normalize_items expects."""
    ref = (spec.get("reference_material") or "").strip()
    labels = spec.get("labels") or None
    out = []
    for c in spec.get("test_cases", [])[:40]:
        text = c["input"]
        if ref:
            text = (f"Reference material:\n{ref}\n\nQuestion: {c['input']}\n"
                    'Answer only from the reference material. If it doesn\'t cover the question, answer "unknown".')
        out.append({"input": text, "expected": c.get("expected", ""), "options": labels, "grounded": bool(ref)})
    return out


# ------------------------------------------------------------------ composing and ranking designs
def _cost(r, price):
    return (r["in_tokens"] * price["in"] + r["out_tokens"] * price["out"]) / 1e6


def _p(a, q):
    if not a:
        return 0.0
    s = sorted(a)
    return s[min(len(s) - 1, int(q * len(s)))]


def _agg(rows):
    """rows: per test case {correct, halluc, cost, ms}. correct None = ungraded."""
    graded = [r for r in rows if r["correct"] is not None]
    n = len(graded)
    return {
        "n": n,
        "right": sum(1 for r in graded if r["correct"]),
        "accuracy": (sum(1 for r in graded if r["correct"]) / n) if n else None,
        "halluc": (sum(1 for r in graded if r["halluc"]) / n) if n else None,
        "cost_per_req": statistics.fmean(r["cost"] for r in rows) if rows else 0.0,
        "p50_ms": _p([r["ms"] for r in rows], 0.5),
        "p95_ms": _p([r["ms"] for r in rows], 0.95),
    }


def _direct_rows(res, model):
    price = catalog.MODEL_BY_ID[model]
    return [{"correct": (False if r["error"] and r["correct"] is not None else r["correct"]),
             "halluc": bool(r.get("hallucinated")), "cost": _cost(r, price), "ms": r["latency_ms"],
             "conf": r.get("confidence"), "error": bool(r["error"])} for r in res[model]]


def compose(arch, models, res, jev, spec):
    """Per-test-case outcomes for one architecture built from measured single-call results.

    `res[model]` is the list of per-case results for that model; `jev` the per-case Jev results (or None).
    Returns (rows, evidence) where evidence is 'measured', 'composed' or 'estimated'.
    """
    if arch in ("single_call", "rag", "batch", "long_context"):
        rows = _direct_rows(res, models["primary"])
        if arch == "rag":  # retrieval step; cases carry the reference text, i.e. retrieval found the right passage
            rows = [{**r, "ms": r["ms"] + 120, "cost": r["cost"] + 0.00002} for r in rows]
        if arch == "batch":
            rows = [{**r, "cost": r["cost"] * 0.5, "ms": 30 * 60 * 1000} for r in rows]
        return rows, "measured" if arch in ("single_call", "long_context") else "composed"
    if arch == "cascade":
        small, large = _direct_rows(res, models["small"]), _direct_rows(res, models["primary"])
        rows = []
        for s, g in zip(small, large):
            if not s["error"] and s["conf"] is not None and s["conf"] >= 80:
                rows.append({**s, "escalated": False})
            else:
                rows.append({**g, "cost": s["cost"] + g["cost"], "ms": s["ms"] + g["ms"], "escalated": True})
        return rows, "composed"
    if arch == "decision_model":
        large = _direct_rows(res, models["primary"])
        rows = []
        for j, g in zip(jev, large):
            jcost = j["in_tokens"] * catalog.JEV["in"] / 1e6
            if not j["error"] and j["confidence"] is not None and j["confidence"] >= 80:
                rows.append({"correct": j["correct"], "halluc": bool(j.get("hallucinated")), "cost": jcost,
                             "ms": j["latency_ms"], "escalated": False})
            else:
                rows.append({**g, "cost": g["cost"] + jcost, "ms": g["ms"] + j["latency_ms"], "escalated": True})
        return rows, "composed"
    base = _direct_rows(res, models["primary"])
    if arch == "workflow":
        small = _direct_rows(res, models["small"])
        return [{**g, "cost": g["cost"] + s["cost"], "ms": g["ms"] + s["ms"] + 40} for g, s in zip(base, small)], "estimated"
    if arch == "evaluator_optimizer":  # draft + review + one revision with the same model
        return [{**g, "cost": g["cost"] * 2.6, "ms": g["ms"] * 2.5} for g in base], "estimated"
    if arch == "tool_agent":  # ~3 model turns with growing context, 2 tool calls of ~300 ms
        return [{**g, "cost": g["cost"] * 3.6, "ms": g["ms"] * 3 + 600} for g in base], "estimated"
    if arch == "multi_agent":  # lead plans + synthesizes, 3 workers in parallel on the small model
        small = _direct_rows(res, models["small"])
        return [{**g, "cost": g["cost"] * 2.4 + s["cost"] * 3 * 1.5, "ms": g["ms"] * 2 + s["ms"] * 1.3 + 300}
                for g, s in zip(base, small)], "estimated"
    raise ValueError(arch)


def applicable(spec):
    """Which architectures fit the task, and why the others don't."""
    t, why_not, ok = spec["task_type"], {}, []
    labels, docs, tools = bool(spec["labels"]), spec["needs_documents"], spec["needs_tools"]
    multi = spec["multi_step"] or t == "research"
    single_step = not tools and not multi
    rules = {
        "single_call": (not tools and not multi, "the task needs actions or several steps, which one call can't do"),
        "cascade": (not tools and not multi, "the task needs actions or several steps"),
        "decision_model": (labels and not tools, "the answer isn't one of a fixed set of labels"),
        "rag": (docs, "answers don't need to come from your own documents"),
        "workflow": (multi or tools or t in ("summarization", "generation"), "the task is a single step"),
        "tool_agent": (tools, "the agent doesn't need to call your systems or take actions"),
        "evaluator_optimizer": (t in ("generation", "summarization", "research") and not tools,
                                "answers are short and checkable, so a review loop adds cost without adding quality"),
        "multi_agent": (t == "research" or (multi and spec["latency"] != "realtime"), "the task isn't open-ended research"),
        "long_context": (docs and spec["document_size"] == "small", "your documents are too large to send with every request"
                         if docs else "answers don't need to come from your own documents"),
        "batch": (spec["latency"] == "background" and single_step,
                  "someone is waiting for the answer" if spec["latency"] != "background"
                  else "batch APIs run single calls; this task needs several steps or tools"),
    }
    if docs:  # grounded tasks must retrieve; a bare call can't see the documents
        rules["single_call"] = (False, "a single call can't see your documents; it would need them pasted in (that's RAG)")
        rules["cascade"] = (False, "without retrieval the model can't see your documents")
    for a, (fits, reason) in rules.items():
        (ok.append(a) if fits else why_not.__setitem__(a, reason))
    return ok, why_not


def pick_models(res, spec, provider_of):
    """Choose primary, small and fallback models from measured single-call results."""
    rows = {}
    for m in res:
        agg = _agg(_direct_rows(res, m))
        rows[m] = agg
    target = spec["accuracy_target"]
    good = [m for m, a in rows.items() if a["accuracy"] is not None and a["accuracy"] >= target
            and a["halluc"] <= spec["max_hallucination"]]
    verified = {m["id"] for m in catalog.MODELS if m.get("status") == "verified"}
    # A source of truth shouldn't lean on unconfirmed prices: prefer verified models when any qualifies.
    good = [m for m in good if m in verified] or good
    by_cost = lambda m: rows[m]["cost_per_req"]  # noqa: E731
    by_quality = lambda m: (-(rows[m]["accuracy"] or 0), rows[m]["halluc"] or 0, rows[m]["cost_per_req"])  # noqa: E731
    primary = min(good, key=by_cost) if good else min(rows, key=by_quality)
    best = min(rows, key=by_quality)
    small_pool = [m for m in rows if catalog.MODEL_BY_ID[m]["tier"] == "small"] or list(rows)
    small = min(small_pool, key=lambda m: (-(rows[m]["accuracy"] or 0), rows[m]["cost_per_req"]))
    large_pool = [m for m in rows if catalog.MODEL_BY_ID[m]["tier"] != "small"]
    best_large = min(large_pool, key=by_quality) if large_pool else None
    others = [m for m in rows if provider_of(m) != provider_of(primary)]
    fb_good = [m for m in others if m in good] or [m for m in others if m in verified and rows[m]["accuracy"] is not None
                                                   and rows[m]["accuracy"] >= target]
    fallback = (min(fb_good, key=by_cost) if fb_good else min(others, key=by_quality)) if others else None
    return {"primary": primary, "best": best, "best_large": best_large, "small": small, "fallback": fallback, "table": rows, "qualifying": good}


def rank_designs(spec, res, jev):
    """Build every candidate design, check it against the requirements, and return the top 3 with reasons."""
    provider_of = lambda m: catalog.MODEL_BY_ID[m]["provider"]  # noqa: E731
    picks = pick_models(res, spec, provider_of)
    ok, why_not = applicable(spec)
    if jev is None and "decision_model" in ok:
        ok.remove("decision_model")
        why_not["decision_model"] = "Jev needs a fixed set of labels to choose from"
    rpd, repeat = spec["requests_per_day"], spec["repeat_rate"]
    designs = []
    for arch in ok:
        if arch == "cascade":
            large = picks["best"] if catalog.MODEL_BY_ID[picks["best"]]["tier"] != "small" else picks["best_large"]
            variants = [(large, "value")] if large and large != picks["small"] else []
        else:
            variants = [(picks["primary"], "value")] + ([(picks["best"], "quality")] if picks["best"] != picks["primary"] else [])
            if picks["fallback"] and picks["fallback"] not in (picks["primary"], picks["best"]):
                variants.append((picks["fallback"], "other provider"))
            if picks["best_large"] and all(picks["best_large"] != v for v, _ in variants):
                variants.append((picks["best_large"], "stronger model"))
        for primary, variant in variants:
            models = {"primary": primary, "small": picks["small"], "fallback": picks["fallback"]}
            rows, evidence = compose(arch, models, res, jev, spec)
            m = _agg(rows)
            addons = addons_for(arch, spec)
            infra = sum(a["per_1k"] for a in addons) / 1000
            cache = next((a for a in addons if a["id"] == "cache"), None)
            per_req = m["cost_per_req"] * (1 - repeat if cache else 1) + infra
            p95 = m["p95_ms"] + sum(a["ms"] for a in addons)
            d = {
                "arch": arch, **ARCHITECTURES[arch],
                "sources": [PATTERN_KNOWLEDGE["sources"][k] for k in ARCHITECTURES[arch].get("sources", [])],
                "variant": variant, "models": models, "evidence": evidence, "addons": addons,
                "accuracy": m["accuracy"], "right": m["right"], "n": m["n"], "halluc": m["halluc"],
                "p50_ms": m["p50_ms"], "p95_ms": p95, "cost_per_req": per_req, "monthly": per_req * rpd * 30,
                "escalation_rate": (sum(1 for r in rows if r.get("escalated")) / len(rows)) if rows and arch in ("cascade", "decision_model") else None,
            }
            d["checks"] = checks(d, spec)
            d["passes"] = all(c["pass"] for c in d["checks"])
            designs.append(d)
    score_designs(designs, spec)
    designs.sort(key=lambda d: (not d["passes"], -d["score"]))
    top, seen = [], set()
    for d in designs:  # prefer three different architectures, then fill with model variants
        if d["arch"] not in seen:
            top.append(d)
            seen.add(d["arch"])
        if len(top) == 3:
            break
    for d in designs:
        if len(top) == 3:
            break
        if d not in top:
            top.append(d)
    explain(top, spec, picks)
    return {"top": top, "all": designs, "not_applicable": why_not, "models": picks}


def addons_for(arch, spec):
    out = [{"id": "gateway", "name": "AI Gateway", "why": "central keys, quotas, cost tracking and provider failover",
            "ms": 5, "per_1k": 0.006}]
    if spec["handles_personal_data"]:
        out[0]["why"] += "; redacts personal data before it reaches the model"
    if arch in ("tool_agent", "workflow", "multi_agent") or spec["needs_memory"]:
        out.append({"id": "runtime", "name": "Agent Runtime", "why": "memory, tool permissions, tracing and policy",
                    "ms": 15, "per_1k": 0.012})
    if spec["repeat_rate"] >= 0.15 and arch not in ("batch", "tool_agent", "multi_agent"):
        out.append({"id": "cache", "name": "Response cache",
                    "why": f"about {round(spec['repeat_rate'] * 100)}% of requests repeat, and those skip the model", "ms": 0, "per_1k": 0.002})
    if spec["risk"] == "high":
        out.append({"id": "review", "name": "Human review", "why": "high-impact decisions below 80% confidence go to a person",
                    "ms": 0, "per_1k": 0})
    if spec["needs_documents"] and arch != "rag":
        out.append({"id": "grounding", "name": "Grounding check", "why": "verify each answer is supported by your documents",
                    "ms": 0, "per_1k": 0})
    return out


def checks(d, spec):
    out = []
    if d["accuracy"] is not None:
        out.append({"key": "accuracy", "label": "Accuracy", "pass": d["accuracy"] >= spec["accuracy_target"],
                    "value": d["accuracy"], "target": spec["accuracy_target"]})
        out.append({"key": "halluc", "label": "Hallucinations", "pass": d["halluc"] <= spec["max_hallucination"],
                    "value": d["halluc"], "target": spec["max_hallucination"]})
    out.append({"key": "latency", "label": "Latency p95", "pass": d["p95_ms"] <= spec["latency_budget_ms"],
                "value": d["p95_ms"], "target": spec["latency_budget_ms"]})
    if spec.get("monthly_budget"):
        out.append({"key": "cost", "label": "Monthly cost", "pass": d["monthly"] <= spec["monthly_budget"],
                    "value": d["monthly"], "target": spec["monthly_budget"]})
    return out


def score_designs(designs, spec):
    """Among designs, reward accuracy, low cost, low latency and simplicity (start simple).
    When nobody waits for the answer, speed doesn't count and its weight moves to cost."""
    if not designs:
        return
    w_cost, w_lat = (0.40, 0.0) if spec["latency"] == "background" else (0.25, 0.15)
    min_cost = min(d["cost_per_req"] for d in designs) or 1e-9
    min_lat = min(d["p95_ms"] for d in designs) or 1
    for d in designs:
        acc = d["accuracy"] if d["accuracy"] is not None else 0.8
        hall_pen = min(1.0, (d["halluc"] or 0) * 5)
        d["score"] = round(100 * (0.35 * acc * (1 - hall_pen * 0.5)
                                  + w_cost * min(1.0, min_cost / max(d["cost_per_req"], 1e-9))
                                  + w_lat * min(1.0, min_lat / max(d["p95_ms"], 1))
                                  + 0.25 * (6 - d["complexity"]) / 5), 1)


def _pct(x):
    return "—" if x is None else f"{round(x * 100)}%"


def _ms(x):
    if x >= 3600 * 1000:
        return f"{x / 3600000:.0f} h"
    if x >= 60 * 1000:
        return f"{x / 60000:.0f} min"
    return f"{x / 1000:.1f} s" if x >= 1000 else f"{round(x)} ms"


def _usd(x):
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def explain(top, spec, picks):
    """Why #1 wins, and what #2/#3 trade against it."""
    if not top:
        return
    first = top[0]
    label = lambda m: catalog.MODEL_BY_ID[m]["label"] if m else "—"  # noqa: E731
    reasons = []
    if first["passes"]:
        reasons.append(f"Meets all {len(first['checks'])} of your requirements.")
    else:
        miss = [c["label"].lower() for c in first["checks"] if not c["pass"]]
        reasons.append(f"No design met every requirement; this is the closest (misses {', '.join(miss)}).")
    if first["accuracy"] is not None:
        reasons.append(f"{first['right']} of {first['n']} test cases correct ({_pct(first['accuracy'])}), "
                       f"hallucinations {_pct(first['halluc'])}.")
    reasons.append(f"About {_usd(first['monthly'])}/month at {spec['requests_per_day']:,} requests/day; "
                   f"95% of requests within {_ms(first['p95_ms'])}.")
    arch_why = {
        "single_call": "The task is a single step, so extra moving parts would add cost and failure points without adding accuracy.",
        "cascade": f"Most requests are easy enough for {label(first['models']['small'])}; only "
                   f"{_pct(first['escalation_rate'])} needed {label(first['models']['primary'])}.",
        "decision_model": f"The answer is one of {len(spec['labels'])} labels, so Jev decides "
                          f"{_pct(1 - (first['escalation_rate'] or 0))} of requests in about a tenth of a second for a fraction of a cent.",
        "long_context": "Your documents are small enough to send with every request, so you get grounded answers without building a search index.",
        "rag": "Answers have to come from your documents; retrieving the right passages first keeps them grounded and lets the model say 'unknown'.",
        "workflow": "The job has predictable steps, so a fixed pipeline is easier to test and debug than a free-roaming agent.",
        "tool_agent": "The agent has to act in your systems, and which tools it needs varies per request.",
        "multi_agent": "The task is broad and open-ended; parallel specialists cover more ground than one model.",
        "batch": "Nobody is waiting for the result, so batch pricing halves the model cost.",
        "evaluator_optimizer": "Written quality matters here, and a second review pass catches issues a single draft misses.",
    }[first["arch"]]
    reasons.append(arch_why)
    reasons.append(f"{label(first['models']['primary'])} is the cheapest tested model that reached your accuracy target."
                   if first["models"]["primary"] in picks["qualifying"]
                   else f"{label(first['models']['primary'])} was the most accurate model tested, but none reached the target.")
    first["why"] = reasons
    for d in top[1:]:
        t = []
        if d["accuracy"] is not None and first["accuracy"] is not None:
            diff = round((d["accuracy"] - first["accuracy"]) * 100)
            if diff:
                t.append(f"{'+' if diff > 0 else ''}{diff} pts accuracy")
            else:
                t.append("same accuracy on your cases")
        ratio = d["monthly"] / first["monthly"] if first["monthly"] else 1
        if ratio >= 1.15:
            t.append(f"{ratio:.1f}× the cost ({_usd(d['monthly'])}/month)")
        elif ratio <= 0.87:
            t.append(f"{round((1 - ratio) * 100)}% cheaper ({_usd(d['monthly'])}/month)")
        else:
            t.append(f"similar cost ({_usd(d['monthly'])}/month)")
        dl = d["p95_ms"] - first["p95_ms"]
        if "batch" in (d["arch"], first["arch"]) and d["arch"] != first["arch"]:
            t.append("answers within 30 minutes instead of seconds" if d["arch"] == "batch"
                     else f"answers in {_ms(d['p95_ms'])} instead of up to 30 minutes")
        elif abs(dl) >= 100:
            t.append(f"{_ms(abs(dl))} {'slower' if dl > 0 else 'faster'} at p95")
        dc = d["complexity"] - first["complexity"]
        if dc:
            t.append("more to build and operate" if dc > 0 else "simpler to build and operate")
        if d["arch"] in ("cascade", "decision_model") and d["escalation_rate"] is not None:
            big = label(d["models"]["primary"])
            t.append(f"{_pct(d['escalation_rate'])} of your cases escalated to {big}"
                     + (" — the harder mix in real traffic will raise that" if d["escalation_rate"] < 0.05 else ""))
        if d["arch"] == first["arch"] and d["models"]["primary"] != first["models"]["primary"]:
            t.insert(0, f"same pattern on {label(d['models']['primary'])}"
                        + (" (a different provider, for failover)" if d["variant"] == "other provider" else ""))
        if not d["passes"]:
            t.append("misses " + ", ".join(c["label"].lower() for c in d["checks"] if not c["pass"]))
        d["tradeoffs"] = t


# ------------------------------------------------------------------ trust helpers
def cross_check(spec: dict, goal: str) -> list:
    """Compare the Claude architect's reading of the goal with the keyword rules. Disagreements are shown
    for the user to confirm, so a misread goal can't silently drive the recommendation."""
    rules = finalize_spec(spec_from_rules(goal))
    names = {"task_type": "Task type", "needs_documents": "Answers from your documents", "needs_tools": "Takes actions",
             "latency": "Who is waiting", "handles_personal_data": "Personal data"}
    out = []
    for key, label in names.items():
        if spec.get(key) != rules.get(key) and not (key == "task_type" and rules[key] == "open_qa"):
            out.append({"field": key, "label": label, "architect": spec.get(key), "rules": rules.get(key)})
    if rules.get("labels") and not spec.get("labels"):
        out.append({"field": "labels", "label": "Fixed answers", "architect": [], "rules": rules["labels"]})
    return out


def knowledge_status(today=None) -> dict:
    """How fresh the model and pattern knowledge is."""
    today = today or datetime.date.today()

    def age(d):
        return (today - datetime.date.fromisoformat(d)).days if d else None
    mk, pk = catalog.MODEL_KNOWLEDGE, PATTERN_KNOWLEDGE
    models = [{**{k: m.get(k) for k in ("id", "label", "provider", "tier", "in", "out", "status", "verified_on", "source", "review_note")},
               "age_days": age(m.get("verified_on"))} for m in mk["models"]]
    stale = [m for m in models if m["status"] != "verified" or (m["age_days"] or 0) > mk["review_every_days"]]
    return {
        "models": models, "watchlist": mk.get("watchlist", []), "model_review_every_days": mk["review_every_days"],
        "models_verified": len(models) - len(stale), "models_total": len(models),
        "models_needing_review": [m["id"] for m in stale],
        "patterns_reviewed_on": pk["reviewed_on"], "patterns_age_days": age(pk["reviewed_on"]),
        "patterns_stale": age(pk["reviewed_on"]) > pk["review_every_days"],
        "principles": pk["principles"], "sources": pk["sources"],
        "jev": {"label": catalog.JEV["label"], "in": catalog.JEV["in"], "status": "needs_review",
                "note": "Price from third-party launch write-ups; confirm with TypeSafe.", "source": "https://typesafe.ai"},
    }
