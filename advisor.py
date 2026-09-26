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
import constraints
import policy

# Measured on 4 goals: Sonnet 5 at medium effort matched Opus 5's reading on 42 of 44 decisions at ~43% of the
# cost ($0.023 vs $0.053 per goal). Set ARCHITECT_MODEL=claude-opus-5 for the strongest reading.
ARCHITECT_MODEL = os.environ.get("ARCHITECT_MODEL", "claude-sonnet-5")
ARCHITECT_EFFORT = os.environ.get("ARCHITECT_EFFORT", "medium")
# $ per 1M tokens (in, out) for architect models that aren't in the catalog as tested offerings.
ARCHITECT_PRICES = {"claude-opus-5": (5, 25), "claude-sonnet-5": (2, 10), "claude-haiku-4-5": (1, 5)}


def compare_estimate(res, eligible):
    """What testing every eligible model on the same cases would cost, from this run's measured token use."""
    rows = [r for rs in res.values() for r in rs]
    if not rows:
        return None
    n = max(len(rs) for rs in res.values())
    tin, tout = statistics.fmean(r["in_tokens"] for r in rows), statistics.fmean(r["out_tokens"] for r in rows)
    per = {m: n * (tin * catalog.MODEL_BY_ID[m]["in"] + tout * catalog.MODEL_BY_ID[m]["out"]) / 1e6 for m in eligible}
    top = max(per, key=per.get)
    return {"models": len(eligible), "cost": sum(per.values()), "largest": catalog.MODEL_BY_ID[top]["label"], "largest_cost": per[top]}


def run_cost(arch_usage, res, mode, scope, judge_usage=None):
    """What this run cost, per action, from actual token usage. Simulated calls cost nothing; what they would
    have cost live is reported separately so the numbers can be projected."""
    steps = []
    if arch_usage:
        pin, pout = ARCHITECT_PRICES.get(arch_usage["model"], (catalog.MODEL_BY_ID.get(arch_usage["model"], {}).get("in", 0),
                                                                catalog.MODEL_BY_ID.get(arch_usage["model"], {}).get("out", 0)))
        steps.append({"step": "Understand the goal and write test cases", "kind": "architect", "model": arch_usage["model"],
                      "calls": 1, "in": arch_usage["in"], "out": arch_usage["out"], "live": True,
                      "cost": (arch_usage["in"] * pin + arch_usage["out"] * pout) / 1e6})
    for m, rows in res.items():
        info = catalog.MODEL_BY_ID[m]
        fresh = [r for r in rows if not r.get("reused")]
        if not fresh:  # every answer reused from earlier in this session: nothing new was spent
            steps.append({"step": f"Test {info['label']} ({info['platform']})", "kind": "test", "model": m, "calls": 0,
                          "in": 0, "out": 0, "live": mode.get(m) == "live", "reused": True, "cost": 0.0, "if_live": 0.0})
            continue
        tin, tout = sum(r["in_tokens"] for r in fresh), sum(r["out_tokens"] for r in fresh)
        priced = (tin * info["in"] + tout * info["out"]) / 1e6
        live = mode.get(m) == "live"
        steps.append({"step": f"Test {info['label']} ({info['platform']})", "kind": "test", "model": m, "calls": len(fresh),
                      "in": tin, "out": tout, "live": live, "cost": priced if live else 0.0, "if_live": priced,
                      "reused_calls": len(rows) - len(fresh)})
    if judge_usage:
        jin, jout = sum(u["in"] for u in judge_usage), sum(u["out"] for u in judge_usage)
        steps.append({"step": "Grade free-text answers (judge)", "kind": "judge", "model": "claude-haiku-4-5", "calls": len(judge_usage),
                      "in": jin, "out": jout, "live": True, "cost": (jin * 1.0 + jout * 5.0) / 1e6})
    steps.sort(key=lambda x: -(x["cost"] or x.get("if_live", 0)))
    total = sum(x["cost"] for x in steps)
    projected = sum(x["cost"] if x["live"] else x.get("if_live", 0) for x in steps)
    return {"scope": scope, "total": total, "if_all_live": projected, "steps": steps, "compare_estimate": None,
            "note": "Ranking, pricing, what-ifs and the system design are computed on the server at no model cost."}

TASK_TYPES = ["classification", "extraction", "grounded_qa", "open_qa", "summarization", "generation",
              "tool_actions", "research", "conversation"]
LATENCY_BUDGET_MS = {"realtime": 1500, "interactive": 5000, "background": 24 * 3600 * 1000}

# ------------------------------------------------------------------ architecture catalog
# Patterns, their best-practice sources and review date live in knowledge/patterns.json.
PATTERN_KNOWLEDGE = json.load(open(os.path.join(catalog.KNOWLEDGE_DIR, "patterns.json")))
ARCHITECTURES = PATTERN_KNOWLEDGE["patterns"]
HARNESS = json.load(open(os.path.join(catalog.KNOWLEDGE_DIR, "harness.json")))
LOOPING = ("tool_agent", "multi_agent", "workflow", "evaluator_optimizer", "agentic_rag", "router")

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
                 "reference_material", "test_cases", "assumptions", "data_class", "needs_images",
                 "steps_known", "request_types", "specialists", "multi_hop", "docs_change_often", "structured_data",
                 "tool_count", "integration"],
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
        "data_class": {"type": "string", "enum": ["public", "internal", "confidential", "regulated"],
                       "description": "Most sensitive data the agent sees: confidential = customer/employee personal data; regulated = health, payment card or financial records."},
        "needs_images": {"type": "boolean", "description": "Inputs include images, scans or screenshots."},
        "steps_known": {"type": "boolean", "description": "The steps are the same for every request (a fixed sequence could handle it)."},
        "request_types": {"type": "integer", "description": "How many distinct kinds of request the agent handles that need different handling (1 if uniform)."},
        "specialists": {"type": "boolean", "description": "Hands work to specialist sub-agents or clearly separate domains."},
        "multi_hop": {"type": "boolean", "description": "Answers must combine several documents or sources, or need a follow-up search."},
        "docs_change_often": {"type": "boolean", "description": "The documents change daily or more often."},
        "structured_data": {"type": "boolean", "description": "Facts it needs live in databases or tables rather than documents."},
        "tool_count": {"type": "integer", "description": "Roughly how many distinct APIs/tools the agent calls (0 if none)."},
        "integration": {"type": "string", "enum": ["direct", "mcp", "both"], "description": "How tools are connected: direct function tools, MCP servers, or both. direct unless MCP is mentioned."},
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
- For agentic goals, count the distinct APIs/tools and request types, and say whether the steps are the same every time.
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
        output_config={**({} if ARCHITECT_MODEL.startswith("claude-haiku") else {"effort": ARCHITECT_EFFORT}),
                       "format": {"type": "json_schema", "schema": SPEC_SCHEMA}},
        messages=[{"role": "user", "content": ARCHITECT_PROMPT + goal}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("the architect model declined this request")
    text = next(b.text for b in resp.content if b.type == "text")
    spec = json.loads(text)
    spec["_usage"] = {"model": ARCHITECT_MODEL, "in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return spec


def _has(text, *words):
    return any(re.search(r"\b" + w, text) for w in words)


def spec_from_rules(goal: str) -> dict:
    """Keyword-based stand-in for the Claude architect when no ANTHROPIC_API_KEY is set."""
    g = goal.lower().replace("-", " ").replace("_", " ")
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
        "needs_documents": task == "grounded_qa" or _has(g, "our docs", "knowledge base", "help cent", "documentation", "manuals", "wiki",
                                                          "handbook", "faq", "policy docs", "policies", "runbook", "documents",
                                                          "pdfs", "contracts", "filings", "quarterly report", "annual report", "our reports"),
        "document_size": ("large" if _has(g, "knowledge base", "help cent", "documentation", "docs", "wiki", "manuals", "all our")
                          else "small") if task == "grounded_qa" or _has(g, "our docs", "knowledge base", "help cent", "documentation") else "none",
        "needs_tools": task in ("tool_actions", "research") or _has(g, " api", "apis", "mcp"),  # research needs search tools
        "needs_memory": task in ("conversation", "tool_actions") or _has(g, "remember", "history", "follow-up", "multi-turn"),
        "multi_step": task in ("research",) or _has(g, "multi step", "several steps", "then ", "workflow", "sub agent", "specialist", "hand off", "hands"),
        "handles_personal_data": _has(g, "customer", "patient", "employee", "user", "email", "phone", "account",
                                      "personal", "payment", "hr "),
        "data_class": ("regulated" if _has(g, "patient", "medical", "health record", "hipaa", "payment card", "credit card", "pci")
                       else None),
        "needs_images": _has(g, "image", "photo", "screenshot", "scan", "scanned", "picture", "diagram"),
        "steps_known": not (task == "research" or _has(g, "open ended", "figure out", "explore", "investigat", "whatever it takes",
                                                        "autonomous", "decide how", "work out how")),
        "request_types": len(labels) if labels else sum(1 for w in ("answer", "look", "issue", "cancel", "updat", "book", "refund", "reset",
                                                                    "track", "chang", "creat", "hand", "escalat", "schedul") if _has(g, w)),
        "specialists": _has(g, "sub agent", "subagent", "specialist", "hand off", "hands off", "handoff", "hands complex",
                            "multiple agents", "team of agents"),
        "multi_hop": _has(g, "compare", "across", "combine", "reconcil", "multi hop", "several documents", "multiple sources", "cross reference"),
        "docs_change_often": _has(g, "daily", "frequently", "often", "constantly", "latest", "change every", "changes every"),
        "structured_data": _has(g, "database", "sql", "warehouse", "tables", "spreadsheet", "analytics", "metrics", "kpi"),
        "tool_count": int(_num_tools.group(1)) if (_num_tools := re.search(r"(\d+)\s*(?:\w+\s+)?(?:tools|apis|endpoints|integrations|mcp servers)", g)) else
                      (max(3, sum(1 for w in ("look", "issue", "cancel", "updat", "book", "refund", "reset", "track", "creat", "send", "schedul", "provision")
                                  if _has(g, w))) if task == "tool_actions" or _has(g, " api", "apis", "mcp", "tool") else 2 if task == "research" else 0),
        "integration": "mcp" if _has(g, "mcp") else "direct",
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
    # Real prompt size: the test cases are short, production prompts aren't. Editable on the page.
    num = lambda k, d: max(0, int(s[k])) if isinstance(s.get(k), (int, float)) else d  # noqa: E731
    s["system_prompt_tokens"] = num("system_prompt_tokens", constraints.DEFAULT_SYSTEM_PROMPT.get(s["task_type"], 1000))
    s["context_tokens"] = num("context_tokens", 3000 if s.get("needs_documents") else 0)
    s["document_tokens"] = num("document_tokens", 30000 if s["document_size"] == "small" else 0)
    s["history_tokens"] = num("history_tokens", 2000 if s.get("needs_memory") else 0)
    s["output_tokens"] = num("output_tokens", 0) or None
    s["prompt_caching"] = s.get("prompt_caching") is not False
    s["peak_factor"] = min(50.0, max(1.0, float(s.get("peak_factor") or 3)))
    s["data_class"] = s.get("data_class") if s.get("data_class") in ("public", "internal", "confidential", "regulated") else (
        "confidential" if s.get("handles_personal_data") else "internal")
    s["residency"] = s.get("residency") if s.get("residency") in ("any", "US", "EU", "APAC") else "any"
    s["needs_images"] = bool(s.get("needs_images"))
    s["existing"] = [x for x in (s.get("existing") or []) if x in ("gateway", "runtime", "cache", "vector_store")]
    s["commitment"] = s.get("commitment") if s.get("commitment") in policy.APPROVED_PLATFORMS else None
    s["engineer_week_usd"] = float(s["engineer_week_usd"]) if s.get("engineer_week_usd") else None
    # Agentic system shape. Defaults come from knowledge/agentic.json; the page can override each one.
    A = constraints.AGENTIC["defaults"]
    s["tool_count"] = min(500, num("tool_count", 5 if s.get("needs_tools") else 0))
    if s["tool_count"] > 0:
        s["needs_tools"] = True
    s["integration"] = s.get("integration") if s.get("integration") in ("direct", "mcp", "both") else "direct"
    s["mcp_servers"] = num("mcp_servers", max(1, -(-s["tool_count"] // 5)) if s["integration"] != "direct" and s["tool_count"] else 0)
    s["tool_calls_per_request"] = max(1, num("tool_calls_per_request", A["tool_calls_per_request"])) if s["needs_tools"] else 0
    s["api_latency_ms"] = num("api_latency_ms", A["api_latency_ms"])
    s["steps_known"] = s.get("steps_known") is not False and s["task_type"] != "research"
    s["request_types"] = max(1, num("request_types", len(s["labels"]) or 1))
    for k in ("specialists", "multi_hop", "docs_change_often", "structured_data"):
        s[k] = bool(s.get(k))
    if s["specialists"]:
        s["multi_step"] = True
    s["multi_model"] = s.get("multi_model") is not False
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
PRICE_MULT: dict = {}  # what-if price changes, set only for the duration of a scenario


def _cost(r, price):
    cached = r.get("cached_tokens", 0)
    tokens = ((r["in_tokens"] - cached) * price["in"] + cached * price["in"] * r.get("cache_mult", 1.0)
              + r["out_tokens"] * price["out"])
    return tokens / 1e6 * PRICE_MULT.get(price["id"], 1.0)


def _p(a, q):
    if not a:
        return 0.0
    s = sorted(a)
    return s[min(len(s) - 1, int(q * len(s)))]


def _agg(rows):
    """rows: per test case {correct, halluc, cost, ms}. correct None = ungraded."""
    graded = [r for r in rows if r["correct"] is not None]
    n = len(graded)
    right = sum(1 for r in graded if r["correct"])
    lo, hi = constraints.wilson(right, n)
    return {
        "n": n, "acc_lo": lo, "acc_hi": hi,
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


def _turn_rows(res, model, turns, grow_tokens, steps=0, step_ms=0):
    """Per-case outcomes for a model that takes `turns` calls, each seeing `grow_tokens` more context than the
    last (tool results or search passages), plus `steps` external calls of `step_ms` each."""
    price = catalog.MODEL_BY_ID[model]
    out = []
    for r in res[model]:
        cost = sum(_cost({**r, "in_tokens": r["in_tokens"] + k * grow_tokens}, price) for k in range(turns))
        out.append({"correct": (False if r["error"] and r["correct"] is not None else r["correct"]),
                    "halluc": bool(r.get("hallucinated")), "cost": cost, "ms": turns * r["latency_ms"] + steps * step_ms,
                    "conf": r.get("confidence"), "error": bool(r["error"]), "calls": turns})
    return out


def compose(arch, models, res, jev, spec):
    """Per-test-case outcomes for one architecture built from measured single-call results.

    `res[model]` is the list of per-case results for that model; `jev` the per-case Jev results (or None).
    Returns (rows, evidence) where evidence is 'measured', 'composed' or 'estimated'. Agentic designs are built
    from the measured per-call results and the system shape you gave (tool calls, API latency), so their
    formula is visible, but the number of turns is your estimate, not a measurement.
    """
    A = constraints.AGENTIC["defaults"]
    api_ms, n_calls = spec["api_latency_ms"], spec["tool_calls_per_request"]
    if arch in ("single_call", "rag", "batch", "long_context"):
        rows = _direct_rows(res, models["primary"])
        if arch == "rag":  # retrieval step; cases carry the reference text, i.e. retrieval found the right passage
            rows = [{**r, "ms": r["ms"] + A["search_latency_ms"]} for r in rows]
        if arch == "batch":
            rows = [{**r, "cost": r["cost"] * 0.5, "ms": 30 * 60 * 1000} for r in rows]
        return rows, "measured" if arch in ("single_call", "long_context") else "composed"
    if arch in ("cascade", "rag_cascade"):
        small, large = _direct_rows(res, models["small"]), _direct_rows(res, models["primary"])
        rows = []
        for sm, g in zip(small, large):
            if not sm["error"] and sm["conf"] is not None and sm["conf"] >= 80:
                rows.append({**sm, "escalated": False, "calls": 1})
            else:
                rows.append({**g, "cost": sm["cost"] + g["cost"], "ms": sm["ms"] + g["ms"], "escalated": True, "calls": 2})
        if arch == "rag_cascade":
            rows = [{**r, "ms": r["ms"] + A["search_latency_ms"]} for r in rows]
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
    small = _direct_rows(res, models["small"])
    if arch == "workflow":  # classify (small) -> handler (main); tools are called by code, not by the model
        return [{**g, "cost": g["cost"] + sm["cost"], "ms": g["ms"] + sm["ms"] + n_calls * api_ms, "calls": 2}
                for g, sm in zip(base, small)], "estimated"
    if arch == "router":  # router (small) -> handler; with multi-model, confident simple requests stay on the small model
        rows = []
        for g, sm in zip(base, small):
            simple = spec["multi_model"] and not sm["error"] and sm["conf"] is not None and sm["conf"] >= 80
            h = sm if simple else g
            rows.append({**h, "cost": sm["cost"] * 0.3 + h["cost"], "ms": sm["ms"] * 0.5 + h["ms"] + n_calls * api_ms,
                         "calls": 2, "escalated": not simple})
        return rows, "estimated"
    if arch == "evaluator_optimizer":  # draft + review + one revision
        return [{**g, "cost": g["cost"] * 2.6, "ms": g["ms"] * 2.5, "calls": 3} for g in base], "estimated"
    if arch == "tool_agent":  # one model call per tool call plus the final answer; each turn reads the tool results so far
        return _turn_rows(res, models["primary"], n_calls + 1, A["tool_result_tokens"], n_calls, api_ms), "estimated"
    if arch == "agentic_rag":  # search, read, search again, answer
        k = A["agentic_searches"]
        return _turn_rows(res, models["primary"], k + 1, spec["context_tokens"] or 2000, k, A["search_latency_ms"]), "estimated"
    if arch == "multi_agent":  # lead plans, workers run in parallel (each a small tool loop), lead merges
        w, wt = A["workers"], max(2, n_calls // A["workers"] + 1)
        lead = _turn_rows(res, models["primary"], 2, w * 500)
        worker = _turn_rows(res, models["small"], wt, A["tool_result_tokens"], max(1, n_calls // w), api_ms)
        return [{**l, "cost": l["cost"] + w * wk["cost"], "ms": l["ms"] + wk["ms"] * 1.2, "calls": 2 + w * wt}
                for l, wk in zip(lead, worker)], "estimated"
    raise ValueError(arch)


ANSWER_ONLY = ("single_call", "cascade", "decision_model", "rag", "rag_cascade", "long_context", "batch",
               "evaluator_optimizer", "agentic_rag")


def applicable(spec):
    """Which architectures fit the goal, and why the others don't. Every rule here is also listed, with its
    source, in the decision trace (`decision_trace`)."""
    TH = constraints.AGENTIC["thresholds"]
    t, why_not, ok = spec["task_type"], {}, []
    labels, docs, tools = bool(spec["labels"]), spec["needs_documents"], spec["needs_tools"]
    data = docs or spec["structured_data"]
    multi = spec["multi_step"] or t == "research"
    single_step = not tools and not multi
    rules = {
        "single_call": (not tools and not multi, "the task needs actions or several steps, which one call can't do"),
        "cascade": (not tools and not multi, "the task needs actions or several steps"),
        "decision_model": (labels and not tools, "the answer isn't one of a fixed set of labels"),
        "rag": (docs and not spec["multi_hop"], "answers don't need to come from your own documents" if not docs
                else "answers combine several documents, and one search pass often misses part of them"),
        "rag_cascade": (docs and not spec["multi_hop"], "answers don't need to come from your own documents" if not docs
                        else "answers combine several documents, and one search pass often misses part of them"),
        "agentic_rag": (data and (spec["multi_hop"] or spec["structured_data"]),
                        "one search per question is enough here" if data else "answers don't need your documents or data"),
        "workflow": ((multi or tools or t in ("summarization", "generation")) and spec["steps_known"],
                     "the steps vary per request, so a fixed chain can't cover them" if not spec["steps_known"] else "the task is a single step"),
        "router": (spec["request_types"] >= TH["router_at_request_types"],
                   f"requests don't split into {TH['router_at_request_types']}+ distinct types that need different handling"),
        "tool_agent": (tools, "the agent doesn't need to call your systems or take actions"),
        "evaluator_optimizer": (t in ("generation", "summarization", "research") and not tools,
                                "answers are short and checkable, so a review loop adds cost without adding quality"),
        "multi_agent": ((t == "research" or spec["specialists"] or spec["tool_count"] >= TH["split_agents_at_tools"])
                        and spec["latency"] != "realtime",
                        "one agent can handle this: no specialist domains and fewer than "
                        f"{TH['split_agents_at_tools']} tools" if spec["latency"] != "realtime" else "someone needs the answer in real time"),
        "long_context": (docs and spec["document_size"] == "small" and not spec["multi_hop"],
                         "your documents are too large to send with every request" if docs
                         else "answers don't need to come from your own documents"),
        "batch": (spec["latency"] == "background" and single_step,
                  "someone is waiting for the answer" if spec["latency"] != "background"
                  else "batch APIs run single calls; this task needs several steps or tools"),
    }
    if data:  # grounded tasks must retrieve; a bare call can't see the documents or tables
        what = "documents" if docs else "tables"
        rules["single_call"] = (False, f"a single call can't see your {what}; it needs a retrieval step")
        rules["cascade"] = (False, f"without retrieval the model can't see your {what}")
        rules["decision_model"] = (False, f"without retrieval the model can't see your {what}")
        rules["batch"] = (False, f"batch calls can't search your {what}") if rules["batch"][0] else rules["batch"]
    if tools:  # it must act: designs that only answer can't do the job, whatever they cost
        for a in ANSWER_ONLY:
            rules[a] = (False, "it can't search as it works; research needs tools in a loop" if t == "research"
                        else "it only answers questions; this agent must take actions in your systems")
    if not spec["multi_model"]:
        for a in ("cascade", "rag_cascade"):
            if rules[a][0]:
                rules[a] = (False, "it needs two models, and multi-model is switched off")
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


def build_design(arch, models, variant, res, jev, spec):
    """One architecture with specific models: metrics composed from measured results, checked against requirements."""
    if not spec["multi_model"]:  # one model for every role
        models = {**models, "small": models["primary"]}
    elif models.get("small") in res and models["small"] != models["primary"] and arch != "cascade" and arch != "rag_cascade":
        own = lambda m: _agg(_direct_rows(res, m))["cost_per_req"]  # noqa: E731
        if own(models["small"]) >= own(models["primary"]):  # a "small" role only makes sense if it's cheaper
            models = {**models, "small": models["primary"]}
    res = constraints.inflate(res, spec, arch, list(models.values()))
    rows, evidence = compose(arch, models, res, jev, spec)
    m = _agg(rows)
    addons = addons_for(arch, spec)
    infra = sum(a["per_1k"] for a in addons) / 1000
    cache = next((a for a in addons if a["id"] == "cache"), None)
    per_req = m["cost_per_req"] * (1 - spec["repeat_rate"] if cache else 1) + infra
    p95 = m["p95_ms"] + sum(a["ms"] for a in addons)
    per_req += constraints.embedding_cost_per_req(arch, spec)
    harness = harness_for(arch, models, res, rows, spec) if spec.get("harness") else None
    if harness:
        per_req += harness["cost_per_req"]
        p95 += harness["ms"]
    own = res[models["primary"]]
    tokens_per_call = statistics.fmean(r["in_tokens"] + r["out_tokens"] for r in own) if own else 0
    cached_share = statistics.fmean(r["cached_tokens"] / r["in_tokens"] for r in own if r["in_tokens"]) if own else 0
    t = constraints.tco(arch, spec, bool(harness))
    d = {
        "arch": arch, **ARCHITECTURES[arch],
        "sources": [PATTERN_KNOWLEDGE["sources"][k] for k in ARCHITECTURES[arch].get("sources", [])],
        "variant": variant, "models": dict(models), "evidence": evidence, "addons": addons, "harness": harness,
        "accuracy": m["accuracy"], "right": m["right"], "n": m["n"], "halluc": m["halluc"],
        "acc_lo": m["acc_lo"], "acc_hi": m["acc_hi"], "tokens_per_call": tokens_per_call, "cached_share": cached_share,
        "infra": t["infra"], "infra_monthly": t["infra_monthly"], "build_weeks": t["build_weeks"], "build_usd": t["build_usd"],
        "p50_ms": m["p50_ms"], "p95_ms": p95, "cost_per_req": per_req, "monthly": per_req * spec["requests_per_day"] * 30,
        "escalation_rate": (sum(1 for r in rows if r.get("escalated")) / len(rows)) if rows and arch in ("cascade", "rag_cascade", "decision_model") else None,
    }
    d["total_monthly"] = d["monthly"] + d["infra_monthly"]
    d["calls_per_req"] = statistics.fmean(r.get("calls", 1) for r in rows) if rows else 1
    d["flow"] = flow_for(arch, spec, d["flow"])
    d["roles"] = roles_for(arch, d["models"], spec)
    d["peak"] = constraints.peak(d, spec, tokens_per_call)
    d["context_needed"] = constraints.required_context(spec, arch)
    d["checks"] = checks(d, spec)
    d["passes"] = all(c["pass"] for c in d["checks"])
    return d


def model_table(first, spec, res, jev, picks, mode):
    """Every model offering priced under the #1 architecture, so the table and the recommendation agree.
    Tested models use their measured results; 'reported' ones get a cost estimate from a tested model of the
    same tier (same token counts, their own price) and no accuracy."""
    rows = []
    def row(m, d, tested, ref=None):
        info = catalog.MODEL_BY_ID[m]
        acc = d["accuracy"] if tested else None
        return {"id": m, "label": info["label"], "maker": info["maker"], "platform": info["platform"],
                "tier": info["tier"], "in": info["in"], "out": info["out"], "status": info.get("status"),
                "verified_on": info.get("verified_on"), "source": info.get("source"), "note": info.get("review_note"),
                "tested": tested, "estimated_from": ref, "mode": mode.get(m, "simulated") if tested else "estimate", "accuracy": acc, "halluc": d["halluc"] if tested else None,
                "acc_lo": d["acc_lo"] if tested else None, "acc_hi": d["acc_hi"] if tested else None,
                "context": constraints.caps(m)["context"], "retirement": constraints.retirement(m),
                "cached_share": d["cached_share"], "failed": [c["key"] for c in d["checks"] if not c["pass"]] if tested else [],
                "p95_ms": d["p95_ms"] if tested else None, "monthly": d["monthly"], "cost_per_req": d["cost_per_req"],
                "cost_per_1k_correct": (d["cost_per_req"] * 1000 / acc) if acc else None,
                "own_cost_per_req": _agg(_direct_rows(res if tested else {**res, m: res[ref]}, m))["cost_per_req"],
                "meets": d["passes"] if tested else None, "checks": d["checks"] if tested else []}
    for m in res:
        d = build_design(first["arch"], {**first["models"], "primary": m}, "table", res, jev, spec)
        rows.append(row(m, d, True))
    tested_ids = list(res)
    for info in catalog.MODELS:
        # Not tested this run: reported models, and (on a shortlist run) models not compared yet.
        if info["id"] in res or not constraints.screen([info["id"]], spec)[0]:
            continue
        same = [t for t in tested_ids if catalog.MODEL_BY_ID[t]["tier"] == info["tier"]] or tested_ids
        ref = next((t for t in same if catalog.MODEL_BY_ID[t]["maker"] == info["maker"]), same[0])
        d = build_design(first["arch"], {**first["models"], "primary": info["id"]}, "table", {**res, info["id"]: res[ref]}, jev, spec)
        rows.append(row(info["id"], d, False, ref))
    # Cheapest first; ties (e.g. when Jev answers nearly everything) go to the model that's cheaper on its own.
    rows.sort(key=lambda r: (round(r["monthly"], 2), r["own_cost_per_req"]))
    return rows


def choose_from_table(table, first, spec=None):
    """Primary: the cheapest tested model meeting every requirement (verified prices first; a model on the
    platform you have a spend commitment with wins if it's within 15%).
    Fallback: the cheapest qualifying model on a different platform, so an outage of one doesn't take both down."""
    tested = [r for r in table if r["tested"]]
    all_good = [r for r in tested if r["meets"]]
    live = [r for r in tested if r["mode"] == "live"]
    # Prefer rows measured live over simulated ones whenever any live row qualifies.
    if any(r["meets"] for r in live):
        tested = live
    good = [r for r in tested if r["meets"]]
    pool = [r for r in good if r["status"] == "verified"] or good
    # If nothing qualifies, the closest one must come from real measurements when there are any.
    primary = pool[0] if pool else max(live or tested, key=lambda r: (r["accuracy"] or 0, -(r["halluc"] or 0), -r["monthly"]))
    commit = spec and spec.get("commitment")
    if pool and commit and primary["platform"] != commit:
        on = [r for r in pool if r["platform"] == commit]
        if on and on[0]["monthly"] <= primary["monthly"] * 1.15:
            primary = on[0]
    # Fallback: qualifying on another platform, live-measured first, else qualifying in simulation.
    others = [r for r in good if r["platform"] != primary["platform"]] or [r for r in all_good if r["platform"] != primary["platform"]]
    others = ([r for r in others if r["status"] == "verified" and r["maker"] != primary["maker"]]
              or [r for r in others if r["status"] == "verified"] or others)
    if not others:
        others = sorted([r for r in (live if not pool and live else tested) if r["platform"] != primary["platform"]]
                        or [r for r in tested if r["platform"] != primary["platform"]],
                        key=lambda r: (-(r["accuracy"] or 0), r["monthly"]))
    return primary["id"], (others[0]["id"] if others else None), [r["id"] for r in good]


def cheaper_unverified(table, primary_id):
    """A cheaper model that met every requirement but whose price isn't verified, so it wasn't picked."""
    prim = next((r for r in table if r["id"] == primary_id), None)
    for r in table:
        if r["id"] == primary_id or not prim:
            break
        if r["tested"] and r["meets"] and r["status"] != "verified" and r["monthly"] < prim["monthly"] * 0.9:
            return {"id": r["id"], "label": r["label"], "platform": r["platform"], "monthly": r["monthly"]}
    return None


def rank_designs(spec, res, jev, mode=None, jev_blocked=False):
    """For every architecture that fits: price every model inside it, pick its best model and fallback,
    check it against the requirements; then rank the architectures and return the top 3 with reasons."""
    mode = mode or {}
    provider_of = lambda m: catalog.MODEL_BY_ID[m]["provider"]  # noqa: E731
    picks = pick_models(res, spec, provider_of)
    ok, why_not = applicable(spec)
    if jev is None and "decision_model" in ok:
        ok.remove("decision_model")
        why_not["decision_model"] = ("Jev (TypeSafe) is not on the company's approved vendor list" if jev_blocked
                                     else "Jev needs a fixed set of labels to choose from")
    designs = []
    for arch in ok:
        cascade = arch in ("cascade", "rag_cascade")
        base = {"primary": picks["best_large"] if cascade else picks["primary"], "small": picks["small"], "fallback": None}
        probe = build_design(arch, base, "probe", res, jev, spec)
        table = model_table(probe, spec, res, jev, picks, mode)
        if cascade:  # the escalation model has to be a bigger model than the small one
            table = [r for r in table if r["tier"] != "small" or not r["tested"]]
        if not any(r["tested"] for r in table):
            continue
        primary, fallback, good = choose_from_table(table, probe, spec)
        live_ok = [r for r in table if r["tested"] and r["meets"] and r["mode"] == "live"]
        d = build_design(arch, {**base, "primary": primary, "fallback": fallback}, "value", res, jev, spec)
        d["_table"], d["_good"], d["_cheaper"] = table, good, cheaper_unverified(table, primary)
        d["_live_ok"] = len(live_ok)
        designs.append(d)
    score_designs(designs, spec)
    designs.sort(key=lambda d: (not d["passes"], -d["score"]))
    top = designs[:3]
    table = []
    if top:
        first = top[0]
        table = first["_table"]
        picks.update(primary=first["models"]["primary"], fallback=first["models"]["fallback"], qualifying=first["_good"],
                     cheaper_unverified=first["_cheaper"], live_qualified=first["_live_ok"])
        prim = next(r for r in table if r["id"] == first["models"]["primary"])
        # Cheaper models that missed only on accuracy/hallucinations, but where the 95% range still reaches the
        # target: more test cases could make them qualify. And qualifying models the winner isn't clearly better than.
        picks["near_misses"] = [{"id": r["id"], "label": r["label"], "platform": r["platform"], "monthly": r["monthly"],
                                 "accuracy": r["accuracy"], "acc_hi": r["acc_hi"]}
                                for r in table if r["tested"] and not r["meets"] and r["monthly"] < prim["monthly"]
                                and set(r["failed"]) <= {"accuracy", "halluc"} and "accuracy" in r["failed"]
                                and (r["acc_hi"] or 0) >= spec["accuracy_target"]][:5]
        picks["committed"] = bool(spec["commitment"] and prim["platform"] == spec["commitment"])
        picks["retirement"] = prim["retirement"]
        picks["fallback_retirement"] = next((r["retirement"] for r in table if r["id"] == first["models"]["fallback"]), None)
        if first["models"]["fallback"]:
            fbd = build_design(first["arch"], {**first["models"], "primary": first["models"]["fallback"]}, "fallback", res, jev, spec)
            picks["fallback_design"] = {k: fbd[k] for k in ("accuracy", "right", "n", "halluc", "p95_ms", "monthly", "cost_per_req", "passes", "acc_lo", "acc_hi")}
    for d in designs:
        for k in ("_table", "_good", "_cheaper", "_live_ok"):
            d.pop(k, None)
    explain(top, spec, picks)
    system = system_design(top[0], spec) if top else None
    if system:  # say what each rule led to: the rank it got, or why it was ruled out
        rank = {d["arch"]: i + 1 for i, d in enumerate(designs)}
        for t in system["trace"]:
            t["effects"] = [{"arch": a, "name": ARCHITECTURES[a]["name"], "rank": rank.get(a), "of": len(designs),
                             "ruled_out": why_not.get(a)} for a in t.pop("patterns")]
    return {"top": top, "all": designs, "not_applicable": why_not, "models": picks, "model_table": table, "system": system}


def robustness(spec, res, jev, mode, jev_blocked, base):
    """Re-rank under what-if changes (no new model calls) and report whether the recommendation holds."""
    if not base["top"]:
        return []
    first = base["top"][0]
    arch, model = first["arch"], first["models"]["primary"]
    name = catalog.MODEL_BY_ID[model]["label"]
    prompt_keys = ("system_prompt_tokens", "context_tokens", "document_tokens", "history_tokens")
    runs = [
        ("10× the volume", {"requests_per_day": spec["requests_per_day"] * 10}, None),
        ("A tenth of the volume", {"requests_per_day": max(1, spec["requests_per_day"] // 10)}, None),
        ("Accuracy target 5 points higher", {"accuracy_target": min(1.0, spec["accuracy_target"] + 0.05)}, None)
        if spec["accuracy_target"] < 0.99 else None,
        ("Prompts twice as long", {k: spec[k] * 2 for k in prompt_keys}, None),
        (f"{name} costs 30% more", {}, {model: 1.3}),
        ("Without the production harness" if spec.get("harness") else "With a production harness",
         {"harness": not spec.get("harness")}, None),
        ("One model for every role" if spec["multi_model"] else "Allowing a different model per role",
         {"multi_model": not spec["multi_model"]}, None),
    ]
    out = []
    for run in filter(None, runs):
        label, change, prices = run
        PRICE_MULT.clear()
        PRICE_MULT.update(prices or {})
        try:
            r = rank_designs({**spec, **change}, res, jev, mode, jev_blocked)
        finally:
            PRICE_MULT.clear()
        t = r["top"][0] if r["top"] else None
        if t:
            out.append({"scenario": label, "arch": t["arch"], "arch_name": t["name"], "model": t["models"]["primary"],
                        "model_label": catalog.MODEL_BY_ID[t["models"]["primary"]]["label"], "monthly": t["total_monthly"],
                        "passes": t["passes"], "same_arch": t["arch"] == arch, "same_model": t["models"]["primary"] == model})
    return out


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
    if spec["needs_documents"] and arch not in ("rag", "rag_cascade", "long_context") and not spec.get("harness"):
        out.append({"id": "grounding", "name": "Grounding check", "why": "verify each answer is supported by your documents",
                    "ms": 0, "per_1k": 0})
    for a in out:  # already running it: reuse, no new cost
        if a["id"] in spec["existing"]:
            a.update(per_1k=0, existing=True, why=a["why"] + " (you already run one, so reuse it)")
    return out


def flow_for(arch, spec, flow):
    """The pattern's flow, filled in with this system's tools, integration and data sources."""
    tools = spec["tool_count"] if spec["needs_tools"] else 0
    via = {"direct": "", "mcp": " via MCP", "both": " via MCP + direct"}[spec["integration"]]
    search = "search your docs" if spec["needs_documents"] else ""
    sql = "query your tables" if spec["structured_data"] else ""
    apis = ("web search + fetch" if spec["task_type"] == "research" and tools <= 2 else f"{tools} APIs{via}") if tools else ""
    what = ", ".join(x for x in (search, sql, apis) if x)
    if arch == "tool_agent":
        return ["App", "Agent Runtime", f"LLM ⇄ {what or 'your tools'}", "Human approval" if spec["risk"] == "high" else "Answer"]
    if arch == "multi_agent" and tools:
        return ["App", "Lead agent (plans)", f"Specialist agents ⇄ {what}", "Lead merges"]
    if arch == "workflow" and (tools or search):
        return ["App", "Classify", "Tested handler" + (f" ({what})" if what else ""), "Check", "Human approval" if spec["risk"] == "high" else "Done"]
    if arch == "router":
        return ["App", "Router (small model)", f"{spec['request_types']} handlers" + (f" ({what})" if what else ""), "Check"]
    if arch == "agentic_rag":
        return ["App", f"LLM ⇄ {what or 'search tools'}", "Answer with citations"]
    return flow


ROLE_NAMES = {
    "single_call": [("answer", "primary")], "rag": [("answer from retrieved passages", "primary")],
    "long_context": [("answer", "primary")], "batch": [("answer (batch)", "primary")],
    "cascade": [("first try", "small"), ("escalation when unsure", "primary")],
    "rag_cascade": [("first try", "small"), ("escalation when unsure", "primary")],
    "decision_model": [("decision (Jev)", None), ("fallback when unsure", "primary")],
    "workflow": [("classify / extract", "small"), ("decide and draft", "primary")],
    "router": [("router", "small"), ("simple requests", "small"), ("complex requests", "primary")],
    "tool_agent": [("agent: plans and calls tools", "primary")],
    "agentic_rag": [("searches and answers", "primary")],
    "multi_agent": [("lead: plans and merges", "primary"), ("specialist agents", "small")],
    "evaluator_optimizer": [("writer", "primary"), ("reviewer", "primary")],
}


def roles_for(arch, models, spec):
    out = []
    for role, key in ROLE_NAMES[arch]:
        m = "jev" if key is None else models[key]
        if arch == "router" and role == "simple requests" and not spec["multi_model"]:
            m = models["primary"]
        out.append({"role": role, "model": m})
    if models.get("fallback"):
        out.append({"role": "fallback (another platform)", "model": models["fallback"]})
    return out


def _src(keys):
    return [PATTERN_KNOWLEDGE["sources"][k] for k in keys]


def decision_trace(spec):
    """The facts about the goal that decided which patterns were considered, each with the rule it triggered
    and the published guidance behind it. Shown on the page so a reviewer can check the reasoning."""
    R, TH = constraints.AGENTIC["rules"], constraints.AGENTIC["thresholds"]
    out = []
    EFFECT = {"acts": ["tool_agent", "workflow", "router", "multi_agent"], "uses_tools": ["tool_agent", "multi_agent"], "steps_known": ["workflow", "router"], "steps_vary": ["tool_agent"],
              "split_agents": ["multi_agent"], "single_agent": ["tool_agent"], "request_types": ["router"],
              "multi_hop": ["agentic_rag"], "retrieval_in_agent": [], "tool_search": [], "multi_model_on": ["cascade", "rag_cascade", "router"],
              "multi_model_off": []}
    add = lambda fact, key: out.append({"fact": fact, "rule": R[key]["text"], "sources": _src(R[key]["sources"]),  # noqa: E731
                                        "patterns": EFFECT.get(key, [])})
    if spec["needs_tools"]:
        if spec["task_type"] == "research":
            add(f"Needs search tools as it works ({spec['tool_count']})", "uses_tools")
        else:
            add(f"Takes actions in your systems ({spec['tool_count']} APIs/tools)", "acts")
        add("Steps are the same every time" if spec["steps_known"] else "Steps vary per request",
            "steps_known" if spec["steps_known"] else "steps_vary")
        if spec["specialists"] or spec["tool_count"] >= TH["split_agents_at_tools"]:
            add("Specialist domains" if spec["specialists"] else f"{spec['tool_count']} tools", "split_agents")
        else:
            add(f"{spec['tool_count']} tools, no separate specialist domains", "single_agent")
        if spec["tool_count"] >= TH["tool_search_at_tools"]:
            add(f"{spec['tool_count']} tools", "tool_search")
    elif spec["multi_step"]:
        add("Several steps", "steps_known" if spec["steps_known"] else "steps_vary")
    if spec["request_types"] >= TH["router_at_request_types"]:
        add(f"{spec['request_types']} distinct request types", "request_types")
    if spec["needs_documents"] and spec["needs_tools"]:
        add("Uses your documents and takes actions", "retrieval_in_agent")
    if spec["multi_hop"]:
        add("Answers combine several documents or sources", "multi_hop")
    add("Multi-model allowed" if spec["multi_model"] else "Multi-model switched off", "multi_model_on" if spec["multi_model"] else "multi_model_off")
    return out


def integration_plan(spec, d):
    """How to connect the tools: MCP or direct function tools, and what that costs and needs."""
    if not spec["needs_tools"]:
        return None
    I, TH = constraints.AGENTIC["integration"], constraints.AGENTIC["thresholds"]
    n = spec["tool_count"]
    if spec["integration"] in ("mcp", "both"):
        rec, why = ("MCP servers" if spec["integration"] == "mcp" else "MCP plus direct tools"), ["your goal or settings say the tools are exposed via MCP"]
    elif n >= TH["mcp_at_tools"]:
        rec, why = "Consider MCP", [f"{n} tools are easier to govern behind MCP servers"] + I["mcp_when"][:2]
    else:
        rec, why = "Direct function tools", I["direct_when"]
    seen = constraints.tools_seen(spec, d["arch"])
    return {"recommendation": rec, "why": why, "tool_count": n, "tools_per_call": seen,
            "definition_tokens": constraints.tool_definition_tokens(spec, d["arch"]),
            "tool_search": n >= TH["tool_search_at_tools"] and d["arch"] in constraints.TOOL_ARCHS,
            "mcp_servers": spec["mcp_servers"], "calls_per_request": spec["tool_calls_per_request"],
            "api_latency_ms": spec["api_latency_ms"],
            "controls": I["controls"] if spec["integration"] != "direct" or n >= TH["mcp_at_tools"] else I["controls"][1:3] + I["controls"][4:],
            "sources": _src(I["sources"])}


def retrieval_plan(spec, d):
    """Which retrieval approach fits, and the components it needs, each with the reason it applies."""
    if not (spec["needs_documents"] or spec["structured_data"]):
        return None
    C, arch = constraints.AGENTIC["retrieval"]["components"], d["arch"]
    strategy = {"long_context": "Whole documents in the prompt, cached",
                "rag": "Classic RAG: one search per question", "rag_cascade": "Classic RAG with small-to-large escalation",
                "agentic_rag": "Agentic retrieval: search, read, search again"}.get(
        arch, "Retrieval as a tool the agent calls" if arch in ("tool_agent", "multi_agent")
        else "Retrieval inside the handlers that need it" if arch == "router" else "Retrieval as a step in the workflow")
    pick = []
    if spec["needs_documents"] and arch != "long_context":
        pick += ["chunking", "hybrid"]
        if spec["document_size"] == "large" and spec["accuracy_target"] >= 0.9:
            pick.append("rerank")
        if spec["multi_hop"] or spec["data_class"] in ("confidential", "regulated"):
            pick.append("filters")
    if spec["needs_documents"] and spec["docs_change_often"]:
        pick.append("freshness")
    if spec["risk"] != "low" or spec["max_hallucination"] <= 0.02:
        pick.append("citations")
    if spec["structured_data"]:
        pick.append("sql")
    if spec["needs_documents"] and arch != "long_context":
        pick.append("eval")
    return {"strategy": strategy, "caveat": constraints.AGENTIC["retrieval"]["caveat"],
            "components": [{"id": k, "name": C[k]["name"], "why": C[k]["why"], "sources": _src(C[k]["sources"])} for k in pick]}


def system_design(d, spec):
    return {"trace": decision_trace(spec), "roles": d["roles"], "integration": integration_plan(spec, d),
            "retrieval": retrieval_plan(spec, d)}


def harness_for(arch, models, res, rows, spec):
    """The production controls around one architecture. Controls that call a model are priced from this run's
    measured results (a small-model call per request, retries at the measured failure rate); the rest add no
    model cost. Every figure carries the basis it was worked out from."""
    main = _agg(_direct_rows(res, models["primary"]))
    small_id = models.get("small") if models.get("small") in res else models["primary"]
    small = _agg(_direct_rows(res, small_id))
    s_name = catalog.MODEL_BY_ID[small_id]["label"]
    tools = spec["needs_tools"] or arch in ("tool_agent", "multi_agent")
    fail = (sum(1 for r in rows if r.get("error")) / len(rows)) if rows else 0.0
    sample = HARNESS["assumptions"]["live_eval_sample_rate"]
    n = len(rows)
    plan = {
        "input_guard": (True, small["cost_per_req"], small["p50_ms"] if tools else 0,
                        f"one {s_name} call per request, " + ("finished before any action is taken" if tools
                                                              else "run alongside the main call so it adds no wait")),
        "output_check": (True, fail * main["cost_per_req"],
                         main["p50_ms"] if fail >= HARNESS["assumptions"]["retry_latency_threshold"] else 0,
                         f"retries the {round(fail * 100)}% of answers that failed in testing" if fail
                         else "no answer failed in testing, so retries add nothing measurable"),
        "grounding": (spec["needs_documents"], small["cost_per_req"], small["p50_ms"],
                      f"one {s_name} call per answer, before it is shown"),
        "tool_gate": (tools, 0.0, 0, "approvals wait for a person, outside the request"),
        "mcp_governance": (spec["needs_tools"] and spec["integration"] != "direct", 0.0, 0, "configuration and review; no model cost"),
        "limits": (arch in LOOPING, 0.0, 0, "configuration in the agent runtime"),
        "calibration": (arch in ("cascade", "rag_cascade"), 0.0, 0, "computed from the traces"),
        "failover": (True, 0.0, 0, "no cost unless the main model fails"),
        "tracing": (True, 0.0, 0, "log storage is part of hosting"),
        "regression": (True, 0.0, 0, f"re-runs the {n} test cases at the main model's price: "
                                      + (f"about {_usd(n * main['cost_per_req'])}" if n * main["cost_per_req"] >= 0.01 else "under a cent")
                                      + " per release"),
        "live_eval": (True, sample * main["cost_per_req"], 0,
                      f"{round(sample * 100)}% of requests graded by a judge call about the size of the main call (assumption)"),
    }
    controls = []
    for cid, (applies, cost, ms, basis) in plan.items():
        if not applies:
            continue
        c = HARNESS["controls"][cid]
        controls.append({"id": cid, "name": c["name"], "layer": c["layer"], "layer_name": HARNESS["layers"][c["layer"]],
                         "what": c["what"], "cost_per_req": cost, "monthly": cost * spec["requests_per_day"] * 30,
                         "ms": ms, "basis": basis, "calls_model": cost > 0,
                         "sources": [PATTERN_KNOWLEDGE["sources"][k] for k in c["sources"]]})
    cost = sum(c["cost_per_req"] for c in controls)
    return {"controls": controls, "cost_per_req": cost, "monthly": cost * spec["requests_per_day"] * 30,
            "ms": sum(c["ms"] for c in controls), "per_release": n * main["cost_per_req"],
            "sample_rate": sample, "evidence": "estimated"}


def harness_advice(spec):
    """Whether this goal should run with a production harness, and why."""
    why = []
    if spec["needs_tools"]:
        why.append("it takes actions in your systems")
    if spec["risk"] == "high":
        why.append("a wrong answer has high impact")
    if spec["handles_personal_data"]:
        why.append("it handles personal data")
    return {"on": bool(spec.get("harness")), "recommended": bool(why), "reasons": why}


def checks(d, spec):
    out = []
    if d["accuracy"] is not None:
        out.append({"key": "accuracy", "label": "Accuracy", "pass": d["accuracy"] >= spec["accuracy_target"],
                    "value": d["accuracy"], "target": spec["accuracy_target"]})
        out.append({"key": "halluc", "label": "Hallucinations", "pass": d["halluc"] <= spec["max_hallucination"],
                    "value": d["halluc"], "target": spec["max_hallucination"]})
    if out:  # sure = the whole 95% range clears the target, not just the point estimate
        out[0]["sure"] = d["acc_lo"] is not None and d["acc_lo"] >= spec["accuracy_target"]
        out[0]["lo"], out[0]["hi"] = d["acc_lo"], d["acc_hi"]
    out.append({"key": "latency", "label": "Latency p95", "pass": d["p95_ms"] <= spec["latency_budget_ms"],
                "value": d["p95_ms"], "target": spec["latency_budget_ms"]})
    if spec.get("monthly_budget"):
        out.append({"key": "cost", "label": "Monthly cost", "pass": d["total_monthly"] <= spec["monthly_budget"],
                    "value": d["total_monthly"], "target": spec["monthly_budget"]})
    ctx = constraints.caps(d["models"]["primary"])["context"]
    if ctx and d["context_needed"] > ctx * 0.25:
        out.append({"key": "context", "label": "Context window", "pass": d["context_needed"] <= ctx,
                    "value": d["context_needed"], "target": ctx})
    if d["peak"]["quota"]:
        out.append({"key": "quota", "label": "Peak load", "pass": not d["peak"]["over"],
                    "value": d["peak"]["tpm"], "target": d["peak"]["quota"].get("tpm")})
    return out


def score_designs(designs, spec):
    """Among designs, reward accuracy, low cost, low latency and simplicity (start simple).
    When nobody waits for the answer, speed doesn't count and its weight moves to cost."""
    if not designs:
        return
    w_cost, w_lat = (0.40, 0.0) if spec["latency"] == "background" else (0.25, 0.15)
    min_cost = min(d["total_monthly"] for d in designs) or 1e-9
    min_lat = min(d["p95_ms"] for d in designs) or 1
    for d in designs:
        acc = d["accuracy"] if d["accuracy"] is not None else 0.8
        hall_pen = min(1.0, (d["halluc"] or 0) * 5)
        d["score"] = round(100 * (0.35 * acc * (1 - hall_pen * 0.5)
                                  + w_cost * min(1.0, min_cost / max(d["total_monthly"], 1e-9))
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
    if first.get("harness"):
        h = first["harness"]
        reasons.append(f"Includes a production harness of {len(h['controls'])} controls: {_usd(h['monthly'])}/month"
                       + (f" and {_ms(h['ms'])} of added wait" if h["ms"] >= 1 else " with no added wait") + ".")
    arch_why = {
        "single_call": "The task is a single step, so extra moving parts would add cost and failure points without adding accuracy.",
        "cascade": f"Most requests are easy enough for {label(first['models']['small'])}; only "
                   f"{_pct(first['escalation_rate'])} needed {label(first['models']['primary'])}.",
        "decision_model": f"The answer is one of {len(spec['labels'])} labels, so Jev decides "
                          f"{_pct(1 - (first['escalation_rate'] or 0))} of requests in about a tenth of a second for a fraction of a cent.",
        "long_context": "Your documents are small enough to send with every request, so you get grounded answers without building a search index.",
        "rag_cascade": "Answers come from your documents, and most questions are easy enough for the small model; only the unsure ones pay for the large model.",
        "rag": "Answers have to come from your documents; retrieving the right passages first keeps them grounded and lets the model say 'unknown'.",
        "workflow": "The steps are the same every time, so a fixed workflow is easier to test, cheaper and more predictable than an agent that decides its own steps.",
        "tool_agent": f"The agent has to act in your systems and the steps vary per request; priced at about {spec['tool_calls_per_request']} tool calls per request.",
        "router": f"Requests split into {spec['request_types']} types; each gets its own tested handler"
                  + (", and simple ones stay on the small model." if spec["multi_model"] else "."),
        "agentic_rag": "Answers combine several documents or sources, so the model searches again after reading the first results.",
        "multi_agent": "The task is broad and open-ended; parallel specialists cover more ground than one model.",
        "batch": "Nobody is waiting for the result, so batch pricing halves the model cost.",
        "evaluator_optimizer": "Written quality matters here, and a second review pass catches issues a single draft misses.",
    }[first["arch"]]
    reasons.append(arch_why)
    if first.get("infra_monthly"):
        reasons.append(f"Plus about {_usd(first['infra_monthly'])}/month of fixed infrastructure "
                       f"({', '.join(i['name'].lower() for i in first['infra'] if i['monthly'])}).")
    if first.get("cached_share", 0) >= 0.2:
        reasons.append(f"Prompt caching covers {round(first['cached_share'] * 100)}% of input tokens, priced at the cached rate.")
    if picks.get("committed"):
        reasons.append(f"Runs on {catalog.MODEL_BY_ID[first['models']['primary']]['platform']}, where you have a spend commitment.")
    reasons.append(f"{label(first['models']['primary'])} is the cheapest model that meets every requirement in this architecture."
                   if first["models"]["primary"] in picks["qualifying"]
                   else f"{label(first['models']['primary'])} was the most accurate model tested, but none met every requirement.")
    first["why"] = reasons
    for d in top[1:]:
        t = []
        if d["accuracy"] is not None and first["accuracy"] is not None:
            diff = round((d["accuracy"] - first["accuracy"]) * 100)
            noise = (d["acc_lo"] is not None and first["acc_lo"] is not None
                     and d["acc_lo"] <= first["acc_hi"] and first["acc_lo"] <= d["acc_hi"])
            if diff:
                t.append(f"{'+' if diff > 0 else ''}{diff} pts accuracy" + (f" (within noise at {d['n']} cases)" if noise else ""))
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
        if d.get("harness") and first.get("harness"):
            dh = len(d["harness"]["controls"]) - len(first["harness"]["controls"])
            if dh > 0:
                t.append(f"{dh} more harness control{'s' if dh > 1 else ''} to run")
        dc = d["complexity"] - first["complexity"]
        if dc:
            t.append("more to build and operate" if dc > 0 else "simpler to build and operate")
        if d["arch"] in ("cascade", "rag_cascade", "decision_model") and d["escalation_rate"] is not None:
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
    models = [{**{k: m.get(k) for k in ("id", "label", "maker", "platform", "provider", "tier", "in", "out", "status",
                                         "verified_on", "source", "review_note", "price_source", "callable")},
               "age_days": age(m.get("verified_on"))} for m in mk["models"]]
    stale = [m for m in models if m["status"] != "verified" or (m["age_days"] or 0) > mk["review_every_days"]]
    reported = [m for m in models if m["status"] == "reported"]
    return {
        "models": models, "watchlist": mk.get("watchlist", []), "model_review_every_days": mk["review_every_days"],
        "models_verified": len(models) - len(stale), "models_total": len(models), "models_reported": len(reported),
        "models_needing_review": [m["id"] for m in stale],
        "patterns_reviewed_on": pk["reviewed_on"], "patterns_age_days": age(pk["reviewed_on"]),
        "patterns_stale": age(pk["reviewed_on"]) > pk["review_every_days"],
        "principles": pk["principles"], "sources": pk["sources"],
        "jev": {"label": catalog.JEV["label"], "in": catalog.JEV["in"], "status": "needs_review",
                "note": "Price from third-party launch write-ups; confirm with TypeSafe.", "source": "https://typesafe.ai"},
    }
