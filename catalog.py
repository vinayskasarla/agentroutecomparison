"""Static catalog: models + list prices (from knowledge/models.json), the eval suite, and the call paths.

Prices are USD per 1M tokens (list price). They are defaults only — the UI lets
you override them to match your negotiated contract before running.
"""
import json
import os

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
# Models and prices live in knowledge/models.json so they can be reviewed and dated in one place.
MODEL_KNOWLEDGE = json.load(open(os.path.join(KNOWLEDGE_DIR, "models.json")))
MODELS = MODEL_KNOWLEDGE["models"]

PROVIDERS = {
    "openai": {"label": "OpenAI GPT", "env": "OPENAI_API_KEY", "small": "gpt-5-mini", "fallback": "claude-sonnet-5"},
    "anthropic": {"label": "Anthropic Claude", "env": "ANTHROPIC_API_KEY", "small": "claude-haiku-4-5", "fallback": "gpt-5"},
    "xai": {"label": "xAI Grok", "env": "XAI_API_KEY", "small": "grok-4-fast", "fallback": "gemini-2.5-pro"},
    "google": {"label": "Google Gemini", "env": "GEMINI_API_KEY", "small": "gemini-2.5-flash", "fallback": "claude-sonnet-5"},
}

MODEL_BY_ID = {m["id"]: m for m in MODELS}

# Simulator profile per model: time-to-first-token (ms), output tokens/sec, and
# the probability of answering an eval item correctly at each difficulty.
SIM_PROFILE = {
    "claude-fable-5-1": {"ttft": 1500, "tps": 60, "skill": {1: 0.995, 2: 0.98, 3: 0.96}},
    "gpt-5": {"ttft": 900, "tps": 90, "skill": {1: 0.99, 2: 0.96, 3: 0.92}},
    "gpt-5-mini": {"ttft": 450, "tps": 150, "skill": {1: 0.97, 2: 0.88, 3: 0.74}},
    "claude-opus-5-5": {"ttft": 800, "tps": 85, "skill": {1: 0.99, 2: 0.97, 3: 0.94}},
    "claude-sonnet-5": {"ttft": 550, "tps": 110, "skill": {1: 0.99, 2: 0.94, 3: 0.88}},
    "claude-haiku-4-5": {"ttft": 350, "tps": 170, "skill": {1: 0.97, 2: 0.87, 3: 0.72}},
    "grok-4": {"ttft": 1100, "tps": 70, "skill": {1: 0.99, 2: 0.95, 3: 0.90}},
    "grok-4-fast": {"ttft": 400, "tps": 180, "skill": {1: 0.96, 2: 0.86, 3: 0.70}},
    "gemini-2.5-pro": {"ttft": 1000, "tps": 95, "skill": {1: 0.99, 2: 0.95, 3: 0.91}},
    "gemini-2.5-flash": {"ttft": 380, "tps": 200, "skill": {1: 0.97, 2: 0.87, 3: 0.73}},
}

# Eval suite: short questions with known answers so accuracy can be graded
# automatically. Two items are near-duplicates on purpose — they show when a
# similarity (semantic) cache saves money and when it returns a WRONG answer.
SUITE = [
    {"id": "q1", "q": "What is 17 * 23?", "accept": ["391"], "difficulty": 1},
    {"id": "q2", "q": "What is the capital city of Australia?", "accept": ["canberra"], "difficulty": 1},
    {"id": "q3", "q": "What is the capital of Australia?", "accept": ["canberra"], "difficulty": 1},
    {"id": "q4", "q": "What is the capital city of Austria?", "accept": ["vienna", "wien"], "difficulty": 1},
    {"id": "q5", "q": "How many times does the letter r appear in the word strawberry?", "accept": ["3", "three"], "difficulty": 2},
    {"id": "q6", "q": "Which number is larger: 9.11 or 9.9?", "accept": ["9.9"], "difficulty": 2},
    {"id": "q7", "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost?", "accept": ["5", "five", "0.05"], "difficulty": 3},
    {"id": "q8", "q": "A train departs at 15:40 and the trip takes 2 hours 35 minutes. At what time (24h HH:MM) does it arrive?", "accept": ["18:15"], "difficulty": 3},
    {"id": "q9", "q": "Which HTTP status code means Too Many Requests?", "accept": ["429"], "difficulty": 1},
    {"id": "q10", "q": "A customer email reads: 'Hi, I'm John (john.doe@acme.com, card 4111 1111 1111 1111). What is 12 + 30?' Reply with just the number.", "accept": ["42"], "difficulty": 1},
]

# Jev (TypeSafe's System One model) returns typed decisions, not text, so for the
# eval suite it picks from candidate answers via a `choice` question.
JEV = {
    "label": "Jev (TypeSafe System One)", "model": "jev-latest", "env": "TYPESAFE_API_KEY",
    "in": 0.042, "out": 0.0,  # $/1M tokens as reported at launch; output tokens are free
    "skill": {1: 0.95, 2: 0.78, 3: 0.55},  # simulator: fast "System 1" judgment is weakest on multi-step items
}
OPTIONS = {
    "q1": ["391", "381", "401", "361"], "q2": ["Canberra", "Sydney", "Melbourne", "Perth"],
    "q3": ["Canberra", "Sydney", "Melbourne", "Perth"], "q4": ["Vienna", "Salzburg", "Graz", "Canberra"],
    "q5": ["3", "2", "1", "4"], "q6": ["9.9", "9.11", "They are equal"], "q7": ["5", "10", "1", "15"],
    "q8": ["18:15", "18:05", "17:75", "18:25"], "q9": ["429", "503", "403", "408"], "q10": ["42", "43", "41", "32"],
}

# Plausible wrong answers the simulator returns when a model "misses".
SIM_WRONG = {
    "q1": "381", "q2": "Sydney", "q3": "Sydney", "q4": "Salzburg", "q5": "2", "q6": "9.11",
    "q7": "10", "q8": "18:05", "q9": "503", "q10": "43",
}
SIM_RIGHT = {
    "q1": "391", "q2": "Canberra", "q3": "Canberra", "q4": "Vienna", "q5": "3", "q6": "9.9",
    "q7": "5", "q8": "18:15", "q9": "429", "q10": "42",
}

# Each path is an ordered chain of hops the request walks through before the LLM.
PATHS = [
    {
        "id": "direct", "name": "Direct", "short": "Agent → LLM",
        "chain": ["Agent", "LLM"],
        "blurb": "Each app calls the provider SDK itself. Fastest and simplest; every team re-builds keys, retries, logging and safety on its own.",
    },
    {
        "id": "direct_small", "name": "Direct · small model", "short": "Agent → provider's small LLM",
        "chain": ["Agent", "Small LLM"],
        "blurb": "Same as Direct, but always on the provider's small, cheap model. Shows whether your task actually needs the big one.",
    },
    {
        "id": "redis", "name": "Redis cache", "short": "Agent → Redis → LLM",
        "chain": ["Agent", "Redis", "LLM"],
        "blurb": "Exact-match response cache in front of the model. Repeat questions return in ~1 ms for $0; anything worded differently still hits the LLM.",
    },
    {
        "id": "jev", "name": "Jev only", "short": "Agent → Jev",
        "chain": ["Agent", "Jev"],
        "blurb": "TypeSafe's Jev answers as a typed decision with calibrated probabilities instead of text. Here it picks from candidate answers. Very fast and cheap, but it can't write prose.",
    },
    {
        "id": "jev_llm", "name": "Jev → LLM cascade", "short": "Agent → Jev → (unsure) LLM",
        "chain": ["Agent", "Jev", "LLM if unsure"],
        "blurb": "Jev answers when its confidence clears the threshold; otherwise the request escalates to the LLM (System 1 → System 2). With your own prompt, Jev picks the small or large model instead.",
    },
    {
        "id": "gateway", "name": "AI Gateway", "short": "Agent → AI Gateway → LLM",
        "chain": ["Agent", "AI Gateway", "LLM"],
        "blurb": "Central gateway owns provider keys, quotas, guardrails, cost metering and provider failover for every app.",
    },
    {
        "id": "gateway_jev", "name": "AI Gateway → Jev → LLM", "short": "Gateway, then Jev, then LLM if unsure",
        "chain": ["Agent", "AI Gateway", "Jev", "LLM if unsure"],
        "blurb": "Everything the AI Gateway provides, plus Jev answering fixed-answer decisions when it's confident. Only unsure or free-text requests reach the LLM.",
    },
    {
        "id": "platform", "name": "Agent Runtime → AI Gateway → LLM", "short": "Enablement platform",
        "chain": ["Agent Runtime", "AI Gateway", "LLM"],
        "blurb": "The enablement platform. A managed agent runtime (memory, policy, tracing) sends every model call through the AI Gateway. The gateway also caches exact and similar questions, and uses Jev to decide per request whether the provider's small model is enough.",
    },
]

# What each path gives a platform team out of the box.
CAPABILITIES = [
    ("keys", "Central API-key custody"),
    ("quota", "Per-app rate limits & quotas"),
    ("pii", "PII redaction"),
    ("inject", "Prompt-injection guard"),
    ("audit", "Audit log & tracing"),
    ("chargeback", "Cost attribution per app"),
    ("cache", "Response caching"),
    ("failover", "Provider failover"),
    ("routing", "Cost-aware model routing"),
    ("memory", "Agent memory & tool policy"),
    ("calibrated", "Calibrated confidence (probabilities)"),
]
PATH_CAPS = {
    "direct": [],
    "direct_small": [],
    "redis": ["cache"],
    "jev": ["calibrated"],
    "jev_llm": ["routing", "calibrated"],
    "gateway": ["keys", "quota", "pii", "inject", "audit", "chargeback", "failover"],
    "gateway_jev": ["keys", "quota", "pii", "inject", "audit", "chargeback", "failover", "routing", "calibrated"],
    "platform": ["keys", "quota", "pii", "inject", "audit", "chargeback", "cache", "failover", "routing", "memory"],
}

# Default knobs, all overridable from the UI "Assumptions" panel.
DEFAULT_ASSUMPTIONS = {
    "jev_threshold": 0.8,        # cascade: Jev's confidence needed to answer without the LLM
    "jev_price_in": JEV["in"],   # $/1M input tokens for Jev
    "hop_rtt_ms": 2.0,           # extra network RTT added per service hop (same-region)
    "semantic_threshold": 0.8,   # token-overlap similarity needed for a semantic-cache hit
    "requests_per_day": 100000,  # for monthly cost projection
    "infra_per_1k": {            # USD per 1K requests for the middleware itself
        "direct": 0.0, "direct_small": 0.0, "redis": 0.002, "jev": 0.0, "jev_llm": 0.0, "gateway": 0.006, "gateway_jev": 0.006, "platform": 0.015,
    },
    "runtime_context_tokens": 180,  # memory + policy preamble the runtime injects
}


UNKNOWN_ANSWERS = ["unknown", "not specified", "not mentioned", "not covered", "does not say", "doesn't say",
                   "don't know", "do not know", "cannot be determined", "no information"]


def suite_items():
    """The built-in quiz as normalized test cases."""
    return [{**s, "options": OPTIONS[s["id"]], "sim_right": SIM_RIGHT[s["id"]], "sim_wrong": SIM_WRONG[s["id"]],
             "unanswerable": False} for s in SUITE]
