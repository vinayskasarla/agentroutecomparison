"""Grading a model's answer against a test case's expected answer.

Three methods, cheapest first, and every grade records which one decided it:
  exact   - the answer must pick one of the allowed labels; the first label it names is its choice
  keyword - normalised match of an expected phrase (case, punctuation, markdown, number formats ignored)
  judge   - for free-text answers the keyword check can't confirm, a small Claude model decides whether the
            answer says the same thing as the expected one (temperature 0, fixed rubric, cached per answer)
Declining ("unknown", "not in the documents", ...) is recognised in many phrasings.
"""
import hashlib
import json
import os
import re
import unicodedata

import anthropic

import catalog
import policy

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")
JUDGE_ENABLED = os.environ.get("JUDGE_ENABLED", "true").lower() != "false"
JUDGE_PRICE = (1.0, 5.0)  # $ per 1M tokens (in, out) for claude-haiku-4-5

DECLINE = re.compile(
    r"\b(unknown|i (?:do not|don't) know|not (?:in|covered|mentioned|stated|specified|provided|available|included)"
    r"|no (?:information|mention|data|details)|(?:cannot|can't|unable to) (?:determine|find|answer|say|tell)"
    r"|insufficient (?:information|data|context)|(?:does not|do not|doesn't|don't) (?:say|mention|specify|cover|include|contain|state)"
    r"|not enough (?:information|context))\b")


def normalize(text):
    """Lowercase, strip markdown and punctuation, unify quotes/dashes/number formats and whitespace."""
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    t = re.sub(r"[*_`#>]+", " ", t)                       # markdown
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)             # 12,000 -> 12000
    t = re.sub(r"[$€£](?=\d)", "", t)                     # $49 -> 49
    t = re.sub(r"(?<=\d)\.0+\b", "", t)                   # 49.00 -> 49
    t = re.sub(r"[^\w\s'.%/-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _contains(haystack, needle):
    n = normalize(needle)
    return bool(n) and re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", haystack) is not None


def is_decline(answer):
    return bool(DECLINE.search(normalize(answer)))


def first_label(answer, options):
    """The label the answer commits to: the one it names first."""
    a, best = normalize(answer), None
    for o in options:
        m = re.search(r"(?<![\w])" + re.escape(normalize(o)) + r"(?![\w])", a)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), o)
    return best[1] if best else None


def grade(answer, accept):
    """Backwards-compatible keyword grade: True if any accepted phrase appears (normalised)."""
    a = normalize(answer)
    return any(_contains(a, x) for x in accept)


def grade_item(answer, item):
    """(correct, method, needs_judge) for one graded test case."""
    accept = item.get("accept") or []
    if item.get("unanswerable"):
        return is_decline(answer), "keyword", False
    if item.get("options"):
        choice = first_label(answer, item["options"])
        return (choice is not None and any(normalize(choice) == normalize(x) for x in accept)), "exact", False
    if grade(answer, accept):
        return True, "keyword", False
    # Free text the keyword check can't confirm: a paraphrase may still be right. Short expected answers
    # (an ID, a number, one word) are left to the keyword check; phrases go to the judge.
    long_expected = any(len(normalize(x).split()) >= 2 for x in accept)
    return False, "keyword", long_expected


# ------------------------------------------------------------------ judge
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["same_meaning", "reason"],
                "properties": {"same_meaning": {"type": "boolean"}, "reason": {"type": "string"}}}
JUDGE_PROMPT = """You grade a test answer. Decide if the ANSWER states the same fact or decision as the EXPECTED answer.
Paraphrases, synonyms, extra correct detail and different formatting count as the same. A different fact, a
contradiction, hedging between options, or a refusal does not.

QUESTION:
{q}

EXPECTED (any one is enough; | separates alternatives):
{e}

ANSWER:
{a}"""
_verdicts: dict = {}


def judge_allowed(spec):
    """The judge sends test cases and answers to Anthropic, so it must be allowed for this data class and region."""
    return (JUDGE_ENABLED and bool(os.environ.get("ANTHROPIC_API_KEY"))
            and policy.check_data(JUDGE_MODEL if JUDGE_MODEL in catalog.MODEL_BY_ID else "claude-haiku-4-5",
                                  spec["data_class"], spec["residency"])[0])


async def judge(question, expected, answer):
    """(same_meaning, usage) — cached per (question, expected, answer), so the same answer is judged once."""
    key = hashlib.sha256(json.dumps([question, expected, normalize(answer)]).encode()).hexdigest()
    if key in _verdicts:
        return _verdicts[key], None
    client = anthropic.AsyncAnthropic()
    r = await client.messages.create(
        model=JUDGE_MODEL, max_tokens=300, temperature=0,
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=question[:4000], e=" | ".join(expected), a=(answer or "")[:4000])}])
    verdict = bool(json.loads(next(b.text for b in r.content if b.type == "text"))["same_meaning"])
    if len(_verdicts) > 50000:
        _verdicts.clear()
    _verdicts[key] = verdict
    return verdict, {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
