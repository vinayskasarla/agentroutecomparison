import asyncio

import grading


def test_normalize_handles_formats():
    assert grading.normalize("**$12,000.00** refund") == "12000 refund"
    assert grading.normalize("It’s — done") == "it's - done"


def test_keyword_grade_is_normalized():
    assert grading.grade("Order #A-1234 was found", ["a-1234"])
    assert grading.grade("The total is $1,250", ["1250"])
    assert not grading.grade("order A-12345", ["A-1234"])  # no partial-token matches


def test_label_uses_first_label_named():
    item = {"options": ["billing", "technical", "account", "sales"], "accept": ["billing"]}
    assert grading.grade_item("Billing — not a technical issue", item)[0:2] == (True, "exact")
    assert grading.grade_item("Technical, maybe billing", item)[0] is False
    assert grading.grade_item("I can't tell", item)[0] is False


def test_declines_recognised_in_many_phrasings():
    item = {"unanswerable": True, "accept": ["unknown"]}
    for a in ["Unknown.", "The documents don't mention this.", "I cannot determine that from the policy",
              "That is not covered in the reference material", "There is insufficient information"]:
        assert grading.grade_item(a, item)[0], a
    assert not grading.grade_item("Returns are accepted within 60 days.", item)[0]


def test_paraphrase_goes_to_judge_only_for_phrases():
    phrase = {"accept": ["full refund within 30 days"]}
    assert grading.grade_item("You get all your money back if you return it in a month", phrase) == (False, "keyword", True)
    short = {"accept": ["A-1234"]}
    assert grading.grade_item("order B-9", short) == (False, "keyword", False)


def test_judge_not_allowed_without_key_or_for_blocked_data(monkeypatch):
    from conftest import make_spec
    assert not grading.judge_allowed(make_spec())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert grading.judge_allowed(make_spec(data_class="internal"))
    assert not grading.judge_allowed(make_spec(data_class="regulated"))  # Anthropic API isn't approved for regulated data


def test_judge_is_deterministic_and_cached(monkeypatch):
    calls = []

    class Msgs:
        async def create(self, **kw):
            calls.append(kw)
            Block = type("B", (), {"type": "text", "text": '{"same_meaning": true, "reason": "same"}'})
            Usage = type("U", (), {"input_tokens": 200, "output_tokens": 10})
            return type("R", (), {"content": [Block()], "usage": Usage()})()

    monkeypatch.setattr(grading.anthropic, "AsyncAnthropic", lambda: type("C", (), {"messages": Msgs()})())
    grading._verdicts.clear()
    v1, u1 = asyncio.run(grading.judge("q", ["full refund"], "all your money back"))
    v2, u2 = asyncio.run(grading.judge("q", ["full refund"], "All your money back!"))
    assert v1 is True and v2 is True and u1 and u2 is None  # second answer normalises to the same -> cached
    assert len(calls) == 1 and calls[0]["temperature"] == 0 and calls[0]["model"] == grading.JUDGE_MODEL
