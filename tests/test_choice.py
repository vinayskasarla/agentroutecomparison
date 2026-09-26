"""Picking the best model and the fallback from a priced table (advisor.choose_from_table)."""
import advisor


def row(id, platform, monthly, meets=True, mode="live", status="verified", maker=None, acc=0.95):
    return {"id": id, "label": id, "platform": platform, "maker": maker or platform, "monthly": monthly, "meets": meets, "mode": mode,
            "status": status, "tested": True, "accuracy": acc, "halluc": 0.0}


def test_cheapest_qualifying_verified_model_wins():
    t = [row("a", "P1", 10, status="needs_review"), row("b", "P1", 20), row("c", "P2", 30)]
    primary, fallback, good = advisor.choose_from_table(t, None)
    assert primary == "b" and fallback == "c"


def test_live_measurements_beat_simulated_ones():
    t = [row("sim", "P1", 5, mode="simulated"), row("live", "P2", 50)]
    assert advisor.choose_from_table(t, None)[0] == "live"


def test_when_nothing_qualifies_closest_comes_from_live_rows():
    """Regression: a simulated model used to win over live-measured ones when no model met every requirement."""
    t = [row("sim", "P1", 5, meets=False, mode="simulated", acc=0.99), row("live1", "P2", 50, meets=False, acc=0.88),
         row("live2", "P2", 60, meets=False, acc=0.80)]
    assert advisor.choose_from_table(t, None)[0] == "live1"


def test_fallback_is_on_another_platform_even_if_only_simulated_qualifies():
    """Regression: with only one platform measured live, the fallback used to be 'none'."""
    t = [row("live", "Anthropic API", 10), row("sim", "Amazon Bedrock", 12, mode="simulated")]
    primary, fallback, _ = advisor.choose_from_table(t, None)
    assert primary == "live" and fallback == "sim"


def test_spend_commitment_wins_within_15_percent():
    t = [row("a", "OpenAI API", 100), row("b", "Amazon Bedrock", 110), row("c", "Azure AI Foundry", 130)]
    assert advisor.choose_from_table(t, None, {"commitment": "Amazon Bedrock"})[0] == "b"
    assert advisor.choose_from_table(t, None, {"commitment": "Azure AI Foundry"})[0] == "a"  # 30% dearer: no


def test_cheaper_unverified_is_reported_not_picked():
    t = [row("u", "P1", 5, status="needs_review"), row("v", "P2", 20)]
    assert advisor.choose_from_table(t, None)[0] == "v"
    assert advisor.cheaper_unverified(t, "v")["id"] == "u"


def test_fallback_named_even_when_nothing_on_another_platform_qualifies():
    """Regression (live run, 2026-09-26): no fallback was shown when the only other-platform model missed a limit."""
    t = [row("a", "Anthropic API", 10), row("b", "Anthropic API", 20),
         row("x", "Amazon Bedrock", 5, meets=False, mode="simulated", acc=0.93)]
    primary, fallback, _ = advisor.choose_from_table(t, None)
    assert primary == "a" and fallback == "x"
