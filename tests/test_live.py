"""Live API checks. Skipped unless LIVE_ANTHROPIC_API_KEY is set (e.g. a CI secret), so the normal suite stays offline and free."""
import asyncio
import os

import pytest

import grading

KEY = os.environ.get("LIVE_ANTHROPIC_API_KEY")
pytestmark = pytest.mark.skipif(not KEY, reason="set LIVE_ANTHROPIC_API_KEY to run live API checks")


def test_judge_call_is_accepted_and_sensible(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    grading._verdicts.clear()
    same, _ = asyncio.run(grading.judge("What is the refund window?", ["full refund within 30 days"],
                                        "You can return it within a month and get all your money back."))
    diff, _ = asyncio.run(grading.judge("What is the refund window?", ["full refund within 30 days"], "Store credit only, within 14 days."))
    assert same is True and diff is False
