"""The web API end to end, offline (simulated model calls): sessions, stability, comparison limits, policy."""
import json

import pytest
from fastapi.testclient import TestClient

import app as appmod

GOAL = "Route every support ticket to billing, technical, account or sales. 50k a day"


def advise(client, **body):
    r = client.post("/api/advise", json={"goal": GOAL, "force_sim": True, **body})
    if r.status_code != 200:
        return r.status_code, r.json()
    lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    errors = [x for x in lines if x["type"] == "error"]
    assert not errors, errors
    return 200, next(x for x in lines if x["type"] == "advice")


@pytest.fixture
def client():
    appmod._advice_cache.clear(); appmod._spec_pins.clear(); appmod._measured.clear(); appmod._compare_log.clear()
    with TestClient(appmod.app) as c:
        c.get("/")  # sets the browser-session cookie
        yield c


def test_pages_and_config(client):
    for p in ("/", "/ask", "/benchmark", "/news"):
        assert client.get(p).status_code == 200
    cfg = client.get("/api/config").json()
    assert len(cfg["policy"]["approved_platforms"]) == 6


def test_first_run_is_a_shortlist_with_costs(client):
    _, a = advise(client)
    assert a["scope"] == "shortlist" and len(a["evaluated_models"]) == 4 and a["eligible_models"] > 4
    assert a["run_cost"]["compare_estimate"]["cost"] > 0 and a["system"] and a["robustness"]


def test_same_session_same_goal_gives_the_same_answer(client):
    _, a1 = advise(client)
    _, a2 = advise(client, harness=True)  # a different option re-ranks the same evidence
    assert a2["stability"]["reused"] == a2["stability"]["measured"] > 0
    assert a1["models"]["primary"] == a2["models"]["primary"]
    _, a3 = advise(client)  # identical request: served from the session cache
    assert a3.get("cached_at") and a3["models"] == a1["models"]


def test_new_browser_session_runs_fresh():
    appmod._advice_cache.clear(); appmod._spec_pins.clear(); appmod._measured.clear()
    with TestClient(appmod.app) as c1, TestClient(appmod.app) as c2:
        c1.get("/"); c2.get("/")
        _, a = advise(c1)
        _, b = advise(c2)
        assert not b.get("cached_at") and b["stability"]["reused"] == 0 and a["models"]["primary"] == b["models"]["primary"]


def test_compare_reuses_shortlist_measurements_and_is_limited(client, monkeypatch):
    monkeypatch.setattr(appmod, "COMPARE_LIMIT", 2)
    _, first = advise(client)
    spec = client.post("/api/advise", json={"goal": GOAL, "force_sim": True}).text  # cached; fetch spec event
    spec = next(json.loads(x) for x in spec.splitlines() if '"type": "spec"' in x)["spec"]
    _, full = advise(client, spec=spec, compare=True)
    assert full["scope"] == "full" and len(full["evaluated_models"]) == full["eligible_models"]
    assert full["stability"]["reused"] >= 4 * 1  # the shortlisted models weren't re-called
    assert advise(client, goal=GOAL + " v2", compare=True, spec={**spec, "summary": "x"})[0] == 200
    code, body = advise(client, goal=GOAL + " v3", compare=True, spec={**spec, "summary": "y"})
    assert code == 429 and body["quota"]["left"] == 0
    assert advise(client, goal=GOAL + " v4")[0] == 200  # first runs are never limited


def test_policy_blocks_unapproved_model(client):
    r = client.post("/api/run", json={"model": "vertex-gemini-2.5-flash", "force_sim": True})
    assert r.status_code == 403


def test_regulated_data_only_goes_to_approved_platforms(client):
    _, a = advise(client, constraints={"data_class": "regulated"})
    assert all(m.startswith(("bedrock-", "azure-")) for m in a["evaluated_models"])
    assert any("regulated" in e["reason"] for e in a["excluded_models"])


def test_password_gate(monkeypatch, client):
    monkeypatch.setattr(appmod, "APP_PASSWORD", "s3cret")
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("anyone", "wrong")).status_code == 401
    assert client.get("/", auth=("anyone", "s3cret")).status_code == 200


TEST_SET = {"name": "ticket routing v1", "summary": "Route support tickets to the right team.",
            "labels": ["billing", "technical", "account", "sales"],
            "test_cases": [{"input": f"Ticket {i}: I was charged twice", "expected": "billing"} for i in range(10)]}


def test_saved_test_set_is_used_exactly(client):
    r = client.post("/api/advise", json={"goal": GOAL, "force_sim": True, "test_set": TEST_SET})
    lines = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    spec = next(x for x in lines if x["type"] == "spec")["spec"]
    adv = next(x for x in lines if x["type"] == "advice")
    assert spec["test_set"] == {"name": "ticket routing v1", "count": 10} and spec["summary"] == TEST_SET["summary"]
    assert [c["expected"] for c in adv["cases"]] == ["billing"] * 10


def test_bad_test_set_is_refused(client):
    code, body = advise(client, test_set={"test_cases": []})
    assert code == 400 and "test set" in body["error"]


def test_priorities_are_part_of_the_cached_result(client):
    _, a = advise(client)
    _, b = advise(client, priorities=["accuracy"])
    assert a["key"] != b["key"]
