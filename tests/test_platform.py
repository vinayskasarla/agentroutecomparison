"""Supporting pieces: PII redaction, SSO identity, audit events, news parsing, the benchmark route and small endpoints."""
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app as appmod
import audit
import news
import pii


def test_pii_redacted():
    out = pii.redact("Email jane.doe@acme.com or call +1 415 555 0100, card 4111 1111 1111 1111")
    assert "jane.doe@acme.com" not in out and "4111 1111 1111 1111" not in out


def test_identity_from_sso_headers():
    claims = base64.urlsafe_b64encode(json.dumps({"sub": "u1", "email": "a@b.com", "name": "Ann"}).encode()).decode().rstrip("=")
    assert audit.identity({"x-amzn-oidc-data": f"h.{claims}.s"})["email"] == "a@b.com"
    assert audit.identity({"x-forwarded-email": "c@d.com"})["id"] == "c@d.com"
    assert audit.identity({})["id"] == "anonymous"


def test_news_classification_and_catalog_check():
    cat, models = news.classify({"title": "Introducing Claude Haiku 4.5", "summary": "Our fastest model"})
    assert cat == "new_model" and any(m["in_catalog"] for m in models)
    assert news.classify({"title": "Claude Sonnet 3.7 will be retired on March 1"})[0] == "retirement"
    rss = """<rss><channel><item><title>New pricing for GPT-5 mini</title><link>https://x/1</link>
             <pubDate>Mon, 21 Sep 2026 10:00:00 GMT</pubDate><description>Cheaper</description></item></channel></rss>"""
    items = news.parse_rss(rss, {"name": "t", "provider": "OpenAI", "url": "https://x"})
    assert items and items[0]["title"].startswith("New pricing")


@pytest.fixture
def client(monkeypatch):
    # The benchmark route calls the gateway/runtime services over HTTP; route those hops to the app in-process.
    monkeypatch.setattr(appmod, "_http", httpx.AsyncClient(transport=httpx.ASGITransport(app=appmod.app), base_url="http://t"))
    monkeypatch.setattr(appmod, "SELF_URL", "http://t")
    with TestClient(appmod.app) as c:
        c.get("/")
        yield c


def test_benchmark_run_streams_results(client):
    r = client.post("/api/run", json={"model": "claude-haiku-4-5", "force_sim": True, "passes": 1, "workload": "suite",
                                      "paths": ["direct", "gateway"]})
    lines = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    assert r.status_code == 200 and lines[-1]["type"] == "done"
    calls = [x for x in lines if x["type"] == "call"]
    assert {c["path"] for c in calls} == {"direct", "gateway"} and all(c["correct"] is not None for c in calls)


def test_small_endpoints(client):
    assert client.get("/api/me").json()["signed_in"] is False
    assert client.get("/api/policy").json()["approved_models"] > 0
    k = client.get("/api/knowledge").json()
    assert k["models_total"] >= 30 and k["sources"]
    assert client.post("/api/events", json={"event": "adr_downloaded", "data": {"x": 1}}).json()["ok"]
    assert client.post("/api/events", json={"event": "not_allowed", "data": {}}).json()["ok"]
    assert client.get("/api/advice/doesnotexist").status_code == 404
    assert client.get("/api/compare-quota").json()["limit"] == appmod.COMPARE_LIMIT


def test_client_ip_uses_trusted_proxy_entry(monkeypatch):
    req = type("R", (), {"headers": {"x-forwarded-for": "6.6.6.6, 10.0.0.9"}, "client": None})()
    assert appmod.client_ip(req) == "10.0.0.9"  # the forged left-most entry is ignored
