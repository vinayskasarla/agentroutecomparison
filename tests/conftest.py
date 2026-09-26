"""Tests run offline: no API keys, no cloud credentials, no price sync. Model results are synthetic, so every
test is deterministic and free."""
import os
import sys

for k in list(os.environ):
    if k.startswith(("AWS_", "AZURE_", "GOOGLE_")) or k.endswith("_API_KEY") or k in ("ANTHROPIC_AUTH_TOKEN",):
        del os.environ[k]
os.environ.update(ENV_FILE="/nonexistent/.env", PRICE_SYNC="off", AWS_EC2_METADATA_DISABLED="true", AWS_CONFIG_FILE="/dev/null",
                  AWS_SHARED_CREDENTIALS_FILE="/dev/null", AUDIT_LOG="off")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import advisor  # noqa: E402
import catalog  # noqa: E402


def make_spec(goal="Route every support ticket to billing, technical, account or sales. 50k a day", **over):
    return advisor.finalize_spec({**advisor.spec_from_rules(goal), **over})


def make_rows(n=12, right=12, conf=90, in_tok=110, out_tok=20, ms=600, error=None, simulated=False, halluc=0):
    rows = []
    for i in range(n):
        ok = i < right
        rows.append({"answer": "x", "confidence": conf, "correct": ok, "hallucinated": (not ok and i - right < halluc),
                     "in_tokens": in_tok, "out_tokens": out_tok, "latency_ms": ms + i, "error": error, "simulated": simulated})
    return rows


def make_res(spec_models, n=12):
    """{model_id: rows} where spec_models maps model -> number right (or (right, kwargs))."""
    out = {}
    for m, v in spec_models.items():
        right, kw = (v, {}) if isinstance(v, int) else v
        out[m] = make_rows(n=n, right=right, **kw)
    return out


@pytest.fixture
def spec():
    return make_spec()


@pytest.fixture(autouse=True)
def _clean_price_mult():
    advisor.PRICE_MULT.clear()
    yield
    advisor.PRICE_MULT.clear()


ALL_APPROVED = [m["id"] for m in catalog.CALLABLE if __import__("policy").model_allowed(m["id"])]
