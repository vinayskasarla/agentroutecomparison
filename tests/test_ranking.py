"""End-to-end ranking on synthetic measurements: deterministic, sensible, and consistent with the table."""
import advisor
import catalog
from conftest import make_res, make_spec

MODELS = {"claude-haiku-4-5": 11, "claude-sonnet-5": 12, "bedrock-nova-micro": 10, "bedrock-gpt-oss-20b": 12, "gpt-5-mini": 12}
MODE = {m: "live" for m in MODELS}


def rank(spec, res=None):
    return advisor.rank_designs(spec, res or make_res(MODELS), None, MODE)


def summary(r):
    return [(d["arch"], d["models"]["primary"], d["models"]["fallback"], round(d["monthly"], 6)) for d in r["top"]]


def test_same_inputs_same_answer():
    spec = make_spec()
    assert summary(rank(spec)) == summary(rank(spec)) == summary(rank(dict(spec)))


def test_winner_matches_its_table_row_and_table_is_cheapest_first():
    r = rank(make_spec())
    first = r["top"][0]
    row = next(x for x in r["model_table"] if x["id"] == first["models"]["primary"])
    assert abs(row["monthly"] - first["monthly"]) < 1e-9
    monthly = [x["monthly"] for x in r["model_table"]]
    assert monthly == sorted(monthly)


def test_action_agent_recommendation_can_act():
    spec = make_spec("A support agent that answers from our help-center docs and issues refunds through our payments APIs. 20k a day.")
    r = rank(spec)
    assert r["top"] and all(d["arch"] in ("tool_agent", "workflow", "router", "multi_agent") for d in r["top"])
    assert r["system"]["retrieval"]["strategy"] and r["system"]["integration"]


def test_fallback_on_different_platform():
    first = rank(make_spec())["top"][0]
    fb = first["models"]["fallback"]
    assert fb and catalog.MODEL_BY_ID[fb]["platform"] != catalog.MODEL_BY_ID[first["models"]["primary"]]["platform"]


def test_small_role_only_used_when_cheaper():
    spec = make_spec(request_types=4)
    r = rank(spec)
    for d in r["all"]:
        if d["arch"] in ("router", "workflow"):
            s, p = d["models"]["small"], d["models"]["primary"]
            cost = lambda m: catalog.MODEL_BY_ID[m]["in"] + catalog.MODEL_BY_ID[m]["out"]  # noqa: E731
            assert s == p or cost(s) < cost(p)


def test_untested_models_are_listed_not_compared():
    r = rank(make_spec())
    untested = [x for x in r["model_table"] if not x["tested"]]
    assert untested and all(x["accuracy"] is None and x["monthly"] > 0 for x in untested)


def test_robustness_is_side_effect_free():
    spec = make_spec()
    res = make_res(MODELS)
    base = advisor.rank_designs(spec, res, None, MODE)
    scen = advisor.robustness(spec, res, None, MODE, False, base)
    assert len(scen) >= 6 and not advisor.PRICE_MULT
    assert summary(advisor.rank_designs(spec, res, None, MODE)) == summary(base)


def test_accuracy_range_and_near_misses():
    res = make_res({**MODELS, "bedrock-nova-micro": 11})
    r = advisor.rank_designs(make_spec(accuracy_target=0.95), res, None, MODE)
    first = r["top"][0]
    assert first["acc_lo"] < first["accuracy"] <= first["acc_hi"]


def test_supporting_roles_use_live_models_when_any_exist():
    """Regression (live report): a simulated small model inside a cascade made a live design look 50x cheaper."""
    res = make_res({"claude-haiku-4-5": 11, "claude-sonnet-5": 12, "bedrock-nova-micro": 12, "bedrock-gpt-oss-20b": 12})
    mode = {"claude-haiku-4-5": "live", "claude-sonnet-5": "live", "bedrock-nova-micro": "simulated", "bedrock-gpt-oss-20b": "simulated"}
    spec = make_spec("Answer customer questions from our help center docs. 20k a day", task_type="grounded_qa",
                     needs_documents=True, document_size="large")
    r = advisor.rank_designs(spec, res, None, mode)
    for d in r["all"]:
        for role in ("primary", "small"):
            assert mode[d["models"][role]] == "live", (d["arch"], role, d["models"][role])
