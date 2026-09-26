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


def test_cheaper_option_is_shown_with_what_it_gives_up():
    """A VP asked why the $10.9k design won over a $1.7k one: the page must name the cheaper model and why not."""
    spec = make_spec("Answer customer questions from our help center docs. 20k a day.", accuracy_target=0.9)
    res = make_res({"claude-sonnet-5": 12, "claude-haiku-4-5": 12, "bedrock-gpt-oss-120b": 9})
    mode = {"claude-sonnet-5": "live", "claude-haiku-4-5": "live", "bedrock-gpt-oss-120b": "simulated"}
    r = advisor.rank_designs(spec, res, None, mode)
    c = r["models"]["cheaper_option"]
    assert c and c["monthly"] < r["top"][0]["monthly"] and c["saves"] > 0
    if not c["meets"]:
        assert "accuracy" in c["failed"] and any("Why not the cheaper" in w for w in r["top"][0]["why"])


PRIO_MODELS = {"claude-haiku-4-5": 11, "claude-sonnet-5": 12, "claude-opus-5-5": 12, "bedrock-nova-micro": 10, "gpt-5-mini": 12}


def _prio_rank(priorities, **over):
    spec = make_spec(accuracy_target=0.9, priorities=priorities, **over)
    return advisor.rank_designs(spec, make_res(PRIO_MODELS), None, {m: "live" for m in PRIO_MODELS})


def test_default_and_cost_first_pick_the_cheapest_qualifying_model():
    r = _prio_rank([])
    qualifying = [x for x in r["model_table"] if x["tested"] and x["meets"]]
    assert r["models"]["primary"] == qualifying[0]["id"]  # table is cheapest first


def test_accuracy_first_picks_the_most_accurate_qualifying_model():
    r = _prio_rank(["accuracy"])
    best = max(x["accuracy"] for x in r["model_table"] if x["tested"] and x["meets"])
    row = next(x for x in r["model_table"] if x["id"] == r["models"]["primary"])
    assert row["accuracy"] == best == 1.0
    assert any("highest accuracy first" in w for w in r["top"][0]["why"])


def test_cost_first_accepts_a_small_accuracy_shortfall_but_never_hallucinations():
    spec = make_spec(accuracy_target=0.95, priorities=["cost"])
    assert advisor.accuracy_floor(spec) == 0.9 and advisor.accuracy_floor({**spec, "priorities": ["cost", "accuracy"]}) == 0.95
    # 11/12 = 92%: fails a 95% target, passes it with cost first
    d = {"accuracy": 11 / 12, "halluc": 0.0, "acc_lo": 0.6, "acc_hi": 0.99, "p95_ms": 100, "total_monthly": 1,
         "models": {"primary": "claude-haiku-4-5"}, "context_needed": 1000, "peak": {"quota": None}}
    acc = lambda s: next(c for c in advisor.checks(d, s) if c["key"] == "accuracy")  # noqa: E731
    assert not acc({**spec, "priorities": []})["pass"] and acc(spec)["pass"] and acc(spec)["relaxed_from"] == 0.95
    bad = {**d, "halluc": 0.2}
    assert not next(c for c in advisor.checks(bad, spec) if c["key"] == "halluc")["pass"]


def test_priorities_change_the_answer_only_through_the_spec():
    a, b = _prio_rank(["accuracy"]), _prio_rank(["accuracy"])
    assert summary(a) == summary(b)
    assert advisor.finalize_spec({**make_spec(), "priorities": ["speed", "bogus", "cost"]})["priorities"] == ["cost", "speed"]


def test_p90_is_between_median_and_p95():
    d = _prio_rank([])["top"][0]
    assert d["p50_ms"] <= d["p90_ms"] <= d["p95_ms"]


def test_cost_first_architectures_keep_hallucinations_first():
    spec = make_spec("Answer customer questions from our help center docs. 20k a day.", priorities=["cost"], accuracy_target=0.9)
    res = make_res({"claude-sonnet-5": (11, {"halluc": 1}), "claude-haiku-4-5": (9, {"halluc": 3})})
    r = advisor.rank_designs(spec, res, None, {m: "live" for m in res})
    failing = [d for d in r["top"] if not d["passes"]]
    rates = [d["halluc"] for d in failing]
    assert rates == sorted(rates)
