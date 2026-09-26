"""Token pricing, prompt caching, harness, total cost of ownership and the per-run cost breakdown."""
import advisor
import catalog
import constraints
from conftest import make_res, make_spec


def test_cached_tokens_are_discounted():
    price = catalog.MODEL_BY_ID["claude-haiku-4-5"]
    full = advisor._cost({"in_tokens": 10000, "out_tokens": 0}, price)
    cached = advisor._cost({"in_tokens": 10000, "out_tokens": 0, "cached_tokens": 9000, "cache_mult": 0.1}, price)
    assert abs(full - 0.01) < 1e-9 and abs(cached - (1000 + 900) / 1e6) < 1e-9


def test_whole_documents_cache_but_retrieved_text_does_not():
    spec = make_spec(task_type="grounded_qa", needs_documents=True, document_size="small")
    res = make_res({"claude-haiku-4-5": 12})
    lc = constraints.inflate(res, spec, "long_context", ["claude-haiku-4-5"])["claude-haiku-4-5"][0]
    rag = constraints.inflate(res, spec, "rag", ["claude-haiku-4-5"])["claude-haiku-4-5"][0]
    assert lc["cached_tokens"] >= spec["document_tokens"] and rag["cached_tokens"] < spec["context_tokens"]


def test_no_caching_where_platform_has_none():
    spec = make_spec(system_prompt_tokens=4000)
    r = constraints.inflate(make_res({"bedrock-gpt-oss-20b": 12}), spec, "single_call", ["bedrock-gpt-oss-20b"])
    assert r["bedrock-gpt-oss-20b"][0]["cached_tokens"] == 0


def test_harness_adds_cost_and_the_right_controls():
    base = make_spec(task_type="tool_actions", needs_tools=True, integration="mcp", tool_count=6)
    res = make_res({"claude-haiku-4-5": 12, "bedrock-gpt-oss-20b": 12})
    models = {"primary": "claude-haiku-4-5", "small": "bedrock-gpt-oss-20b", "fallback": None}
    off = advisor.build_design("tool_agent", models, "x", res, None, {**base, "harness": False})
    on = advisor.build_design("tool_agent", models, "x", res, None, {**base, "harness": True})
    ids = {c["id"] for c in on["harness"]["controls"]}
    assert on["monthly"] > off["monthly"] and {"tool_gate", "limits", "mcp_governance", "input_guard"} <= ids


def test_agent_turns_scale_cost_with_tool_calls():
    res = make_res({"claude-haiku-4-5": 12})
    models = {"primary": "claude-haiku-4-5", "small": "claude-haiku-4-5", "fallback": None}
    few = advisor.build_design("tool_agent", models, "x", res, None, make_spec(task_type="tool_actions", needs_tools=True, tool_calls_per_request=1))
    many = advisor.build_design("tool_agent", models, "x", res, None, make_spec(task_type="tool_actions", needs_tools=True, tool_calls_per_request=6))
    assert many["cost_per_req"] > 2 * few["cost_per_req"] and many["p95_ms"] > few["p95_ms"]


def test_infra_and_reuse():
    spec = make_spec(task_type="grounded_qa", needs_documents=True, document_size="large")
    assert constraints.tco("rag", spec, False)["infra_monthly"] > 0
    assert constraints.tco("rag", {**spec, "existing": ["vector_store"]}, False)["infra_monthly"] == 0
    mcp = make_spec(task_type="tool_actions", needs_tools=True, integration="mcp", tool_count=10)
    assert any(i["id"] == "mcp_servers" for i in constraints.tco("tool_agent", mcp, False)["infra"])


def test_run_cost_counts_only_new_live_calls():
    res = make_res({"claude-haiku-4-5": 12, "bedrock-nova-micro": 12})
    res["bedrock-nova-micro"] = [{**r, "reused": True} for r in res["bedrock-nova-micro"]]
    c = advisor.run_cost({"model": "claude-sonnet-5", "in": 1000, "out": 2000}, res,
                         {"claude-haiku-4-5": "live", "bedrock-nova-micro": "live"}, "shortlist", [{"in": 300, "out": 10}])
    steps = {s["kind"] + ":" + str(s.get("model")): s for s in c["steps"]}
    assert abs(steps["architect:claude-sonnet-5"]["cost"] - 0.022) < 1e-9
    assert steps["test:bedrock-nova-micro"]["cost"] == 0 and steps["test:bedrock-nova-micro"]["reused"]
    assert steps["judge:claude-haiku-4-5"]["cost"] > 0
    assert abs(c["total"] - sum(s["cost"] for s in c["steps"])) < 1e-12


def test_simulated_calls_cost_nothing_but_are_projected():
    res = make_res({"claude-haiku-4-5": 12})
    c = advisor.run_cost(None, res, {"claude-haiku-4-5": "simulated"}, "shortlist")
    assert c["total"] == 0 and c["if_all_live"] > 0


def test_compare_estimate_matches_what_the_models_would_cost():
    res = make_res({"claude-haiku-4-5": 12})
    est = advisor.compare_estimate(res, ["claude-haiku-4-5", "claude-sonnet-5"])
    per = lambda m: 12 * (110 * catalog.MODEL_BY_ID[m]["in"] + 20 * catalog.MODEL_BY_ID[m]["out"]) / 1e6  # noqa: E731
    assert abs(est["cost"] - per("claude-haiku-4-5") - per("claude-sonnet-5")) < 1e-12
    assert est["largest"] == "Claude Sonnet 5"


def test_wilson_range():
    lo, hi = constraints.wilson(12, 12)
    assert 0.7 < lo < 0.8 and hi == 1.0
    assert constraints.wilson(0, 0) == (None, None)


def test_guard_checks_read_only_what_they_need():
    """Found live: the input guardrail and grounding check were priced as a full copy of the agent's prompt, so
    the harness was 2/3 of a RAG design's cost. The guard reads the request; grounding the passages and answer."""
    spec = make_spec("Answer customer questions from our help center docs. 20k a day.", harness=True)
    res = make_res({"claude-sonnet-5": 12, "claude-haiku-4-5": 12})
    models = {"primary": "claude-sonnet-5", "small": "claude-haiku-4-5", "guard": "claude-haiku-4-5", "fallback": None}
    lean = advisor.build_design("rag", models, "x", res, None, spec)
    heavy = advisor.build_design("rag", models, "x", res, None, {**spec, "system_prompt_tokens": 20000})
    guard = lambda d: next(c for c in d["harness"]["controls"] if c["id"] == "input_guard")["cost_per_req"]  # noqa: E731
    assert guard(lean) == guard(heavy)  # the agent's own system prompt doesn't reach the guard
    haiku = catalog.MODEL_BY_ID["claude-haiku-4-5"]
    grounding = next(c for c in lean["harness"]["controls"] if c["id"] == "grounding")["cost_per_req"]
    assert grounding < advisor._cost({"in_tokens": spec["context_tokens"] + 2000, "out_tokens": 100}, haiku)
    assert lean["harness"]["monthly"] < lean["monthly"] - lean["harness"]["monthly"]  # controls cost less than the model


def test_guard_model_is_the_same_for_the_table_and_the_winner():
    """Found live: the table priced Sonnet 5's controls on a simulated cheap model, the winner card on Haiku 4.5,
    so the same model showed $2,958 in the table and $6,542 on the card."""
    spec = make_spec("Answer customer questions from our help center docs. 20k a day.", harness=True)
    res = make_res({"claude-sonnet-5": 12, "claude-haiku-4-5": 10, "bedrock-gpt-oss-120b": (12, {"simulated": True})})
    mode = {"claude-sonnet-5": "live", "claude-haiku-4-5": "live", "bedrock-gpt-oss-120b": "simulated"}
    r = advisor.rank_designs(spec, res, None, mode)
    first = r["top"][0]
    row = next(x for x in r["model_table"] if x["id"] == first["models"]["primary"])
    assert abs(row["monthly"] - first["monthly"]) < 1e-9


def test_cached_answers_are_not_grounding_checked_again():
    spec = make_spec("Answer customer questions from our help center docs. 20k a day.", harness=True)
    res = make_res({"claude-sonnet-5": 12, "claude-haiku-4-5": 12})
    models = {"primary": "claude-sonnet-5", "small": "claude-haiku-4-5", "guard": "claude-haiku-4-5", "fallback": None}
    ctl = lambda d, k: next(c for c in d["harness"]["controls"] if c["id"] == k)["cost_per_req"]  # noqa: E731
    none = advisor.build_design("rag", models, "x", res, None, {**spec, "repeat_rate": 0.0})
    half = advisor.build_design("rag", models, "x", res, None, {**spec, "repeat_rate": 0.5})
    if any(a["id"] == "cache" for a in half["addons"]):
        assert abs(ctl(half, "grounding") - 0.5 * ctl(none, "grounding")) < 1e-12
        assert ctl(half, "input_guard") == ctl(none, "input_guard")  # every request is still screened
