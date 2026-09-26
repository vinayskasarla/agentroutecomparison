"""Which architecture patterns are considered for which goals (advisor.applicable) and how goals are read."""
import advisor
import constraints
from conftest import make_spec

SUPPORT = ("A support agent that answers customers from our help-center docs, looks up orders and issues refunds through "
           "our order and payments APIs (exposed via MCP servers), and hands complex cases to a specialist sub-agent. 20k a day.")


def ok_and_out(**over):
    ok, why = advisor.applicable(make_spec(**over))
    return set(ok), why


def test_action_agent_never_gets_answer_only_designs():
    """Regression: plain RAG used to be recommended for an agent that must issue refunds."""
    ok, why = ok_and_out(needs_tools=True, needs_documents=True, document_size="large", multi_step=True, task_type="tool_actions")
    for a in ("rag", "rag_cascade", "long_context", "single_call", "cascade", "batch", "agentic_rag"):
        assert a not in ok and "action" in why[a]
    assert {"tool_agent", "workflow"} <= ok


def test_multi_document_questions_use_agentic_retrieval():
    ok, why = ok_and_out(task_type="grounded_qa", needs_documents=True, document_size="large", multi_hop=True)
    assert "agentic_rag" in ok and "rag" not in ok


def test_table_data_needs_retrieval():
    ok, _ = ok_and_out(task_type="open_qa", structured_data=True, multi_hop=True)
    assert "single_call" not in ok and "agentic_rag" in ok


def test_varying_steps_rule_out_fixed_workflow():
    ok, why = ok_and_out(task_type="tool_actions", needs_tools=True, steps_known=False)
    assert "workflow" not in ok and "vary" in why["workflow"] and "tool_agent" in ok


def test_request_types_enable_router_and_multi_model_off_removes_cascade():
    assert "router" in ok_and_out(request_types=4, needs_tools=True, task_type="tool_actions")[0]
    ok, why = ok_and_out(multi_model=False)
    assert "cascade" not in ok and "multi-model" in why["cascade"]


def test_many_tools_or_specialists_enable_orchestrator():
    assert "multi_agent" in ok_and_out(task_type="tool_actions", needs_tools=True, tool_count=16)[0]
    assert "multi_agent" in ok_and_out(task_type="tool_actions", needs_tools=True, specialists=True)[0]
    assert "multi_agent" not in ok_and_out(task_type="tool_actions", needs_tools=True, tool_count=4, latency="realtime")[0]


def test_goal_reading_rules():
    s = make_spec(SUPPORT)
    assert s["needs_documents"] and s["needs_tools"] and s["integration"] == "mcp" and s["specialists"]
    s = make_spec("An IT ops agent that investigates alerts using logs and runbooks. 40 internal tools via MCP.")
    assert s["tool_count"] == 40 and not s["steps_known"]
    s = make_spec("Research competitors and write a weekly report.")
    assert s["needs_tools"] and s["task_type"] == "research"
    assert make_spec("Extract the invoice number from scanned PDFs")["needs_images"]


def test_tool_definitions_counted_and_tool_search_kicks_in():
    small = make_spec(task_type="tool_actions", needs_tools=True, tool_count=5)
    big = make_spec(task_type="tool_actions", needs_tools=True, tool_count=40)
    assert constraints.tools_seen(small, "tool_agent") == 5 and constraints.tools_seen(small, "workflow") == 0
    assert constraints.tools_seen(big, "tool_agent") == 6  # tool search: loads a few definitions on demand


def test_decision_trace_cites_sources():
    trace = advisor.decision_trace(make_spec(SUPPORT))
    assert trace and all(t["sources"] and t["rule"] for t in trace)


def test_labels_dropped_when_expected_answers_are_not_labels():
    """Regression (live run): labels on a multi-field extraction forced one-word answers and 0% accuracy."""
    s = make_spec(task_type="extraction", labels=["refund", "exchange", "cancellation"],
                  test_cases=[{"input": f"email {i}", "expected": f"order_number: {i}; action: refund"} for i in range(8)])
    assert s["labels"] == [] and any("dropped" in a for a in s["assumptions"])
    keep = make_spec(labels=["billing", "sales"], test_cases=[{"input": "a", "expected": "billing"}, {"input": "b", "expected": "sales"}])
    assert keep["labels"] == ["billing", "sales"]


def test_router_not_used_for_single_step_label_or_question_tasks():
    ok, why = ok_and_out(task_type="classification", request_types=4)
    assert "router" not in ok and "one step" in why["router"]
    assert "router" in ok_and_out(task_type="tool_actions", needs_tools=True, request_types=4)[0]


def test_every_test_prompt_states_the_task_and_field_format():
    """Regression (live report): extraction prompts lacked the task, so models replied to the email instead."""
    s = make_spec("Extract the order number and requested action from support emails", task_type="extraction", labels=[],
                  test_cases=[{"input": "Hi, order 58234 arrived broken, refund please", "expected": "order_number: 58234; action: refund"},
                              {"input": "Where is my parcel?", "expected": "order_number: unknown; action: status"}])
    items = advisor.items_for(s)
    assert items[0]["input"].startswith("Task: Extract the order number")
    assert "order_number: ...; action: ..." in items[0]["input"] and "Hi, order 58234" in items[0]["input"]
