import catalog
import constraints
import policy
from conftest import make_spec


def test_unapproved_platforms_and_services_blocked():
    assert not policy.model_allowed("vertex-gemini-2.5-flash")
    assert policy.model_allowed("claude-haiku-4-5") and not policy.service_allowed("jev")
    assert policy.safe_fallback("claude-haiku-4-5") and policy.model_allowed(policy.safe_fallback("claude-haiku-4-5"))


def test_data_class_and_region_rules():
    assert not policy.check_data("gemini-2.5-flash", "regulated")[0]
    assert policy.check_data("bedrock-nova-micro", "regulated")[0]
    assert not policy.check_data("claude-haiku-4-5", "internal", "EU")[0]
    assert policy.check_data("bedrock-nova-micro", "internal", "EU")[0]


def test_screen_rules_out_by_data_tools_and_images():
    keep, out = constraints.screen([m["id"] for m in catalog.CALLABLE if policy.model_allowed(m["id"])],
                                   make_spec(data_class="regulated", needs_images=True))
    plats = {catalog.MODEL_BY_ID[m]["platform"] for m in keep}
    assert plats <= {"Amazon Bedrock", "Azure AI Foundry"}
    assert "bedrock-nova-micro" not in keep  # text-only model can't read images
    assert all(o["reason"] for o in out)


def test_shortlist_spans_tiers_and_another_platform():
    ids = [m["id"] for m in catalog.CALLABLE if policy.model_allowed(m["id"])]
    sl = constraints.shortlist(ids, make_spec(), lambda m: catalog.MODEL_BY_ID[m]["provider"] == "anthropic")
    assert len(sl) == 4 and len({catalog.MODEL_BY_ID[m]["platform"] for m in sl}) >= 2
    assert {catalog.MODEL_BY_ID[m]["tier"] for m in sl} >= {"small", "medium"}
