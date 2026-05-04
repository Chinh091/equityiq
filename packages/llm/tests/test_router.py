from equityiq_llm import ModelRouter, ModelTier


def test_short_query_routes_to_fallback():
    r = ModelRouter()
    decision = r.route("AAPL latest revenue?")
    assert decision.tier is ModelTier.FALLBACK


def test_citation_required_routes_to_primary():
    r = ModelRouter()
    decision = r.route("AAPL latest revenue?", requires_citations=True)
    assert decision.tier is ModelTier.PRIMARY


def test_complexity_marker_routes_to_primary():
    r = ModelRouter()
    decision = r.route("compare NVDA and AMD margins year over year")
    assert decision.tier is ModelTier.PRIMARY
    assert "complexity" in decision.reason


def test_long_query_routes_to_primary():
    r = ModelRouter(long_query_chars=50)
    decision = r.route("a" * 51)
    assert decision.tier is ModelTier.PRIMARY


def test_force_primary_overrides_all():
    r = ModelRouter(force_primary=True)
    decision = r.route("hi")
    assert decision.tier is ModelTier.PRIMARY
