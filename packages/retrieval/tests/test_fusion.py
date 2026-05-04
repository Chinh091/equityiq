from equityiq_retrieval import reciprocal_rank_fusion


def test_single_ranker_yields_decreasing_scores():
    fused = reciprocal_rank_fusion([[1, 2, 3]], k=60)
    ids = list(fused.keys())
    assert ids == [1, 2, 3]
    assert fused[1] > fused[2] > fused[3]


def test_doc_in_both_rankers_outranks_single_appearance():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]], k=60)
    # 3 appears in both; should beat docs that appear in only one.
    assert list(fused.keys())[0] in (1, 3)
    assert fused[3] > fused[5]
    assert fused[3] > fused[2]


def test_missing_from_all_rankers_is_absent():
    fused = reciprocal_rank_fusion([[1, 2], [2, 3]], k=60)
    assert 4 not in fused


def test_k_controls_top_rank_dominance():
    low_k = reciprocal_rank_fusion([[1, 2, 3]], k=1)
    high_k = reciprocal_rank_fusion([[1, 2, 3]], k=1000)
    # Smaller k → bigger gap between rank 1 and rank 2.
    assert low_k[1] - low_k[2] > high_k[1] - high_k[2]
