from tool_hallucination_detection.baselines import _normalize_pipeline_scores


def test_normalize_pipeline_scores_accepts_dict():
    assert _normalize_pipeline_scores({"label": "ENTAILMENT", "score": 0.9}) == {"entailment": 0.9}


def test_normalize_pipeline_scores_accepts_nested_lists():
    result = [[{"label": "CONTRADICTION", "score": 0.7}, {"label": "NEUTRAL", "score": 0.2}]]
    assert _normalize_pipeline_scores(result) == {"contradiction": 0.7, "neutral": 0.2}
