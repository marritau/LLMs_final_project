from tool_hallucination_detection.baselines import _normalize_pipeline_scores, value_checker_predict


def test_normalize_pipeline_scores_accepts_dict():
    assert _normalize_pipeline_scores({"label": "ENTAILMENT", "score": 0.9}) == {"entailment": 0.9}


def test_normalize_pipeline_scores_accepts_nested_lists():
    result = [[{"label": "CONTRADICTION", "score": 0.7}, {"label": "NEUTRAL", "score": 0.2}]]
    assert _normalize_pipeline_scores(result) == {"contradiction": 0.7, "neutral": 0.2}


def test_value_checker_detects_numeric_mismatch():
    record = {
        "query": "How many views are there?",
        "context": '{"views": 150, "title": "Demo"}',
        "tool_call": "",
        "output": "The video has 172 views.",
        "labels": [{"start": 14, "end": 17, "text": "172", "label_type": "tool_contradiction"}],
    }
    prediction = value_checker_predict([record])[0]
    assert prediction["score"] > 0
    assert any(span["text"] == "172" for span in prediction["spans"])


def test_value_checker_avoids_hard_clean_refusal():
    record = {
        "query": "Can you book a flight?",
        "context": '{"weather": "sunny"}',
        "tool_call": "",
        "output": "I cannot book a flight because no booking tool is available.",
        "labels": [],
    }
    prediction = value_checker_predict([record])[0]
    assert prediction["spans"] == []
    assert prediction["score"] == 0.0
