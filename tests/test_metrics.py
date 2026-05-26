from tool_hallucination_detection.metrics import evaluate_predictions, span_metrics


def test_span_metrics_char_overlap_and_relaxed_iou():
    gold = [[{"start": 10, "end": 20, "text": "abcdefghij", "label_type": "x"}]]
    pred = [[{"start": 12, "end": 20, "text": "cdefghij", "label_type": "x"}]]
    metrics = span_metrics(gold, pred, output_lengths=[30], iou_threshold=0.5)
    assert metrics["char_precision"] == 1.0
    assert metrics["char_recall"] == 0.8
    assert metrics["relaxed_span_f1"] == 1.0


def test_evaluate_predictions_sentence_label_from_spans():
    records = [
        {"output": "clean", "labels": []},
        {"output": "bad span", "labels": [{"start": 0, "end": 3, "text": "bad", "label_type": "x"}]},
    ]
    predictions = [
        {"score": 0.1, "spans": []},
        {"score": 0.9, "spans": [{"start": 0, "end": 3, "text": "bad", "label_type": "x"}]},
    ]
    evaluated = evaluate_predictions(records, predictions, threshold=0.5)
    assert evaluated["sentence"]["f1"] == 1.0
    assert evaluated["span"]["char_f1"] == 1.0
