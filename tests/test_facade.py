from tool_hallucination_detection import evaluate_experiment, prepare_dataset


def test_prepare_dataset_quick_has_all_splits():
    dataset = prepare_dataset(quick=True, cache_dir=None)
    assert set(dataset) == {"train", "validation", "test"}
    assert all(dataset[split] for split in dataset)


def test_evaluate_experiment_quick_runs():
    result = evaluate_experiment(quick=True)
    assert "sentence_metrics" in result
    assert "span_metrics" in result
    assert "training_error" in result
    assert len(result["test_records"]) > 0
