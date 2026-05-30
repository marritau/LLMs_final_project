from tool_hallucination_detection.corruption import build_corrupted_dataset
from tool_hallucination_detection.data import split_by_source_id, synthetic_toolace_records
from tool_hallucination_detection.schema import validate_labels


def test_corruption_labels_match_offsets():
    records = build_corrupted_dataset(synthetic_toolace_records())
    assert records
    for record in records:
        validate_labels(record)
        if record["corruption_type"] in {"clean", "clean_hard"}:
            assert record["labels"] == []
        else:
            assert record["labels"]


def test_build_dataset_can_include_hard_clean_examples():
    records = build_corrupted_dataset(synthetic_toolace_records(), include_clean_hard=True)
    hard_clean = [record for record in records if record["corruption_type"] == "clean_hard"]
    assert hard_clean
    assert all(record["labels"] == [] for record in hard_clean)
    assert all(record["corruption_style"] for record in hard_clean)


def test_split_keeps_source_variants_together():
    records = build_corrupted_dataset(synthetic_toolace_records())
    splits = split_by_source_id(records, seed=42)
    source_to_split = {}
    for split, split_records in splits.items():
        assert split_records
        for record in split_records:
            previous = source_to_split.setdefault(record["source_id"], split)
            assert previous == split
