from __future__ import annotations

import argparse

from _run_utils import corruption_audit, dataset_kwargs, dataset_statistics, load_config, output_dirs, write_json, write_table
from tool_hallucination_detection import export_splits, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts_dir, results_dir = output_dirs(config)
    kwargs = dataset_kwargs(config)
    dataset = prepare_dataset(**kwargs)
    export_dir = kwargs.get("cache_dir") or artifacts_dir / "dataset"
    paths = export_splits(dataset, export_dir)
    write_json(paths, artifacts_dir / "dataset_paths.json")
    write_table(dataset_statistics(dataset), results_dir / "dataset_statistics.csv")
    write_table(corruption_audit(dataset), results_dir / "corruption_audit.csv")
    for split, records in dataset.items():
        print(f"{split}: {len(records)} records")


if __name__ == "__main__":
    main()
