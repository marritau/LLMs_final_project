from __future__ import annotations

import argparse

from _run_utils import corruption_audit, dataset_kwargs, dataset_statistics, load_config, output_dirs, write_table
from tool_hallucination_detection import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    _artifacts_dir, results_dir = output_dirs(config)
    dataset = prepare_dataset(**dataset_kwargs(config))
    write_table(dataset_statistics(dataset), results_dir / "dataset_statistics.csv")
    write_table(corruption_audit(dataset), results_dir / "corruption_audit.csv")
    print(f"Report tables written to {results_dir}")


if __name__ == "__main__":
    main()
