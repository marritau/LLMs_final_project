from __future__ import annotations

import argparse

from _run_utils import baseline_kwargs, dataset_kwargs, load_config, output_dirs, write_json, write_table
from tool_hallucination_detection import prepare_dataset, run_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts_dir, results_dir = output_dirs(config)
    dataset = prepare_dataset(**dataset_kwargs(config))
    result = run_baselines(quick=bool(config.get("quick", False)), dataset=dataset, **baseline_kwargs(config))
    write_table(result["metrics"], results_dir / "baseline_metrics.csv")
    write_table(result.get("availability", []), results_dir / "baseline_availability.csv")
    write_json(result.get("thresholds", {}), artifacts_dir / "baseline_thresholds.json")
    write_json(result.get("predictions", {}), artifacts_dir / "baseline_predictions.json")
    print(result["metrics"])


if __name__ == "__main__":
    main()
