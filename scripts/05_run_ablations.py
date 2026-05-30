from __future__ import annotations

import argparse

from _run_utils import baseline_kwargs, dataset_kwargs, load_config, output_dirs, table_rows, write_table
from tool_hallucination_detection import prepare_dataset, run_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--sizes", default="50,100,200")
    args = parser.parse_args()

    config = load_config(args.config)
    _artifacts_dir, results_dir = output_dirs(config)
    rows = []
    for size in [int(item.strip()) for item in args.sizes.split(",") if item.strip()]:
        kwargs = dataset_kwargs(config)
        kwargs["max_base_records"] = size
        dataset = prepare_dataset(**kwargs)
        result = run_baselines(quick=bool(config.get("quick", False)), dataset=dataset, **baseline_kwargs(config))
        for row in table_rows(result["metrics"]):
            rows.append({"base_records": size, **row})
    write_table(rows, results_dir / "ablation_data_size_baselines.csv")
    print(f"Saved {len(rows)} ablation rows")


if __name__ == "__main__":
    main()
