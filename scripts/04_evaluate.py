from __future__ import annotations

import argparse

from _run_utils import baseline_kwargs, dataset_kwargs, load_config, output_dirs, training_kwargs, write_json, write_table
from tool_hallucination_detection import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts_dir, results_dir = output_dirs(config)
    dataset = dataset_kwargs(config)
    training = training_kwargs(config)
    evaluation = config.get("evaluation", {}) or {}
    result = evaluate_experiment(
        quick=dataset["quick"],
        seed=dataset["seed"],
        cache_dir=dataset["cache_dir"],
        max_base_records=dataset["max_base_records"],
        include_clean_hard=dataset["include_clean_hard"],
        model_path=args.model_path,
        strict_training=training.pop("strict"),
        score_aggregator=evaluation.get("score_aggregator", "max_token_prob"),
        **training,
        **baseline_kwargs(config),
    )
    write_table(result["all_metrics"], results_dir / "all_metrics.csv")
    write_table(result["sentence_metrics"], results_dir / "sentence_metrics.csv")
    write_table(result["span_metrics"], results_dir / "span_metrics.csv")
    write_table(result["per_type_metrics"], results_dir / "per_type_metrics.csv")
    write_table(result.get("baseline_availability", []), results_dir / "baseline_availability.csv")
    write_json(result.get("thresholds", {}), artifacts_dir / "thresholds.json")
    write_json(result.get("threshold_info", {}), artifacts_dir / "threshold_info.json")
    write_json({"model_path": result.get("model_path"), "training_error": result.get("training_error")}, artifacts_dir / "evaluation_summary.json")
    print(result["all_metrics"])


if __name__ == "__main__":
    main()
