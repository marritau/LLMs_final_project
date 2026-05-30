from __future__ import annotations

import argparse
from pathlib import Path

from _run_utils import baseline_kwargs, dataset_kwargs, load_config, output_dirs, training_kwargs, write_json, write_table
from tool_hallucination_detection import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to an already trained token classifier. If omitted, the script reuses training.output_dir when it contains a saved model.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore any existing training.output_dir model and train inside evaluate_experiment.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts_dir, results_dir = output_dirs(config)
    dataset = dataset_kwargs(config)
    training = training_kwargs(config)
    evaluation = config.get("evaluation", {}) or {}
    model_path = args.model_path
    if model_path is None and not args.retrain:
        model_path = _existing_model_path(training.get("output_dir"))
        if model_path is not None:
            print(f"Using existing model from {model_path}. Pass --retrain to train during evaluation.")
    result = evaluate_experiment(
        quick=dataset["quick"],
        seed=dataset["seed"],
        cache_dir=dataset["cache_dir"],
        max_base_records=dataset["max_base_records"],
        include_clean_hard=dataset["include_clean_hard"],
        model_path=model_path,
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


def _existing_model_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    has_config = (candidate / "config.json").exists()
    has_weights = any((candidate / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin"))
    has_tokenizer = any((candidate / filename).exists() for filename in ("tokenizer.json", "tokenizer_config.json"))
    if has_config and has_weights and has_tokenizer:
        return str(candidate)
    return None


if __name__ == "__main__":
    main()
