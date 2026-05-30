from __future__ import annotations

import argparse

from _run_utils import dataset_kwargs, load_config, output_dirs, training_kwargs, write_json
from tool_hallucination_detection import prepare_dataset, train_best_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts_dir, _results_dir = output_dirs(config)
    dataset = prepare_dataset(**dataset_kwargs(config))
    model_or_detector = train_best_model(quick=bool(config.get("quick", False)), dataset=dataset, **training_kwargs(config))
    if hasattr(model_or_detector, "predict"):
        write_json({"mode": "quick_or_fallback_detector", "training_error": getattr(model_or_detector, "training_error", None)}, artifacts_dir / "training_result.json")
    else:
        write_json({"model_path": str(model_or_detector)}, artifacts_dir / "training_result.json")
    print(model_or_detector)


if __name__ == "__main__":
    main()
