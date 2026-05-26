"""Facade API for the tool-calling hallucination detection project."""

from .facade import (
    evaluate_experiment,
    prepare_dataset,
    predict_spans,
    run_baselines,
    train_best_model,
)

__all__ = [
    "prepare_dataset",
    "run_baselines",
    "train_best_model",
    "predict_spans",
    "evaluate_experiment",
]
