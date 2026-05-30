# Span-Level Hallucination Detection in Tool-Calling Dialogues

Final project for the Skoltech NLP course. The repository builds a controlled
ToolACE-derived benchmark for span-level hallucination detection in tool-calling
dialogues and compares trivial, lexical, NLI, LettuceDetect, LookBackLens-style,
ModernBERT, and hybrid neural-symbolic detectors.

The current source of truth for the executed assignment run is
`Urazmetova_final_project.ipynb`; the reusable implementation is mirrored in
`src/tool_hallucination_detection`.

## Project Idea

Each example has:

```text
query
tools
tool_call
context = tool output
output = final assistant answer
labels = RAGTruth-like hallucinated spans
```

The benchmark creates clean and corrupted variants:

- `clean`
- `clean_hard`, with refusal/limitation wording and empty labels
- `tool_contradiction`
- `overgeneration`
- `missing_tool`

The main improved method is:

```text
Hybrid ModernBERT + Value Consistency Detector
```

ModernBERT handles general unsupported spans, while the value checker targets
short numeric/date/entity/unit/status mismatches that are common in
`tool_contradiction`.

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest
python scripts/01_build_dataset.py --config configs/debug.yaml
python scripts/02_run_baselines.py --config configs/debug.yaml
python scripts/04_evaluate.py --config configs/debug.yaml
```

The debug config uses the offline synthetic sample, so it does not require
network access or GPU.

## Kaggle GPU Run

1. Create a Kaggle Notebook.
2. Settings -> Accelerator -> GPU.
3. Internet: On.
4. Upload or clone this repository.
5. Run `notebooks/kaggle_run_all.ipynb`.

Equivalent commands:

```bash
pip install -q -r requirements-kaggle.txt
pip install -q -e ".[dev]"
python -m py_compile src/tool_hallucination_detection/*.py
python -m pytest -q
python scripts/00_check_lettucedetect.py
```

Debug smoke run:

```bash
python scripts/01_build_dataset.py --config configs/debug.yaml
python scripts/02_run_baselines.py --config configs/debug.yaml
python scripts/04_evaluate.py --config configs/debug.yaml
python scripts/06_make_report_tables.py --config configs/debug.yaml
```

Small GPU run:

```bash
python scripts/01_build_dataset.py --config configs/kaggle_small.yaml
python scripts/02_run_baselines.py --config configs/kaggle_small.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/03_train_modernbert.py --config configs/kaggle_small.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/04_evaluate.py \
  --config configs/kaggle_small.yaml \
  --model-path /kaggle/working/artifacts/kaggle_small/modernbert-token-classifier
python scripts/06_make_report_tables.py --config configs/kaggle_small.yaml
```

Full-safe pipeline check:

```bash
python scripts/01_build_dataset.py --config configs/kaggle_full_safe.yaml
python scripts/02_run_baselines.py --config configs/kaggle_full_safe.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/03_train_modernbert.py --config configs/kaggle_full_safe.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/04_evaluate.py \
  --config configs/kaggle_full_safe.yaml \
  --model-path /kaggle/working/artifacts/kaggle_full/modernbert-token-classifier
python scripts/06_make_report_tables.py --config configs/kaggle_full_safe.yaml
```

Final full run:

```bash
python scripts/01_build_dataset.py --config configs/kaggle_full.yaml
python scripts/02_run_baselines.py --config configs/kaggle_full.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/03_train_modernbert.py --config configs/kaggle_full.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/04_evaluate.py \
  --config configs/kaggle_full.yaml \
  --model-path /kaggle/working/artifacts/kaggle_full/modernbert-token-classifier
python scripts/06_make_report_tables.py --config configs/kaggle_full.yaml
```

`CUDA_VISIBLE_DEVICES=0` keeps Hugging Face Trainer out of `DataParallel` on
2x T4 Kaggle sessions. Always pass `--model-path` after running
`03_train_modernbert.py`; otherwise evaluation may train a second model if no
saved model is found. LettuceDetect is configured through
`evaluation.lettuce_model_path` in the Kaggle YAML files.

Kaggle outputs are written to:

```text
/kaggle/working/artifacts/
/kaggle/working/results/
```

## Configs

- `configs/debug.yaml`: offline smoke run, 50-record setting, no external models.
- `configs/kaggle_small.yaml`: ToolACE small run, 200 base records, 1 epoch.
- `configs/kaggle_full.yaml`: ToolACE full run, 1000 base records, 3 epochs.
- `configs/kaggle_full_safe.yaml`: full dataset pipeline check that does not fail the run if external LettuceDetect is unavailable.
- `configs/baselines.yaml`: documented baseline families.
- `configs/model_modernbert.yaml`: ablation grid for model settings.

## Baselines and Methods

Implemented baseline families:

- `always_clean`, `always_hallucinated`
- `random_balanced`, `random_prior`
- `keyword_tool_action`
- `tfidf_logreg`
- `value_checker`
- `nli_sentence_verifier`
- `lettucedetect_real`
- `lookback_lens_style_proxy`
- `attention_lookback_lens_adapted`

Improved methods:

- `modernbert_token_classifier`
- `hybrid_modernbert_value_checker`
- `hybrid_value_ensemble` for quick/no-GPU fallback

## Public API

```python
from tool_hallucination_detection import (
    prepare_dataset,
    run_baselines,
    train_best_model,
    predict_spans,
    evaluate_experiment,
    export_splits,
)
```

## Repository Layout

```text
configs/       experiment configs
scripts/       reproducible command-line experiment steps
notebooks/     Kaggle wrapper notebook
src/           reusable package implementation
tests/         fast offline tests
artifacts/     generated datasets/models/thresholds, gitignored
results/       generated tables, gitignored
```

## Report Tables

The scripts generate:

- dataset statistics
- corruption audit with span length, insertion position, and style
- baseline metrics
- sentence/span metrics
- per-type metrics
- data-size baseline ablation

Use `scripts/06_make_report_tables.py` after a run to refresh dataset and
corruption audit tables.

## Current Kaggle Small Results

On `kaggle_small.yaml` we evaluated 150 test examples with positive rate 0.6.
The best method in the current run is `modernbert_token_classifier`:

- sentence macro-F1: 0.772
- balanced accuracy: 0.792
- span char-F1: 0.910
- relaxed span-F1: 0.601

TF-IDF + Logistic Regression is competitive at sentence level but fails at span
localization. The hardest class is `tool_contradiction`, where ModernBERT recall
remains low.

## Known Limitations

- `tool_contradiction` remains the hardest type.
- `hybrid_modernbert_value_checker` currently does not improve span localization.
- `attention_lookback_lens_adapted` is an approximation, not a full reproduction of LookBack Lens.
- LettuceDetect requires an explicit `lettuce_model_path`.
