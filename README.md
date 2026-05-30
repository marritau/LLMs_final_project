# Span-Level Hallucination Detection in Tool-Calling Dialogues

Final project for the Skoltech NLP course. The project builds a controlled
ToolACE-derived benchmark for detecting hallucinated spans in final answers of
tool-calling dialogues.

The executable implementation lives in `src/tool_hallucination_detection`.
The reproducible Kaggle entrypoint is `notebooks/kaggle_run_all.ipynb`.

## Task

Each record contains:

```text
query
tools
tool_call
context = tool output
output = final assistant answer
labels = RAGTruth-like hallucinated spans
```

The generated variants are:

- `clean`
- `clean_hard`
- `tool_contradiction`
- `overgeneration`
- `missing_tool`

Splits are grouped by `source_id`, so variants derived from the same ToolACE
dialogue never leak across train, validation, and test.

## Methods

The project compares:

- trivial and random baselines;
- keyword and value-consistency rules;
- TF-IDF + Logistic Regression;
- NLI sentence verification;
- LettuceDetect as a real external RAG hallucination detector;
- LookBackLens-style proxy and adapted attention baseline;
- fine-tuned `answerdotai/ModernBERT-base` token classifier;
- `Hybrid ModernBERT + Value Consistency Detector`.

The main model is ModernBERT token classification over answer tokens. Query and
tool context tokens are masked from the loss.

## Quick Local Check

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/01_build_dataset.py --config configs/debug.yaml
python scripts/02_run_baselines.py --config configs/debug.yaml
python scripts/04_evaluate.py --config configs/debug.yaml
```

The debug config uses synthetic offline examples and does not require GPU or
network access.

## Kaggle Full Run

In Kaggle, enable GPU and Internet, then run:

```text
notebooks/kaggle_run_all.ipynb
```

The notebook performs a clean clone, installs dependencies, checks
LettuceDetect, builds the full dataset, runs baselines, trains ModernBERT,
evaluates the trained model, and creates a lightweight results archive.

Equivalent command sequence:

```bash
python scripts/00_check_lettucedetect.py
python scripts/01_build_dataset.py --config configs/kaggle_full.yaml
python scripts/02_run_baselines.py --config configs/kaggle_full.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/03_train_modernbert.py --config configs/kaggle_full.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/04_evaluate.py \
  --config configs/kaggle_full.yaml \
  --model-path /kaggle/working/artifacts/kaggle_full/modernbert-token-classifier
python scripts/06_make_report_tables.py --config configs/kaggle_full.yaml
```

`CUDA_VISIBLE_DEVICES=0` avoids unstable multi-GPU `DataParallel` behavior on
Kaggle 2x T4 sessions. Passing `--model-path` prevents evaluation from training
a second model.

## Configs

- `configs/debug.yaml`: local smoke test.
- `configs/kaggle_small.yaml`: 200 base records, 1 epoch.
- `configs/kaggle_full.yaml`: 1000 raw ToolACE rows, 3 epochs, strict external baselines.
- `configs/kaggle_full_safe.yaml`: full-size pipeline check with non-strict external baseline handling.
- `configs/baselines.yaml`: baseline families.
- `configs/model_modernbert.yaml`: model ablation settings.

All experiment configs use dataset seed `42`. ModernBERT training also sets
`seed=42` and `data_seed=42` in Hugging Face `TrainingArguments`.

## Final Full Results

The final Kaggle full run uses 797 valid ToolACE base dialogues and five
variants per dialogue:

```text
train: 2790
validation: 600
test: 595
total: 3985
```

On the full test split, the best method is `modernbert_token_classifier`:

| Metric | Value |
| --- | ---: |
| Sentence macro-F1 | 0.962 |
| Balanced accuracy | 0.963 |
| Span character F1 | 0.983 |
| Relaxed span F1 | 0.928 |

TF-IDF + Logistic Regression is strong at sentence-level detection but weak at
span localization. LettuceDetect runs as a real external baseline, but transfers
imperfectly from RAG to tool-calling hallucinations. The hardest remaining type
is `tool_contradiction`.

## Outputs

Generated files are written to:

```text
artifacts/
results/
```

The main report tables are:

```text
results/kaggle_full/all_metrics.csv
results/kaggle_full/per_type_metrics.csv
results/kaggle_full/baseline_availability.csv
results/kaggle_full/dataset_statistics.csv
```

Do not commit model weights, checkpoints, `.safetensors`, `.pt`, `.bin`, or
large Kaggle archives.

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

## Known Limitations

- The benchmark uses controlled corruptions, not naturally occurring agent errors.
- `tool_contradiction` remains the hardest hallucination type.
- The current hybrid rule is conservative and does not outperform ModernBERT.
- `attention_lookback_lens_adapted` is an approximation, not a full LookBack Lens reproduction.
