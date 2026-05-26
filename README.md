# Hallucination Detection in Tool Calling

Final project for the Skoltech NLP course. The project builds a synthetic
span-level hallucination detection benchmark for tool-calling dialogues, starting
from ToolACE-style conversations, and compares lightweight baselines with an
improved span detector.

The official submission is intended to be a small facade notebook/Colab that
installs this repository and calls the public API in `tool_hallucination_detection`.
Keeping the implementation in a repository makes the notebook reproducible
without turning it into a very large code dump.

Submission notebook: `Hallucination_Detection_in_Tool_Calling.ipynb`.

## Task

For each dialogue we use:

- `query`: the user's request;
- `tools`: available tool descriptions from the system prompt;
- `tool_call`: the assistant tool invocation;
- `context`: the tool response;
- `output`: the final assistant answer.

The dataset contains clean examples and three hallucination types:

- `tool_contradiction`: the final answer contradicts a value from the tool output;
- `overgeneration`: the answer adds unsupported information;
- `missing_tool`: the answer recommends an action requiring an unavailable tool.

Labels follow a RAGTruth-style schema:

```json
{"start": 31, "end": 71, "text": "unsupported text", "label_type": "overgeneration", "meta": "..."}
```

Offsets are character offsets in the `output` field.

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest
python -m tool_hallucination_detection.cli evaluate --quick
```

In Colab, the final notebook should use:

```python
!pip -q install git+https://github.com/marritau/LLMs_final_project.git

from tool_hallucination_detection import evaluate_experiment

result = evaluate_experiment(quick=True)
result["sentence_metrics"]
result["span_metrics"]
```

Use `quick=False` for the full ToolACE run. The full run downloads public
datasets/models and is expected to run on a Colab GPU. If ToolACE cannot be
downloaded, the full run raises the original error instead of silently reporting
quick synthetic metrics.

## Public API

```python
from tool_hallucination_detection import (
    prepare_dataset,
    run_baselines,
    train_best_model,
    predict_spans,
    evaluate_experiment,
)
```

The facade API is intentionally small so that the submitted notebook can be run
cell by cell while delegating the implementation details to this repository.

## Expected Experiments

The report should include sentence-level and span-level metrics for:

- `always_clean`;
- `keyword_tool_action`;
- `nli_sentence_verifier` when the optional NLI model is available;
- `lettucedetect` when the package/model is available;
- `lookback_lens_style` when the optional attention baseline is enabled;
- `improved_ensemble`;
- `modernbert_token_classifier` for the full GPU run.

Sentence-level binary F1 is the main optimization target because the course chat
indicated it is the likely ranking metric. Span-level F1/IoU is still reported
because the assignment asks for span-based hallucination labels.

## Repository Layout

```text
src/tool_hallucination_detection/
  data.py        # ToolACE loading, normalization, split, JSONL export
  corruption.py  # clean and hallucinated record generation
  metrics.py     # sentence and span metrics
  baselines.py   # sanity baselines and optional external baselines
  modeling.py    # optional transformer token classifier
  facade.py      # notebook-facing API
tests/           # fast offline tests
```

## Notes for Submission

Submit one notebook or public Colab plus this GitHub repository link. If time
allows, publish the prepared dataset and trained model weights on Hugging Face
and add the links to the notebook's "Additional comments" section.
