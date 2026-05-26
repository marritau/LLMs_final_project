"""Build the final submission notebook from a compact cell specification."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Hallucination_Detection_in_Tool_Calling.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md("# 1. Information about the submission\n"),
    md("## 1.1 Name and number of the assignment\n"),
    md("Final project. Hallucination Detection in Tool Calling\n"),
    md("## 1.2 Student name\n"),
    md("TODO: fill in student name before submission. Solo submission.\n"),
    md("## 1.3 Codalab user ID / nickname / username\n"),
    md("No public CodaLab leaderboard was specified for this final project. Repository: https://github.com/marritau/LLMs_final_project\n"),
    md("## 1.4 Additional comments\n"),
    md(
        "This notebook is intentionally a facade over a public GitHub repository. "
        "The course chat allowed repository-based submissions when the project has an elaborated file structure. "
        "The repository can be installed with `pip install git+https://github.com/marritau/LLMs_final_project.git`, "
        "and the notebook calls the public facade functions from `tool_hallucination_detection`.\n"
    ),
    md("# 2. Technical Report\n"),
    md("## 2.1 Methodology\n"),
    md(
        "The project builds a span-level hallucination detection benchmark for tool-calling dialogues. "
        "The base data source is ToolACE, which contains user queries, available tool descriptions, assistant tool calls, tool outputs, and final assistant answers. "
        "Each dialogue is normalized into a RAGTruth-style record with `query`, `context` as the tool output, and `output` as the final answer. "
        "The clean answer is used as a negative example, and three automatic corruption procedures produce positive examples with character-level span labels.\n\n"
        "The three hallucination types follow the assignment statement. "
        "`tool_contradiction` replaces a value supported by the tool output with a conflicting value in the final answer. "
        "`overgeneration` appends plausible but unsupported information. "
        "`missing_tool` adds a recommendation that would require a tool absent from the available tool list. "
        "All labels are stored as spans with exact character offsets in the final answer, so the dataset supports both sentence-level binary evaluation and span-level detection.\n\n"
        "The baseline suite contains an always-clean sanity baseline, a keyword/tool-action heuristic, an optional NLI sentence verifier, an optional LettuceDetect wrapper, and a LookBackLens-style support baseline. "
        "The full neural model is a token-classification encoder that receives `Question`, `Context`, and `Answer`; labels are applied only to answer tokens, while question and context tokens are ignored. "
        "For Colab-scale experiments the repository exposes a ModernBERT token-classifier training function, and the default quick path uses a fitted lightweight ensemble to keep the notebook reproducible without downloading large models.\n\n"
        "The primary optimization target is sentence-level binary F1 because the course chat indicated that sentence-based ranking is the most likely evaluation signal. "
        "Span-level character F1, exact span F1, and relaxed IoU-based span F1 are still reported because the task explicitly asks for span-based labels.\n"
    ),
    md("## 2.2 Discussion of results\n"),
    md(
        "The quick run below is a smoke test on a tiny synthetic ToolACE-like sample; it verifies the full pipeline, offset handling, and metric computation. "
        "The full experiment should be run with `quick=False` on Colab/GPU to download ToolACE and optional model baselines. "
        "The report table produced by the code compares the sanity baselines, LookBackLens-style scoring, and the improved ensemble. "
        "For the final write-up, replace or augment the quick table with the full run table and include per-hallucination-type analysis for contradiction, overgeneration, and missing-tool cases.\n"
    ),
    md("# 3. Code\n"),
    md("## 3.1 Requirements\n"),
    code(
        "# In Colab or a clean environment, install from the public repository.\n"
        "# If this notebook is run from inside the cloned repository, the fallback editable install is used.\n"
        "import importlib.util\n"
        "from pathlib import Path\n"
        "\n"
        "if importlib.util.find_spec('tool_hallucination_detection') is None:\n"
        "    repo_root = Path.cwd()\n"
        "    if (repo_root / 'pyproject.toml').exists() and (repo_root / 'src').exists():\n"
        "        %pip -q install -e .\n"
        "    else:\n"
        "        %pip -q install git+https://github.com/marritau/LLMs_final_project.git\n"
    ),
    md("## 3.2 Download and prepare the data\n"),
    code(
        "from tool_hallucination_detection import prepare_dataset\n"
        "\n"
        "# quick=True uses a tiny offline sample. Use quick=False for the full ToolACE-based run.\n"
        "dataset = prepare_dataset(quick=True, cache_dir=None)\n"
        "{split: len(records) for split, records in dataset.items()}\n"
    ),
    md("## 3.3 Preprocessing and label validation\n"),
    code(
        "from tool_hallucination_detection.schema import validate_labels\n"
        "\n"
        "for split, records in dataset.items():\n"
        "    for record in records:\n"
        "        validate_labels(record)\n"
        "\n"
        "example = dataset['train'][0]\n"
        "example\n"
    ),
    md("## 3.4 Experiments\n"),
    code(
        "from tool_hallucination_detection import evaluate_experiment\n"
        "\n"
        "result = evaluate_experiment(quick=True)\n"
        "result['all_metrics']\n"
    ),
    md("### Sentence-level metrics\n"),
    code("result['sentence_metrics']\n"),
    md("### Span-level metrics\n"),
    code("result['span_metrics']\n"),
    md("### Per-type analysis\n"),
    code("result['per_type_metrics']\n"),
    md("## 3.5 Full GPU run\n"),
    code(
        "# Run this cell in Colab/GPU for the final numbers.\n"
        "# It downloads ToolACE and attempts optional model-based baselines/training.\n"
        "# full_result = evaluate_experiment(quick=False)\n"
        "# full_result['all_metrics']\n"
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(OUTPUT)
