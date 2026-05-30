# Data Artifacts

Generated ToolACE-derived splits are written to `artifacts/*/dataset/` or to the
configured Kaggle artifact directory. The exported JSONL keeps both fields:

- `labels`
- `hallucination_labels`

Both contain RAGTruth-like span labels with `start`, `end`, `text`,
`label_type`, and `meta`.

Use `data/manual_audit/manual_audit_template.csv` as a manual inspection sheet
for a 100-200 example validation subset.
