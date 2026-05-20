# Evaluation

The `evaluate.py` helper runs the LangGraph-based RAG workflow against a question set and scores the outputs with [ragas](https://github.com/explodinggradients/ragas).

## Preparing datasets
- Convert the PDPA QA text dumps into JSONL with:
  ```bash
  python eval/convert_pdpa_txt_to_jsonl.py --inputs eval/PDPA_QA_2564.txt eval/PDPA_QA_2565.txt --output eval/questions_pdpa_full.jsonl
  ```
  Use `--limit` if you want only a subset.

## Prerequisites
- Populate your vector store (Qdrant) and make sure the RAG workflow can answer questions.
- Provide assessment credentials for ragas via `OPENAI_API_KEY` and, if needed, `OPENAI_BASE_URL`.
- Optionally set `RAGAS_LLM_MODEL` and `RAGAS_EMBED_MODEL` to override the assessor models.

## Default dataset
The default dataset lives at `eval/questions_pdpa.jsonl`. It must be a JSONL file with at least the keys `question` and `ground_truth`.

## Running from the command line
```bash
python -m eval.evaluate --dataset eval/questions_pdpa.jsonl --output eval/results.json
```

Arguments:
- `--dataset`: path to a JSONL dataset (defaults to `eval/questions_pdpa.jsonl`).
- `--output`: optional path for saving the metric summary and per-sample table as JSON.
- `--llm-model`: override the LLM used by ragas for scoring (default: `gpt-4o-mini`).
- `--embed-model`: override the embedding model used for ragas context metrics (default: `text-embedding-3-small`).
- `--max-samples`: limit the number of rows evaluated (helpful for quick smoke tests).

The script prints overall metrics to stdout and writes a rich JSON artifact when `--output` is set.
