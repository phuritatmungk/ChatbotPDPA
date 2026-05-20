from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _preprocess_text(raw: str) -> str:
    text = raw.replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"Q(\d+)\s+", r"Q\1: ", text)
    text = re.sub(r"A(\d+)\s+", r"A\1: ", text)
    text = re.sub(r"Q(\d+)\.", r"Q\1: ", text)
    text = re.sub(r"A(\d+)\.", r"A\1: ", text)
    text = re.sub(r"–", "-", text)
    return text


def _tokenize(text: str) -> List[Tuple[str, str, str]]:
    question_pattern = re.compile(r"^(Q\d*(?:\.\d+)?)[:\-]?\s*(.*)$", re.IGNORECASE)
    answer_pattern = re.compile(r"^(A\d*(?:\.\d+)?)[:\-]?\s*(.*)$", re.IGNORECASE)

    tokens: List[Tuple[str, str, str]] = []
    current_kind: str | None = None
    current_label: str | None = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_kind, current_label, buffer
        if current_kind and buffer:
            text_value = _normalize_whitespace(" ".join(buffer))
            if text_value:
                tokens.append((current_kind, current_label or current_kind, text_value))
        current_kind = None
        current_label = None
        buffer = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if all(ch in {"=", "-", "_", "~"} for ch in line):
            continue
        if re.fullmatch(r"\d+\.?", line):
            continue

        q_match = question_pattern.match(line)
        if q_match:
            flush()
            current_kind = "Q"
            current_label = q_match.group(1).upper()
            remainder = q_match.group(2).strip()
            buffer = [remainder] if remainder else []
            continue

        a_match = answer_pattern.match(line)
        if a_match:
            flush()
            current_kind = "A"
            current_label = a_match.group(1).upper()
            remainder = a_match.group(2).strip()
            buffer = [remainder] if remainder else []
            continue

        if current_kind:
            buffer.append(line)

    flush()
    return tokens


def _pair_tokens(tokens: Iterable[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    def extract_base(label: str) -> str:
        match = re.match(r"[QA](\d+(?:\.\d+)?)", label)
        if match:
            return match.group(1)
        return "__default__"

    pending_by_base: Dict[str, List[Dict[str, str]]] = {}
    pending_queue: List[Dict[str, str]] = []
    entries: List[Dict[str, str]] = []
    seen_questions: set[str] = set()
    question_id = 0

    for kind, label, text in tokens:
        base = extract_base(label)
        if kind == "Q":
            question_text = _normalize_whitespace(text)
            if not question_text:
                continue
            question_id += 1
            item = {"id": question_id, "base": base, "text": question_text}
            pending_by_base.setdefault(base, []).append(item)
            pending_queue.append(item)
        elif kind == "A":
            answer_text = _normalize_whitespace(text)
            if not answer_text:
                continue
            item = None
            base_list = pending_by_base.get(base)
            if base_list:
                item = base_list.pop(0)
                if not base_list:
                    pending_by_base.pop(base, None)
            elif pending_queue:
                item = pending_queue.pop(0)
                base_list = pending_by_base.get(item["base"])
                if base_list:
                    try:
                        base_list.remove(item)
                    except ValueError:
                        pass
                    if not base_list:
                        pending_by_base.pop(item["base"], None)
            if item is None:
                continue
            if item in pending_queue:
                pending_queue.remove(item)
            question_text = item["text"]
            if not question_text or question_text in seen_questions:
                continue
            seen_questions.add(question_text)
            entries.append(
                {"question": question_text, "ground_truth": answer_text}
            )

    return entries


def parse_file(path: Path) -> List[Dict[str, str]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _preprocess_text(raw)
    tokens = _tokenize(text)
    return _pair_tokens(tokens)


def convert_files(input_paths: List[Path]) -> List[Dict[str, str]]:
    all_entries: List[Dict[str, str]] = []
    for path in input_paths:
        entries = parse_file(path)
        all_entries.extend(entries)
    return all_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDPA QA text files into JSONL for RAG evaluation."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to PDPA QA text files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of entries to write.",
    )
    args = parser.parse_args()

    entries = convert_files(args.inputs)
    if args.limit is not None:
        entries = entries[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()

