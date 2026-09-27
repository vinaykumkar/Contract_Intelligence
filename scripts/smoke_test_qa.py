#!/usr/bin/env python
"""Phase 2 pretrained QA smoke test (inference plumbing only)."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.src.chunking import build_windows_for_clause, tokenize_context  # noqa: E402
from ml.src.config import (  # noqa: E402
    REPORTS_DIR,
    load_enabled_clauses,
    load_window_config,
)
from ml.src.cuad_loader import load_train_contracts  # noqa: E402
from ml.src.preprocessing import VAL, assign_splits  # noqa: E402

MODEL_ID = "distilbert/distilbert-base-uncased-distilled-squad"
CLAUSES = ["governing_law", "renewal_term", "non_compete"]
MAX_EXAMPLES = 6


def best_span(start, end, first, last, max_tokens):
    """Return best context-only span and start*end confidence."""
    start = torch.softmax(start[first:last], dim=-1)
    end = torch.softmax(end[first:last], dim=-1)

    best = (0, 0, -1.0)
    for i in range(len(start)):
        for j in range(i, min(i + max_tokens, len(start))):
            score = float(start[i] * end[j])
            if score > best[2]:
                best = (i, j, score)
    return best


def main() -> int:
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    enabled = {c.label: c for c in load_enabled_clauses()}
    config = load_window_config()

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", use_fast=True
    )
    config.check_tokenizer_compat(tokenizer.model_max_length)

    started = time.perf_counter()
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
    model.eval()
    print(f"Model loaded in {time.perf_counter() - started:.1f}s")

    records = load_train_contracts(load_enabled_clauses(), strict=True)
    assign_splits(records, seed=42, val_fraction=0.10)
    validation = [r for r in records if r.split == VAL][:2]
    assert validation, "no val contracts selected"

    results = []
    completed = 0

    with torch.inference_mode():
        for record in validation:
            tokens = tokenize_context(tokenizer, record)

            for label in CLAUSES:
                if completed >= MAX_EXAMPLES:
                    break

                clause = enabled[label]
                qids = tokenizer(
                    clause.question, add_special_tokens=False
                )["input_ids"]

                windows = build_windows_for_clause(
                    tokens, qids, label, config,
                    tokenizer.cls_token_id, tokenizer.sep_token_id,
                )
                if not windows:
                    continue

                started = time.perf_counter()
                pad = tokenizer.pad_token_id
                size = max(len(w.input_ids) for w in windows)

                inputs = torch.tensor([
                    w.input_ids + [pad] * (size - len(w.input_ids))
                    for w in windows
                ])
                masks = torch.tensor([
                    [1] * len(w.input_ids) + [0] * (size - len(w.input_ids))
                    for w in windows
                ])

                output = model(input_ids=inputs, attention_mask=masks)
                elapsed = time.perf_counter() - started

                best = None
                for index, window in enumerate(windows):
                    first = window.ctx_first_token_index_in_window
                    last = first + window.ctx_token_end - window.ctx_token_start

                    i, j, score = best_span(
                        output.start_logits[index],
                        output.end_logits[index],
                        first,
                        last,
                        config.max_answer_tokens,
                    )

                    candidate = (index, i, j, score)
                    if best is None or score > best[3]:
                        best = candidate

                index, i, j, score = best
                window = windows[index]

                local_i = i - window.ctx_first_token_index_in_window
                local_j = j - window.ctx_first_token_index_in_window
                char_start = window.ctx_offsets[local_i][0]
                char_end = window.ctx_offsets[local_j][1]
                answer = record.context[char_start:char_end]

                results.append({
                    "contract_id": record.contract_id,
                    "clause": label,
                    "windows": len(windows),
                    "forward_seconds": round(elapsed, 3),
                    "confidence": round(score, 4),
                    "char_offsets": [char_start, char_end],
                    "extracted_text": answer[:300],
                })

                print(
                    f"\n[{record.contract_id[:40]}... | {label}] "
                    f"{len(windows)} windows in {elapsed:.2f}s "
                    f"({len(windows) / elapsed:.1f} win/s)"
                )
                print(f"  confidence={score:.3f} -> {answer[:160]!r}")

                completed += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": MODEL_ID,
            "task": "question-answering (extractive)",
            "license": "apache-2.0",
            "parameters": "~66.4M (fp32, ~253 MB)",
        },
        "device": "cpu",
        "torch_version": torch.__version__,
        "note": (
            "SMOKE TEST ONLY - proves the inference plumbing end to end. "
            "Confidence numbers are zero-shot proxy scores, NOT model performance."
        ),
        "examples": results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "phase2_smoke_test.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSmoke test complete: {completed} examples. Report: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
