"""Evaluate extractive QA predictions using SQuAD-style metrics."""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict


def normalize_answer(text: str) -> str:
    """Apply SQuAD-style answer normalization."""
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred_text: str, gold_text: str) -> tuple[float, float, float]:
    """Return F1, precision and recall using token overlap."""
    pred = normalize_answer(pred_text).split()
    gold = normalize_answer(gold_text).split()

    if not pred or not gold:
        return 0.0, 0.0, 0.0

    common = Counter(pred) & Counter(gold)
    same = sum(common.values())

    if not same:
        return 0.0, 0.0, 0.0

    precision = same / len(pred)
    recall = same / len(gold)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def exact_match(pred_text: str, gold_text: str) -> bool:
    return normalize_answer(pred_text) == normalize_answer(gold_text)


def score_prediction_set(
    predictions: list[dict],
    gold_by_pair: dict[tuple[str, str], dict],
    no_answer_delta: float,
    min_confidence: float,
) -> dict:
    """Score predictions using answer/no-answer thresholds."""
    rows = []

    for pred in predictions:
        key = (pred["contract_id"], pred["clause_label"])
        gold = gold_by_pair.get(key)

        if gold is None:
            continue

        found = (
            pred["span_confidence"] >= min_confidence
            and pred["span_confidence"] - pred["cls_confidence"] >= no_answer_delta
        )
        text = pred["text"] if found else ""

        row = {
            "contract_id": pred["contract_id"],
            "clause_label": pred["clause_label"],
            "is_answerable": gold["is_answerable"],
            "found_decision": bool(found),
        }

        if gold["is_answerable"]:
            scores = [
                (1.0 if exact_match(text, g[2]) else 0.0, *token_f1(text, g[2]))
                for g in gold["golds"]
            ]
            best = max(scores, default=(0.0, 0.0, 0.0, 0.0))

            row.update({
                "em": best[0],
                "f1": best[1],
                "precision": best[2],
                "recall": best[3],
            })
        else:
            correct = not found
            row.update({
                "em": float(correct),
                "f1": float(correct),
                "no_answer_correct": correct,
            })

        rows.append(row)

    ans = [r for r in rows if r["is_answerable"]]
    non = [r for r in rows if not r["is_answerable"]]

    def avg(items, key):
        return sum(r[key] for r in items) / len(items) if items else 0.0

    tp = sum(r["found_decision"] for r in ans)
    fn = len(ans) - tp
    fp = sum(r["found_decision"] for r in non)
    tn = len(non) - fp

    recall = tp / len(ans) if ans else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    macro_f1 = (
        2 * recall * precision / (recall + precision)
        if recall + precision else 0.0
    )

    overall = {
        "em": avg(rows, "em"),
        "f1": avg(rows, "f1"),
        "answerable_em": avg(ans, "em"),
        "answerable_f1": avg(ans, "f1"),
        "answerable_precision": avg(ans, "precision"),
        "answerable_recall": avg(ans, "recall"),
        "answerable_pairs": len(ans),
        "no_answer_accuracy": avg(non, "no_answer_correct"),
        "no_answer_pairs": len(non),
        "found_decision": {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "answerable_detection_recall": round(recall, 4),
            "found_precision": round(precision, 4),
            "macro_f1": round(macro_f1, 4),
        },
    }

    by_clause = defaultdict(list)
    for row in rows:
        by_clause[row["clause_label"]].append(row)

    clause_metrics = {}

    for label, items in by_clause.items():
        a = [r for r in items if r["is_answerable"]]
        n = [r for r in items if not r["is_answerable"]]

        clause_metrics[label] = {
            "pairs": len(items),
            "answerable_pairs": len(a),
            "em": avg(items, "em"),
            "f1": avg(items, "f1"),
            "answerable_f1": avg(a, "f1") if a else None,
            "no_answer_accuracy": (
                avg(n, "no_answer_correct") if n else None
            ),
        }

    return {
        "overall": overall,
        "per_clause": clause_metrics,
        "rows": rows,
    }


def sweep_thresholds(
    predictions: list[dict],
    gold_by_pair: dict[tuple[str, str], dict],
    deltas: list[float],
    min_confs: list[float],
) -> list[dict]:
    """Evaluate threshold combinations and rank by decision macro F1."""
    results = []

    for delta in deltas:
        for confidence in min_confs:
            result = score_prediction_set(
                predictions, gold_by_pair, delta, confidence
            )
            overall = result["overall"]

            results.append({
                "no_answer_delta": delta,
                "min_confidence": confidence,
                "macro_f1": overall["found_decision"]["macro_f1"],
                "answerable_em": overall["answerable_em"],
                "answerable_f1": overall["answerable_f1"],
                "overall_f1": overall["f1"],
                "overall_em": overall["em"],
            })

    return sorted(
        results,
        key=lambda x: (
            -x["macro_f1"],
            -x["answerable_em"],
            x["no_answer_delta"],
        ),
    )
