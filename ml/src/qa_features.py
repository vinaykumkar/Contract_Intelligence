"""Build training/evaluation features and decode QA answer spans."""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import WindowFeatures
from .cuad_loader import AnswerSpan, ClauseQA


@dataclass
class TrainFeatures:
    input_ids: list[int]
    start_position: int
    end_position: int
    contract_id: str
    clause_label: str
    qa_id: str
    window_index: int
    is_no_answer: bool
    gold_char_span: tuple[int, int] | None


@dataclass
class EvalFeatures:
    input_ids: list[int]
    contract_id: str
    clause_label: str
    qa_id: str
    window_index: int
    ctx_offsets: list[tuple[int, int]]
    is_no_answer_example: bool


def cls_index() -> int:
    return 0


def token_overlapping_char(
    offsets: list[tuple[int, int]], char_pos: int
) -> int | None:
    """Return the first token containing the given character position."""
    for i, (start, end) in enumerate(offsets):
        if end > start and start <= char_pos < end:
            return i
        if start > char_pos:
            break
    return None


def answer_token_positions(
    window: WindowFeatures, answer: AnswerSpan
) -> tuple[int, int] | None:
    """Map a complete gold answer span to window token positions."""
    if not window.ctx_offsets:
        return None

    win_start, win_end = window.char_range()

    # Do not clip answers that fall outside or cross the window boundary.
    if answer.start < win_start or answer.end > win_end:
        return None

    start = token_overlapping_char(window.ctx_offsets, answer.start)
    end = token_overlapping_char(window.ctx_offsets, answer.end - 1)

    if start is None or end is None or end < start:
        return None

    first = window.ctx_first_token_index_in_window
    return first + start, first + end


def build_train_features(
    windows: list[WindowFeatures], qa: ClauseQA
) -> list[TrainFeatures]:
    """Create training features for all sliding windows."""
    features = []

    gold = None if qa.is_impossible else (qa.answers[0] if qa.answers else None)

    for window in windows:
        span = answer_token_positions(window, gold) if gold else None

        if span is None:
            features.append(
                TrainFeatures(
                    input_ids=window.input_ids,
                    start_position=cls_index(),
                    end_position=cls_index(),
                    contract_id=window.contract_id,
                    clause_label=window.clause_label,
                    qa_id=qa.qa_id,
                    window_index=window.window_index,
                    is_no_answer=True,
                    gold_char_span=None,
                )
            )
        else:
            features.append(
                TrainFeatures(
                    input_ids=window.input_ids,
                    start_position=span[0],
                    end_position=span[1],
                    contract_id=window.contract_id,
                    clause_label=window.clause_label,
                    qa_id=qa.qa_id,
                    window_index=window.window_index,
                    is_no_answer=False,
                    gold_char_span=(gold.start, gold.end),
                )
            )

    return features


def build_eval_features(
    windows: list[WindowFeatures], qa: ClauseQA
) -> list[EvalFeatures]:
    """Create inference features with context offsets and metadata."""
    return [
        EvalFeatures(
            input_ids=w.input_ids,
            contract_id=w.contract_id,
            clause_label=w.clause_label,
            qa_id=qa.qa_id,
            window_index=w.window_index,
            ctx_offsets=list(w.ctx_offsets),
            is_no_answer_example=qa.is_impossible,
        )
        for w in windows
    ]


def decode_token_span(
    feature: EvalFeatures,
    start_token: int,
    end_token: int,
    ctx_first_token_index: int,
) -> tuple[int, int]:
    """Convert predicted token positions to original-text offsets."""
    start = start_token - ctx_first_token_index
    end = end_token - ctx_first_token_index

    if start < 0 or end >= len(feature.ctx_offsets) or end < start:
        raise ValueError("Predicted token span is outside the context.")

    return feature.ctx_offsets[start][0], feature.ctx_offsets[end][1]


def decode_text(context: str, char_start: int, char_end: int) -> str:
    """Extract the predicted evidence from the original contract text."""
    return context[char_start:char_end]
