"""Token-level sliding-window chunking for long CUAD contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .config import WindowConfig
from .cuad_loader import ContractRecord


class ChunkingError(RuntimeError):
    """Raised when a question/window cannot fit the configured token budget."""


@dataclass
class TokenizedContext:
    """Contract context tokenized once and reused across clauses."""
    contract_id: str
    context: str
    token_ids: list[int]
    offsets: list[tuple[int, int]]


@dataclass
class WindowFeatures:
    """One QA window: [CLS] question [SEP] context [SEP]."""
    input_ids: list[int]
    question_token_len: int
    ctx_token_start: int
    ctx_token_end: int
    ctx_offsets: list[tuple[int, int]]
    contract_id: str
    clause_label: str
    window_index: int

    @property
    def ctx_first_token_index_in_window(self) -> int:
        return self.question_token_len + 2

    def char_range(self) -> tuple[int, int]:
        return self.ctx_offsets[0][0], self.ctx_offsets[-1][1]


def tokenize_context(tokenizer, record: ContractRecord) -> TokenizedContext:
    """Tokenize the contract context once while preserving character offsets."""
    enc = tokenizer(
        record.context,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    return TokenizedContext(
        contract_id=record.contract_id,
        context=record.context,
        token_ids=list(enc["input_ids"]),
        offsets=[(int(s), int(e)) for s, e in enc["offset_mapping"]],
    )


def validate_window_fit(
    config: WindowConfig,
    tokenizer,
    question: str,
    clause: str,
) -> int:
    """Return question token count if it fits without truncation."""
    q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    special = 3

    if len(q_ids) + special >= config.max_seq_len:
        raise ChunkingError(
            f"Question for '{clause}' does not fit max_seq_len="
            f"{config.max_seq_len}."
        )

    return len(q_ids)


def window_token_ranges(
    n_ctx_tokens: int,
    max_ctx_tokens: int,
    doc_stride: int,
) -> list[tuple[int, int]]:
    """Generate overlapping [start, end) context token ranges."""
    if max_ctx_tokens <= doc_stride:
        raise ChunkingError(
            f"max_ctx_tokens ({max_ctx_tokens}) must exceed "
            f"doc_stride ({doc_stride})"
        )

    if not n_ctx_tokens:
        return []

    ranges = []
    start = 0

    while start < n_ctx_tokens:
        end = min(start + max_ctx_tokens, n_ctx_tokens)
        ranges.append((start, end))

        if end == n_ctx_tokens:
            break

        start += max_ctx_tokens - doc_stride

    return ranges


def build_windows_for_clause(
    tokenized: TokenizedContext,
    question_token_ids: list[int],
    clause_label: str,
    config: WindowConfig,
    cls_token_id: int,
    sep_token_id: int,
) -> list[WindowFeatures]:
    """Build windows for the complete contract context."""
    return build_windows_for_token_slice(
        tokenized,
        0,
        len(tokenized.token_ids),
        question_token_ids,
        clause_label,
        config,
        cls_token_id,
        sep_token_id,
    )


def build_windows_for_token_slice(
    tokenized: TokenizedContext,
    slice_start: int,
    slice_end: int,
    question_token_ids: list[int],
    clause_label: str,
    config: WindowConfig,
    cls_token_id: int,
    sep_token_id: int,
    first_window_index: int = 0,
) -> list[WindowFeatures]:
    """Build sliding windows over a tokenized context region."""
    max_ctx = config.max_seq_len - len(question_token_ids) - 3
    windows = []

    ranges = window_token_ranges(
        slice_end - slice_start,
        max_ctx,
        config.doc_stride,
    )

    for index, (start, end) in enumerate(ranges):
        gs, ge = slice_start + start, slice_start + end

        ctx_ids = tokenized.token_ids[gs:ge]
        offsets = tokenized.offsets[gs:ge]

        input_ids = (
            [cls_token_id]
            + question_token_ids
            + [sep_token_id]
            + ctx_ids
            + [sep_token_id]
        )

        windows.append(
            WindowFeatures(
                input_ids=input_ids,
                question_token_len=len(question_token_ids),
                ctx_token_start=gs,
                ctx_token_end=ge,
                ctx_offsets=offsets,
                contract_id=tokenized.contract_id,
                clause_label=clause_label,
                window_index=first_window_index + index,
            )
        )

    return windows
