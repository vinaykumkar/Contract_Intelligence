"""Load CUAD JSON data into validated contract and QA records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    TEST_PATH,
    TRAIN_SEPARATE_PATH,
    CUADV1_PATH,
    ClauseSpec,
    ConfigError,
)

CATEGORY_RE = re.compile(r'related to "(.+?)" that should be reviewed')


class CuadDataError(RuntimeError):
    """Raised when CUAD data violates a required structure."""


@dataclass(frozen=True)
class AnswerSpan:
    """Gold answer span using character offsets in the contract."""
    text: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass
class ClauseQA:
    """One contract/clause QA instance."""
    clause_label: str
    cuad_category: str
    question: str
    answers: list[AnswerSpan] = field(default_factory=list)
    is_impossible: bool = False
    qa_id: str = ""


@dataclass
class ContractRecord:
    """One contract containing its QA instances."""
    contract_id: str
    context: str
    qas: list[ClauseQA] = field(default_factory=list)
    split: str = ""

    def qas_for(self, clause_label: str) -> list[ClauseQA]:
        return [qa for qa in self.qas if qa.clause_label == clause_label]


def validate_answer(
    context: str,
    answer: AnswerSpan,
    contract_id: str,
    qa_id: str,
) -> None:
    """Ensure the recorded answer exactly matches the context."""
    if answer.start < 0:
        raise CuadDataError(
            f"Negative answer_start in contract '{contract_id}' qa '{qa_id}'"
        )

    if answer.end > len(context):
        raise CuadDataError(
            f"Answer out of bounds in contract '{contract_id}' qa '{qa_id}': "
            f"[{answer.start}:{answer.end}]"
        )

    if context[answer.start:answer.end] != answer.text:
        raise CuadDataError(
            f"Answer mismatch in contract '{contract_id}' qa '{qa_id}'"
        )


def _parse_qa(
    qa: dict,
    category: str,
    question: str,
    labels: dict[str, str],
    contract_id: str,
    context: str,
    strict: bool,
) -> ClauseQA:
    qa_id = qa.get("id", "")
    impossible = bool(qa.get("is_impossible", False))
    raw_answers = qa.get("answers") or []

    if impossible and raw_answers:
        raise CuadDataError(
            f"is_impossible=True but answers exist in "
            f"contract '{contract_id}' qa '{qa_id}'"
        )

    answers = [
        AnswerSpan(a["text"], int(a["answer_start"]))
        for a in raw_answers
    ]

    if strict:
        for answer in answers:
            validate_answer(context, answer, contract_id, qa_id)

    return ClauseQA(
        clause_label=labels.get(category, ""),
        cuad_category=category,
        question=question,
        answers=answers,
        is_impossible=impossible,
        qa_id=qa_id,
    )


def load_contracts(
    path: Path,
    clauses: list[ClauseSpec],
    *,
    strict: bool = True,
    limit: int | None = None,
    keep_unenabled: bool = False,
) -> list[ContractRecord]:
    """Load and validate CUAD contracts from a SQuAD-style JSON file."""

    enabled = {c.cuad_category: c for c in clauses if c.enabled}
    labels = {c.cuad_category: c.label for c in clauses if c.enabled}
    questions = {c.cuad_category: c.question for c in clauses if c.enabled}

    if not path.exists():
        raise ConfigError(
            f"CUAD file not found: {path}. "
            "Extract data.zip into data/raw/ first."
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records = []

    for item in payload["data"]:
        if limit is not None and len(records) >= limit:
            break

        title = item.get("title", "")
        if not title:
            raise CuadDataError("Contract with empty title found.")

        for paragraph in item.get("paragraphs", []):
            context = paragraph.get("context", "")

            if not context.strip():
                raise CuadDataError(f"Empty context in contract '{title}'.")

            qas = []

            for raw_qa in paragraph.get("qas", []):
                match = CATEGORY_RE.search(raw_qa.get("question", ""))

                if not match:
                    raise CuadDataError(
                        f"Cannot parse category in contract '{title}' "
                        f"qa '{raw_qa.get('id', '')}'"
                    )

                category = match.group(1)

                if category not in enabled and not keep_unenabled:
                    continue

                qas.append(
                    _parse_qa(
                        raw_qa,
                        category,
                        questions.get(
                            category,
                            raw_qa.get("question", ""),
                        ),
                        labels,
                        title,
                        context,
                        strict,
                    )
                )

            records.append(
                ContractRecord(
                    contract_id=title,
                    context=context,
                    qas=qas,
                )
            )

    ids = [record.contract_id for record in records]
    duplicates = {x for x in ids if ids.count(x) > 1}

    if duplicates:
        raise CuadDataError(
            f"Duplicate contract titles in {path.name}: "
            f"{sorted(duplicates)[:5]}"
        )

    return records


def load_train_contracts(
    clauses: list[ClauseSpec], **kwargs
) -> list[ContractRecord]:
    """Load the official training dataset."""
    return load_contracts(TRAIN_SEPARATE_PATH, clauses, **kwargs)


def load_test_contracts(
    clauses: list[ClauseSpec], **kwargs
) -> list[ContractRecord]:
    """Load the untouched official test dataset."""
    return load_contracts(TEST_PATH, clauses, **kwargs)


def load_merged_contracts(
    clauses: list[ClauseSpec], **kwargs
) -> list[ContractRecord]:
    """Load the merged CUAD dataset for audits and sample extraction."""
    return load_contracts(CUADV1_PATH, clauses, **kwargs)
