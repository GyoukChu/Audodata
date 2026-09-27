"""Weighted binary rubric scoring (paper ambiguity A4: PRBench clipped score)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


class RubricError(ValueError):
    """The rubric or judgments cannot define a valid weighted score."""


@dataclass(frozen=True)
class RubricItem:
    criterion: str
    weight: int
    category: Literal["positive", "negative"]


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    earned: int
    penalty: int
    max_positive: int
    n_pos_satisfied: int
    n_neg_triggered: int


def parse_rubric(obj: Any) -> list[RubricItem]:
    if not isinstance(obj, list) or not obj:
        raise RubricError("rubric must be a non-empty list")
    items: list[RubricItem] = []
    for index, item in enumerate(obj, 1):
        if not isinstance(item, dict):
            raise RubricError(f"criterion {index} must be an object")
        criterion = item.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            raise RubricError(f"criterion {index} requires non-empty criterion text")
        weight = item.get("weight")
        if isinstance(weight, str) and re.fullmatch(r"[+-]?[0-9]+", weight.strip()):
            weight = int(weight)
        if type(weight) is not int or weight == 0:
            raise RubricError(f"criterion {index} requires a non-zero integer weight")
        category = "positive" if weight > 0 else "negative"
        if item.get("category", category) != category:
            raise RubricError(f"criterion {index}: category must match the weight sign ({category})")
        items.append(RubricItem(criterion, weight, category))
    if not any(item.weight > 0 for item in items):
        raise RubricError("rubric needs a positive weight to define the score denominator")
    return items


def score_response(items: list[RubricItem], satisfied: list[bool]) -> ScoreBreakdown:
    if len(satisfied) != len(items) or any(type(value) is not bool for value in satisfied):
        raise RubricError("satisfied must contain exactly one boolean per criterion")
    max_positive = sum(item.weight for item in items if item.weight > 0)
    if max_positive <= 0:
        raise RubricError("rubric needs a positive weight to define the score denominator")
    positive = [item for item, yes in zip(items, satisfied) if yes and item.weight > 0]
    negative = [item for item, yes in zip(items, satisfied) if yes and item.weight < 0]
    earned = sum(item.weight for item in positive)
    penalty = sum(abs(item.weight) for item in negative)
    return ScoreBreakdown(
        score=max(0.0, min(1.0, (earned - penalty) / max_positive)),
        earned=earned, penalty=penalty, max_positive=max_positive,
        n_pos_satisfied=len(positive), n_neg_triggered=len(negative),
    )
