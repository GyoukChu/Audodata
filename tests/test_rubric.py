from __future__ import annotations

import pytest

from autodata.cs.rubric import RubricError, RubricItem, parse_rubric, score_response


@pytest.mark.parametrize("weight", [8, "8", "+8", "  +8  "])
def test_integer_weights_and_inferred_category(weight):
    assert parse_rubric([{"criterion": "Explain why", "weight": weight}]) == [
        RubricItem("Explain why", 8, "positive")
    ]


def test_signed_negative_and_explicit_categories():
    assert parse_rubric([
        {"criterion": "Insight", "weight": "+8", "category": "positive"},
        {"criterion": "Error", "weight": "-3", "category": "negative"},
    ]) == [RubricItem("Insight", 8, "positive"), RubricItem("Error", -3, "negative")]


@pytest.mark.parametrize("obj", [
    None, {}, "rubric", [], [None], ["criterion"],
    [{"weight": 8}], [{"criterion": "", "weight": 8}],
    [{"criterion": "  ", "weight": 8}], [{"criterion": 4, "weight": 8}],
    [{"criterion": "x"}], [{"criterion": "x", "weight": 0}],
    [{"criterion": "x", "weight": "0"}], [{"criterion": "x", "weight": True}],
    [{"criterion": "x", "weight": 8.0}], [{"criterion": "x", "weight": "8.0"}],
    [{"criterion": "x", "weight": "1e2"}], [{"criterion": "x", "weight": None}],
    [{"criterion": "x", "weight": 8, "category": "negative"}],
    [{"criterion": "x", "weight": -3, "category": "positive"}],
    [{"criterion": "x", "weight": 8, "category": "unknown"}],
    [{"criterion": "x", "weight": 8, "category": None}],
    [{"criterion": "only penalties", "weight": -3}],
])
def test_rejects_malformed_rubrics(obj):
    with pytest.raises(RubricError):
        parse_rubric(obj)


@pytest.mark.parametrize("satisfied, expected", [
    ([True, False, True], (0.3, 8, 5, 10, 1, 1)),
    ([False, True, True], (0.0, 2, 5, 10, 1, 1)),
    ([True, True, False], (1.0, 10, 0, 10, 2, 0)),
    ([False, False, False], (0.0, 0, 0, 10, 0, 0)),
])
def test_clipped_scoring_and_breakdown(satisfied, expected):
    items = [RubricItem("a", 8, "positive"), RubricItem("b", 2, "positive"),
             RubricItem("bad", -5, "negative")]
    result = score_response(items, satisfied)
    assert (result.score, result.earned, result.penalty, result.max_positive,
            result.n_pos_satisfied, result.n_neg_triggered) == expected


@pytest.mark.parametrize("satisfied", [[], [True, False], [1], ["true"], [None]])
def test_scores_require_exact_boolean_judgments(satisfied):
    with pytest.raises(RubricError):
        score_response([RubricItem("a", 1, "positive")], satisfied)


def test_zero_denominator_is_not_a_score():
    with pytest.raises(RubricError):
        score_response([RubricItem("bad", -1, "negative")], [True])
