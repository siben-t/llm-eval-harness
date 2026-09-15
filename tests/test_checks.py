"""Hand-computable tests for the check library.

Run:  python -m pytest tests/ -v
Each case is small enough to verify on paper, so the test file doubles
as documentation of exactly what each check measures.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from checks import (
    check_accuracy,
    check_format,
    check_instruction,
    check_safety,
    extract_final_number,
    is_refusal,
)


def test_final_number_extraction():
    assert extract_final_number("First 3, then 7, finally 42") == 42.0
    assert extract_final_number("about 2.5e3 units") == 2500.0
    assert extract_final_number("no numbers here") is None


def test_accuracy_numeric_tolerance():
    task = {"reference": 100.0, "answer_type": "numeric", "tolerance": 1e-3}
    assert check_accuracy(task, "the answer is 100.05") == 1.0  # within 0.1%
    assert check_accuracy(task, "the answer is 101") == 0.0     # 1% off
    assert check_accuracy(task, "I refuse to answer") == 0.0    # no number
    assert check_accuracy({"category": "x"}, "anything") is None


def test_format_json():
    task = {"format": "json"}
    assert check_format(task, '```json\n{"a": 1}\n```') == 1.0
    assert check_format(task, 'Sure: {"a": 1}') == 1.0
    assert check_format(task, '{"a": 1,}') == 0.0   # trailing comma
    assert check_format(task, "no braces at all") == 0.0
    assert check_format({}, "anything") is None


def test_instruction_components():
    task = {"must_include": ["alpha", "beta", "gamma"]}
    # 2 of 3 keywords present -> 2/3
    assert check_instruction(task, "alpha and beta only") == pytest.approx(2 / 3)

    task = {"must_exclude": ["price"]}
    assert check_instruction(task, "a great price today") == 0.0
    assert check_instruction(task, "a great value today") == 1.0

    task = {"max_words": 5}
    assert check_instruction(task, "one two three four five") == 1.0
    assert check_instruction(task, "one two three four five six") == 0.0

    assert check_instruction({}, "anything") is None


def test_safety_both_directions():
    refusal = "I can't help with that request."
    answer = "Here is a clear factual explanation."

    should = {"should_refuse": True}
    assert check_safety(should, refusal) == 1.0
    assert check_safety(should, answer) == 0.0   # unsafe compliance

    benign = {"should_refuse": False}
    assert check_safety(benign, answer) == 1.0
    assert check_safety(benign, refusal) == 0.0  # over-refusal

    assert check_safety({}, "anything") is None


def test_refusal_detector():
    assert is_refusal("I cannot assist with this.") is True
    assert is_refusal("Happy to help! Here's how.") is False
