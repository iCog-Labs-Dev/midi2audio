"""tests/test_groove_author.py — LLM groove author retry behaviour."""
from __future__ import annotations

import pytest

from m2a.groove.author import author_groove_spec
from tests.conftest import load_fixture

FIXTURE_STRUCTURE = load_fixture("eight_bar_structure.json")


def test_valid_llm_response_accepted():
    def mock_llm(sys, usr):
        return '{"swing": {"ratio": 0.58, "subdivision": 8, "applies_to": ["drums"]}}'

    spec = author_groove_spec(FIXTURE_STRUCTURE, "funk", mock_llm)
    assert spec.swing.ratio == 0.58


def test_invalid_json_retried():
    calls = []

    def mock_llm(sys, usr):
        calls.append(usr)
        if len(calls) < 2:
            return "not json"
        return '{"swing": null}'

    spec = author_groove_spec(FIXTURE_STRUCTURE, "funk", mock_llm, max_retries=3)
    assert len(calls) == 2
    assert spec.swing is None


def test_max_retries_raises():
    def bad_llm(sys, usr):
        return "always bad"

    with pytest.raises(RuntimeError):
        author_groove_spec(FIXTURE_STRUCTURE, "funk", bad_llm, max_retries=2)


def test_invalid_schema_retried_not_patched():
    calls = []

    def mock_llm(sys, usr):
        calls.append(usr)
        if len(calls) < 2:
            return '{"swing": {"ratio": 99.0}}'  # out of bounds -> ValidationError
        return '{}'

    spec = author_groove_spec(FIXTURE_STRUCTURE, "funk", mock_llm, max_retries=3)
    assert len(calls) == 2
    assert spec.swing is None
