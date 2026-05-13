"""Tests for agent JSON parsing and the memory store.

We do NOT hit the live Claude API in tests. Instead, we test the parsing
robustness of `extract_first_json` and the SQLite memory store directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_quant_lab.agents.base import extract_first_json
from ai_quant_lab.agents.hypothesis import _payload_to_hypothesis
from ai_quant_lab.agents.memory import ResearchMemory, TrialRecord


def test_extract_json_from_plain_object():
    payload = extract_first_json('{"a": 1, "b": [1, 2]}')
    assert payload == {"a": 1, "b": [1, 2]}


def test_extract_json_from_fenced_markdown():
    text = 'Here is the result:\n```json\n{"verdict": "pass"}\n```\nOK.'
    payload = extract_first_json(text)
    assert payload == {"verdict": "pass"}


def test_extract_json_from_unfenced_text():
    text = "I think the strategy is:\n{\"hypothesis_id\": \"x\", \"value\": 3.14}\nEnd."
    payload = extract_first_json(text)
    assert payload["hypothesis_id"] == "x"
    assert payload["value"] == pytest.approx(3.14)


def test_extract_json_failure_raises():
    with pytest.raises(ValueError):
        extract_first_json("no json here, just words")


def test_hypothesis_payload_parsing():
    payload = {
        "hypothesis_id": "mom_21",
        "title": "Vanilla momentum",
        "rationale": "Established factor.",
        "spec": {"signal": "21d return", "direction": "long"},
        "expected_sharpe_range": [0.3, 0.8],
        "works_in_regime": "trending",
        "breaks_in_regime": "mean-reverting",
    }
    h = _payload_to_hypothesis(payload)
    assert h.hypothesis_id == "mom_21"
    assert h.expected_sharpe_range == (0.3, 0.8)


def test_memory_record_and_count(tmp_path: Path):
    db = tmp_path / "memory.db"
    with ResearchMemory(db) as memory:
        assert memory.n_trials() == 0
        memory.record(
            TrialRecord(
                hypothesis_id="h1",
                hypothesis_text="title",
                rationale="r",
                code="pass",
                metrics={"sharpe_ratio": 1.0},
                accepted=True,
                n_trials_at_time=0,
                iteration=0,
            )
        )
        assert memory.n_trials() == 1
        survivors = memory.survivors()
        assert len(survivors) == 1
        assert survivors[0].accepted is True


def test_memory_summary_for_prompt_handles_empty(tmp_path: Path):
    db = tmp_path / "memory.db"
    with ResearchMemory(db) as memory:
        assert "no prior trials" in memory.summarize_for_prompt()
