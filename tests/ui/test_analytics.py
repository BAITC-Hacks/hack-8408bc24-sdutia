"""Presentation safeguards: missing hours and ambiguous revisions stay visible."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.analytics import clean_submission, daily_energy, revision_pairs  # noqa: E402


def local_day() -> pd.DataFrame:
    return pd.DataFrame({
        "target_time_utc": pd.date_range("2026-02-01T18:00:00Z", periods=24, freq="h"),
        "farm_mean": [0.5] * 24,
        "lead_h": range(24, 48),
    })


def revision(version: int, mean: float, issue_id: str = "2026-02-01_0000") -> dict:
    return {"site": "shelek", "issue_id": issue_id, "issue_time_utc": "2026-01-31T18:00:00Z",
            "target_time_utc": "2026-02-01T18:00:00Z", "lead_h": 24, "product": "day_ahead",
            "entity": "farm", "version": version, "mean": mean}


def test_daily_energy_uses_chosen_fixed_clock():
    energy, partial = daily_energy(local_day(), clock_offset=6)
    assert energy.to_dict("records") == [{"day": "2026-02-02", "energy_norm_h": 12.0}]
    assert partial == 0

    # The same 24 samples cover portions of two official UTC+5 calendar days.
    energy, partial = daily_energy(local_day(), clock_offset=5)
    assert energy.empty
    assert partial == 2


def test_missing_hour_does_not_look_like_complete_day_energy():
    energy, partial = daily_energy(local_day().drop(index=7), clock_offset=6)
    assert energy.empty
    assert partial == 1


def test_duplicate_submission_hours_are_all_excluded_without_changing_input():
    original = local_day()
    duplicate = pd.concat([original, original.iloc[[0]]], ignore_index=True)
    before = duplicate.copy(deep=True)

    cleaned, rejected = clean_submission(duplicate)

    assert rejected == 2
    assert len(cleaned) == 23
    pd.testing.assert_frame_equal(duplicate, before)
    energy, partial = daily_energy(duplicate, clock_offset=6)
    assert energy.empty
    assert partial == 1


def test_revision_delta_pairs_same_issue_and_entity():
    rows = [revision(1, 0.4), revision(2, 0.55), {**revision(2, 0.9), "entity": "t1"}]
    pairs, rejected = revision_pairs(pd.DataFrame(rows))

    assert len(pairs) == 1
    assert pairs.iloc[0]["delta"] == pytest.approx(0.15)
    assert rejected == 0


def test_revision_does_not_pair_different_issues():
    pairs, rejected = revision_pairs(pd.DataFrame([
        revision(1, 0.4, "2026-02-01_0000"), revision(2, 0.6, "2026-02-02_0000"),
    ]))
    assert pairs.empty
    assert rejected == 2


def test_duplicate_revision_is_not_arbitrarily_selected():
    pairs, rejected = revision_pairs(pd.DataFrame([revision(1, 0.4), revision(2, 0.5), revision(2, 0.6)]))
    assert pairs.empty
    assert rejected == 3


def test_two_issues_for_same_heatmap_cell_are_not_silently_averaged():
    pairs, rejected = revision_pairs(pd.DataFrame([
        revision(1, 0.4, "issue_a"), revision(2, 0.5, "issue_a"),
        revision(1, 0.2, "issue_b"), revision(2, 0.8, "issue_b"),
    ]))
    assert pairs.empty
    assert rejected == 4
