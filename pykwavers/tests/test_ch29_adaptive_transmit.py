from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_adaptive_transmit_budget_values_use_case_element_count(monkeypatch):
    import ch29_adaptive_transmit as adaptive

    monkeypatch.delenv("KWAVERS_CH29_ADAPTIVE_TRANSMIT_BUDGETS", raising=False)
    monkeypatch.setenv("KWAVERS_CH29_ADAPTIVE_TRANSMIT_FRACTIONS", "0.25,0.5")

    assert adaptive.transmit_budget_values({"elements": 64}) == [16, 32]

    monkeypatch.setenv("KWAVERS_CH29_ADAPTIVE_TRANSMIT_BUDGETS", "4,16,128")
    assert adaptive.transmit_budget_values({"elements": 32}) == [4, 16, 32]


def test_adaptive_transmit_records_compare_matched_uniform_and_patient_adaptive(monkeypatch):
    import ch29_adaptive_transmit as adaptive

    monkeypatch.setenv("KWAVERS_CH29_ADAPTIVE_TRANSMIT_BUDGETS", "4")
    monkeypatch.setenv("KWAVERS_CH29_ADAPTIVE_INCLUDE_FULL", "0")

    def runner(*args, **kwargs):
        budget = int(kwargs["transmit_budget"])
        strategy = str(kwargs["transmit_schedule_strategy"])
        element_count = int(kwargs["element_count"])
        base = budget / element_count
        bonus = 0.05 if strategy == "patient_adaptive" else 0.0
        return {
            "anatomy": kwargs["anatomy"],
            "element_count": element_count,
            "encoded_measurements": 10 * budget,
            "unencoded_measurements": 20 * budget,
            "transmit_schedule_strategy": strategy,
            "transmit_budget_effective": budget,
            "transmit_budget_fraction": budget / element_count,
            "transmit_sequence_indices": np.arange(budget),
            "metrics": {
                "active_lesion": {"dice_equal_area": base + bonus, "cnr": 2.0 + bonus},
                "fusion": {"dice_equal_area": base + bonus, "cnr": 3.0 + bonus},
            },
        }

    cases = [
        {
            "name": "kidney",
            "ct": "synthetic.nii.gz",
            "seg": None,
            "grid": 24,
            "elements": 16,
            "freq": [500_000.0],
            "offsets": [4],
            "pressure": 1.0,
        }
    ]
    records = adaptive.collect_adaptive_transmit_records(cases, runner, lambda case: {})
    payload = adaptive.adaptive_transmit_payload(records, None)

    assert [record["strategy"] for record in records] == ["uniform", "patient_adaptive"]
    assert records[0]["effective_budget"] == 4
    assert payload["control_surface"]["minimal_parameters"] == [
        "transmit_schedule_strategy",
        "transmit_budget",
    ]
    assert payload["summary"]["mean_patient_adaptive_minus_uniform_active_dice"] > 0.0
