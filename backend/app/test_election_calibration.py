"""
Tests for combined calibration (generic ballot + approval + special elections).
Run: pytest backend/app/test_election_calibration.py -v
"""
import pytest
from .election_forecast import (
    compute_combined_calibration_shift,
    APPROVAL_TO_GENERIC_SLOPE,
    CALIBRATION_WEIGHT_GENERIC_BALLOT,
    CALIBRATION_WEIGHT_APPROVAL,
    CALIBRATION_WEIGHT_SPECIAL_ELECTION,
)


def test_combined_calibration_all_signals():
    # GB 48% Dem -> D-2; approval net -10 -> D+0.12 (R president); special D+2
    # Default race_type (senate) -> gb impact 0.45, so effective_gb = -2 * 0.45 = -0.9
    # combined = 0.5*(-0.9) + 0.3*0.12 + 0.2*2 = -0.45 + 0.036 + 0.4 = -0.014
    shift, meta = compute_combined_calibration_shift(
        generic_ballot_dem_share=48.0,
        approval_net=-10.0,
        special_election_swing_pts=2.0,
        calibration_weight=1.0,
        president_party="R",
    )
    assert "components" in meta
    assert meta["components"].get("generic_ballot_shift") == -2.0
    assert meta["components"].get("generic_ballot_impact_coefficient") == 0.45
    assert meta["components"].get("approval_shift") == round(-APPROVAL_TO_GENERIC_SLOPE * -10, 2)
    assert meta["components"].get("special_election_swing_pts") == 2.0
    assert meta["weights_used"]["generic_ballot"] == 0.5
    assert meta["weights_used"]["approval"] == 0.3
    assert meta["weights_used"]["special_election"] == 0.2
    effective_gb = -2 * 0.45
    expected_combined = 0.5 * effective_gb + 0.3 * (APPROVAL_TO_GENERIC_SLOPE * 10) + 0.2 * 2
    assert abs(meta["combined_shift"] - round(expected_combined, 2)) < 0.02
    assert abs(shift - round(expected_combined, 2)) < 0.02


def test_combined_calibration_slider_scales():
    shift_full, _ = compute_combined_calibration_shift(
        generic_ballot_dem_share=52.0,  # D+2
        calibration_weight=1.0,
    )
    shift_half, _ = compute_combined_calibration_shift(
        generic_ballot_dem_share=52.0,
        calibration_weight=0.5,
    )
    assert abs(shift_half - shift_full * 0.5) < 0.02


def test_combined_calibration_missing_signals_renormalize():
    # Only generic ballot: 52% -> D+2; default race_type (senate) -> coefficient 0.45, so effective = 2 * 0.45 = 0.9
    shift, meta = compute_combined_calibration_shift(
        generic_ballot_dem_share=52.0,
        approval_net=None,
        special_election_swing_pts=None,
        calibration_weight=1.0,
    )
    assert meta["weights_used"]["generic_ballot"] == 1.0
    assert meta["weights_used"]["approval"] == 0.0
    assert meta["weights_used"]["special_election"] == 0.0
    assert abs(shift - 0.9) < 0.02


def test_combined_calibration_no_data_returns_zero():
    shift, meta = compute_combined_calibration_shift(
        generic_ballot_dem_share=None,
        approval_net=None,
        special_election_swing_pts=None,
    )
    assert shift == 0.0
    assert meta["combined_shift"] == 0.0


def test_approval_impact_by_race_type():
    # Approval only: net -10 (R president) -> base shift toward D = 0.012*10 = 0.12
    # House 1.2x -> 0.144; Senate 1.0x -> 0.12; Governor 0.8x -> 0.096
    base = APPROVAL_TO_GENERIC_SLOPE * 10
    shift_house, meta_house = compute_combined_calibration_shift(
        generic_ballot_dem_share=None,
        approval_net=-10.0,
        special_election_swing_pts=None,
        calibration_weight=1.0,
        president_party="R",
        race_type="house",
    )
    assert meta_house["components"].get("approval_impact_coefficient") == 1.2
    assert abs(shift_house - base * 1.2) < 0.02
    shift_gov, meta_gov = compute_combined_calibration_shift(
        generic_ballot_dem_share=None,
        approval_net=-10.0,
        special_election_swing_pts=None,
        calibration_weight=1.0,
        president_party="R",
        race_type="governor",
    )
    assert meta_gov["components"].get("approval_impact_coefficient") == 0.8
    assert abs(shift_gov - base * 0.8) < 0.02
