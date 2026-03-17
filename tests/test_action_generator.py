"""
Tests for ActionPlanGenerator and cost formula safety.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.planning.action_generator import ActionPlanGenerator
from src.planning.cost_calculator import evaluate_cost_formula, format_currency


# ───────────────────────────── Cost calculator ───────────────────────


def test_evaluate_cost_formula_simple():
    result = evaluate_cost_formula("rooms * 30", 50)
    assert result == 1500.0


def test_evaluate_cost_formula_training_base():
    result = evaluate_cost_formula("TRAINING_COST_BASE * 2", 100)
    assert result == 2400.0


def test_evaluate_cost_formula_equipment():
    result = evaluate_cost_formula("EQUIPMENT_COST_PER_ROOM * rooms", 10)
    assert result == 1000.0


def test_evaluate_cost_formula_combined():
    result = evaluate_cost_formula("TRAINING_COST_BASE + STAFF_COST_PER_ROOM * rooms", 100)
    assert result == 1200.0 + 30.0 * 100


def test_evaluate_cost_formula_rejects_import():
    with pytest.raises(Exception):
        evaluate_cost_formula("__import__('os').system('dir')", 10)


def test_evaluate_cost_formula_rejects_exec():
    with pytest.raises(Exception):
        evaluate_cost_formula("exec('import os')", 10)


def test_evaluate_cost_formula_returns_float():
    result = evaluate_cost_formula("1200", 50)
    assert isinstance(result, float)


def test_format_currency_thousands():
    assert format_currency(1500) == "£1,500"


def test_format_currency_small():
    assert format_currency(250) == "£250"


def test_format_currency_large():
    result = format_currency(100000)
    assert "100,000" in result


# ───────────────────────────── ActionPlanGenerator ───────────────────


@pytest.fixture(scope="module")
def generator():
    return ActionPlanGenerator()


def _sample_root_causes():
    return [
        {"cause": "UNDERSTAFFING",   "confidence": 75.0, "description": "Not enough staff", "evidence": []},
        {"cause": "TRAINING_ISSUES", "confidence": 55.0, "description": "Staff need training", "evidence": []},
    ]


def test_generate_plan_returns_dict(generator):
    result = generator.generate_plan("TestHotel", "cleanliness", _sample_root_causes())
    assert isinstance(result, dict)


def test_generate_plan_has_timeframe_keys(generator):
    result = generator.generate_plan("TestHotel", "cleanliness", _sample_root_causes())
    assert "immediate_actions" in result
    assert "short_term_actions" in result
    assert "long_term_actions" in result


def test_generate_plan_has_total_cost(generator):
    result = generator.generate_plan("TestHotel", "cleanliness", _sample_root_causes())
    assert "total_cost" in result
    tc = result["total_cost"]
    assert "one_time" in tc
    assert "monthly" in tc
    assert "three_month_total" in tc


def test_generate_plan_cost_is_non_negative(generator):
    result = generator.generate_plan("TestHotel", "staff", _sample_root_causes(), num_rooms=80)
    tc = result["total_cost"]
    assert tc["one_time"] >= 0
    assert tc["monthly"] >= 0
    assert tc["three_month_total"] >= 0


def test_generate_plan_room_count_affects_cost(generator):
    plan_small = generator.generate_plan("TestHotel", "cleanliness", _sample_root_causes(), num_rooms=20)
    plan_large = generator.generate_plan("TestHotel", "cleanliness", _sample_root_causes(), num_rooms=200)
    # Larger hotel should have higher (or equal) cost
    assert plan_large["total_cost"]["three_month_total"] >= plan_small["total_cost"]["three_month_total"]


def test_each_action_has_required_fields(generator):
    result = generator.generate_plan("TestHotel", "cleanliness", _sample_root_causes())
    for bucket in ("immediate_actions", "short_term_actions", "long_term_actions"):
        for action in result[bucket]:
            assert "description" in action
            assert "cost" in action
            assert "timeline" in action
            assert "expected_impact" in action


def test_generate_plan_empty_root_causes_still_works(generator):
    result = generator.generate_plan("TestHotel", "noise", [], num_rooms=50)
    assert isinstance(result, dict)
