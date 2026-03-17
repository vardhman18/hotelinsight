"""
HotelInsight - Cost Calculator
=================================
Safe evaluation and formatting of cost formulas stored in
``action_templates.json``.

Each template action contains a ``cost_formula`` field which is a
Python expression that may reference:
- ``rooms``: integer number of hotel rooms
- ``TRAINING_COST_BASE``: from settings (£1,200)
- ``EQUIPMENT_COST_PER_ROOM``: from settings (£100/room)

All formulas are validated before evaluation.  Only a restricted namespace
is exposed to ``eval`` so arbitrary code execution is prevented.
"""

import re
from typing import Union

from src.config.settings import EQUIPMENT_COST_PER_ROOM, STAFF_COST_PER_ROOM, TRAINING_COST_BASE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Allowed tokens in a cost formula
# Security: forbid any string not matching this pattern from being eval'd.
# ---------------------------------------------------------------------------
_ALLOWED_FORMULA_PATTERN = re.compile(
    r"^[\d\s\+\-\*\/\(\)\.\,rooms_TRAININGCOSTBASE_EQUIPMENTPEROM]+$"
)

# Safe namespace exposed to eval
_SAFE_BUILTINS: dict = {}  # no builtins at all

def _safe_namespace(rooms: int) -> dict:
    """Return the restricted namespace dictionary for formula evaluation.

    Args:
        rooms: Number of hotel rooms.

    Returns:
        Namespace dict with allowed variable names only.
    """
    return {
        "__builtins__": {},   # block all builtins
        "rooms": int(rooms),
        "TRAINING_COST_BASE": TRAINING_COST_BASE,
        "EQUIPMENT_COST_PER_ROOM": EQUIPMENT_COST_PER_ROOM,
        "STAFF_COST_PER_ROOM": STAFF_COST_PER_ROOM,
    }


def evaluate_cost_formula(formula: str, rooms: int) -> float:
    """Safely evaluate a cost formula string.

    Validates that the formula contains only arithmetic operators, numeric
    literals, and the whitelisted variable names before calling ``eval``.

    Args:
        formula: Python expression string from the action template.
        rooms: Number of hotel rooms (substituted for the ``rooms`` variable).

    Returns:
        Computed cost as a float.  Returns ``0.0`` if the formula is empty
        or evaluation fails.

    Raises:
        ValueError: If the formula contains disallowed characters (potential
            injection attempt).
    """
    if not formula or not formula.strip():
        return 0.0

    formula = formula.strip()

    # Whitelist check – only allow safe characters and known variable names
    sanitised = (
        formula
        .replace("TRAINING_COST_BASE", "T")
        .replace("EQUIPMENT_COST_PER_ROOM", "E")
        .replace("STAFF_COST_PER_ROOM", "S")
        .replace("rooms", "r")
    )

    if not re.match(r"^[\d\s\+\-\*\/\(\)\.rTES]+$", sanitised):
        raise ValueError(
            f"Cost formula contains disallowed characters: '{formula}'. "
            "Only arithmetic expressions with 'rooms', 'TRAINING_COST_BASE', "
            "'EQUIPMENT_COST_PER_ROOM', and 'STAFF_COST_PER_ROOM' are permitted."
        )

    try:
        result = eval(formula, {"__builtins__": {}}, _safe_namespace(rooms))  # noqa: S307
        return float(result)
    except Exception as exc:
        logger.warning("Cost formula evaluation failed '%s': %s", formula, exc)
        return 0.0


def format_currency(amount: float, currency: str = "£") -> str:
    """Format a numeric amount as a currency string.

    Args:
        amount: Monetary value.
        currency: Currency symbol (default ``"£"`` for GBP).

    Returns:
        Formatted string like ``"£1,250"``.
    """
    if amount == 0:
        return f"{currency}0"
    return f"{currency}{amount:,.0f}"


def calculate_total_costs(
    immediate_costs: list,
    short_term_costs: list,
    long_term_costs: list,
) -> dict:
    """Aggregate one-time and monthly costs across action timeframes.

    Immediate and short-term costs are treated as one-time investments;
    long-term costs are treated as recurring monthly overheads.

    Args:
        immediate_costs: List of floats from immediate actions.
        short_term_costs: List of floats from short-term actions.
        long_term_costs: List of floats from long-term actions.

    Returns:
        Dictionary with keys:

        - ``one_time`` (float): Sum of immediate + short-term costs.
        - ``monthly`` (float): Sum of long-term (recurring) costs.
        - ``three_month_total`` (float): one_time + monthly × 3
    """
    one_time = sum(immediate_costs) + sum(short_term_costs)
    monthly = sum(long_term_costs)
    three_month_total = one_time + monthly * 3

    return {
        "one_time": round(one_time, 2),
        "monthly": round(monthly, 2),
        "three_month_total": round(three_month_total, 2),
    }
