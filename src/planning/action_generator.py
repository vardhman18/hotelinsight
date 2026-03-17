"""
HotelInsight - Action Plan Generator
=======================================
Generates structured, costed action plans from inferred root causes.
Templates are read from ``src/config/action_templates.json``.
"""

import json
import os
from typing import Dict, List, Optional

from src.config.settings import ROOT_CAUSE_CONFIDENCE_THRESHOLD
from src.planning.cost_calculator import (
    calculate_total_costs,
    evaluate_cost_formula,
    format_currency,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "action_templates.json"
)


class ActionPlanGenerator:
    """Generate structured action plans from root causes.

    Attributes:
        templates: Loaded action template dictionary.
    """

    def __init__(self) -> None:
        """Load action templates from JSON on initialisation."""
        self.templates: Dict = self._load_templates()
        logger.info(
            "ActionPlanGenerator ready with %d root-cause templates.",
            len(self.templates),
        )

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_templates() -> Dict:
        """Read and return the action templates JSON file.

        Returns:
            Parsed JSON dictionary.

        Raises:
            FileNotFoundError: If the templates file cannot be located.
        """
        path = os.path.abspath(_TEMPLATES_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Action templates file not found: {path}"
            )
        with open(path, "r", encoding="utf-8") as fh:
            templates = json.load(fh)
        logger.debug("Loaded templates from: %s", path)
        return templates

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        hotel_name: str,
        topic: str,
        root_causes: List[Dict],
        num_rooms: int = 100,
    ) -> Dict:
        """Generate a complete, costed action plan.

        Iterates over root causes (sorted by confidence), retrieves matching
        templates, expands cost formulas, and assembles the plan into three
        timeframes.

        Args:
            hotel_name: Name of the hotel.
            topic: The complaint topic being addressed (e.g. ``"cleanliness"``).
            root_causes: List of dicts from
                :func:`~src.analysis.root_cause_analyzer.infer_root_causes`.
            num_rooms: Number of rooms in the hotel (used in cost formulas).

        Returns:
            Nested plan dictionary with keys:

            - ``hotel_name``
            - ``topic``
            - ``immediate_actions``
            - ``short_term_actions``
            - ``long_term_actions``
            - ``total_cost`` (dict with one_time, monthly, three_month_total)
        """
        # Filter to causes above the confidence threshold
        min_confidence = ROOT_CAUSE_CONFIDENCE_THRESHOLD * 100
        active_causes = [
            rc for rc in root_causes if rc.get("confidence", 0) >= min_confidence
        ]

        # If nothing clears the threshold, use top 3 causes anyway so the UI
        # always has something to show
        if not active_causes and root_causes:
            active_causes = root_causes[:3]

        immediate_actions: List[Dict] = []
        short_term_actions: List[Dict] = []
        long_term_actions: List[Dict] = []

        immediate_costs: List[float] = []
        short_term_costs: List[float] = []
        long_term_costs: List[float] = []

        for cause_info in active_causes:
            cause_key = cause_info.get("cause", "")
            confidence = cause_info.get("confidence", 0.0)

            template = self.templates.get(cause_key)
            if not template:
                logger.debug("No template found for cause: '%s'", cause_key)
                continue

            for timeframe, target_list, cost_list in (
                ("immediate", immediate_actions, immediate_costs),
                ("short_term", short_term_actions, short_term_costs),
                ("long_term", long_term_actions, long_term_costs),
            ):
                for action_tmpl in template.get(timeframe, []):
                    cost = evaluate_cost_formula(
                        action_tmpl.get("cost_formula", "0"), num_rooms
                    )
                    target_list.append(
                        {
                            "description": action_tmpl.get("action", ""),
                            "cost": format_currency(cost),
                            "cost_value": cost,
                            "timeline": action_tmpl.get("timeline", ""),
                            "expected_impact": action_tmpl.get("impact", ""),
                            "root_cause": cause_key,
                            "confidence": confidence,
                        }
                    )
                    cost_list.append(cost)

        total_cost = calculate_total_costs(
            immediate_costs, short_term_costs, long_term_costs
        )

        return {
            "hotel_name": hotel_name,
            "topic": topic,
            "immediate_actions": immediate_actions,
            "short_term_actions": short_term_actions,
            "long_term_actions": long_term_actions,
            "total_cost": total_cost,
        }
