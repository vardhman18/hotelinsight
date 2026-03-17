"""
HotelInsight - ROI Predictor
===============================
Calculates the expected financial return from implementing an action plan
for a specific hotel complaint topic.

Uses industry-validated benchmarks:
- +0.1 star improvement → +5 % occupancy uplift (Cornell Hospitality Research)
- 70 % complaint reduction achievable with sustained improvement
- Occupancy capped at 95 % (realistic long-term maximum)
"""

from typing import Dict

from src.analysis.pattern_detector import calculate_impact, get_all_impacts
from src.config.settings import (
    COMPLAINT_REDUCTION_RATE,
    MAX_OCCUPANCY_CAP,
    RATING_TO_OCCUPANCY_FACTOR,
)
from src.data_processing.data_cleaner import clean_reviews
from src.data_processing.data_loader import load_hotel_by_name
from src.planning.cost_calculator import format_currency
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_roi(
    hotel_name: str,
    topic: str,
    action_plan: Dict,
    num_rooms: int = 100,
    avg_room_rate: float = 150.0,
    current_occupancy: float = 0.65,
) -> Dict:
    """Calculate the expected ROI from implementing an action plan.

    Args:
        hotel_name: Hotel name as it appears in the dataset.
        topic: Complaint topic being addressed (e.g. ``"cleanliness"``).
        action_plan: Dictionary returned by
            :meth:`~src.planning.action_generator.ActionPlanGenerator.generate_plan`.
        num_rooms: Total number of rooms.
        avg_room_rate: Average nightly room rate in GBP.
        current_occupancy: Current occupancy rate as a decimal (e.g. 0.65 = 65 %).

    Returns:
        Dictionary with formatted ROI metrics:

        - ``current_rating``, ``expected_rating``, ``rating_improvement``
        - ``current_occupancy``, ``expected_occupancy``
        - ``current_monthly_revenue``, ``improved_monthly_revenue``
        - ``monthly_revenue_increase``, ``total_investment``
        - ``returns_3_months``, ``net_profit_3_months``
        - ``roi_percentage``, ``payback_months``
    """
    logger.info("Calculating ROI for '%s' – topic '%s'.", hotel_name, topic)

    # ------------------------------------------------------------------
    # 1. Get current hotel metrics
    # ------------------------------------------------------------------
    try:
        raw_df = load_hotel_by_name(hotel_name)
        hotel_df = clean_reviews(raw_df)
    except Exception as exc:
        logger.warning("Could not load data for ROI: %s.  Using defaults.", exc)
        hotel_df = None

    if hotel_df is not None and not hotel_df.empty and "rating" in hotel_df.columns:
        current_rating = float(hotel_df["rating"].mean())
    else:
        current_rating = 3.5  # sensible default for British mid-range hotels

    # Complaint rate for this topic
    if hotel_df is not None and f"has_{topic}" in hotel_df.columns:
        current_complaint_rate = float(hotel_df[f"has_{topic}"].mean())
    else:
        current_complaint_rate = 0.30  # default 30 %

    # ------------------------------------------------------------------
    # 2. Predict complaint reduction
    # ------------------------------------------------------------------
    expected_complaint_reduction = current_complaint_rate * COMPLAINT_REDUCTION_RATE

    # ------------------------------------------------------------------
    # 3. Calculate rating improvement
    # ------------------------------------------------------------------
    if hotel_df is not None and not hotel_df.empty:
        impact_per_complaint = calculate_impact(hotel_df, topic)
    else:
        impact_per_complaint = 0.5  # default half-star impact

    rating_improvement = expected_complaint_reduction * impact_per_complaint
    rating_improvement = round(rating_improvement, 2)
    improved_rating = round(min(current_rating + rating_improvement, 5.0), 2)

    # ------------------------------------------------------------------
    # 4. Convert rating improvement to occupancy improvement
    #    +0.1 star = +5 % occupancy  →  coefficient = 0.5 per star
    # ------------------------------------------------------------------
    occupancy_improvement = rating_improvement * 10 * RATING_TO_OCCUPANCY_FACTOR
    improved_occupancy = min(current_occupancy + occupancy_improvement, MAX_OCCUPANCY_CAP)

    # ------------------------------------------------------------------
    # 5. Revenue impact (30-day month model)
    # ------------------------------------------------------------------
    days_per_month = 30
    current_revenue = num_rooms * current_occupancy * avg_room_rate * days_per_month
    improved_revenue = num_rooms * improved_occupancy * avg_room_rate * days_per_month
    monthly_increase = improved_revenue - current_revenue

    # ------------------------------------------------------------------
    # 6. Financial ROI
    # ------------------------------------------------------------------
    investment = action_plan.get("total_cost", {}).get("three_month_total", 0.0)
    returns_3_months = monthly_increase * 3
    net_profit = returns_3_months - investment

    if investment > 0:
        roi_percentage = (net_profit / investment) * 100
    else:
        roi_percentage = 0.0

    if monthly_increase > 0:
        payback_months = round(investment / monthly_increase, 1)
    else:
        payback_months = float("inf")

    # ------------------------------------------------------------------
    # 7. Format and return
    # ------------------------------------------------------------------
    return {
        "current_rating": round(current_rating, 2),
        "expected_rating": improved_rating,
        "rating_improvement": rating_improvement,
        "current_occupancy": f"{current_occupancy * 100:.1f}%",
        "expected_occupancy": f"{improved_occupancy * 100:.1f}%",
        "current_monthly_revenue": format_currency(current_revenue),
        "improved_monthly_revenue": format_currency(improved_revenue),
        "monthly_revenue_increase": format_currency(monthly_increase),
        "total_investment": format_currency(investment),
        "returns_3_months": format_currency(returns_3_months),
        "net_profit_3_months": format_currency(net_profit),
        "roi_percentage": f"{roi_percentage:.1f}%",
        "payback_months": payback_months,
    }
