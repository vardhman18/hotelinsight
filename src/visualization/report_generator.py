"""
HotelInsight - Report Generator
==================================
Exports analysis results to Excel (.xlsx) and generates basic formatted
reports.  PDF export is available when ``reportlab`` or ``weasyprint`` is
installed; otherwise falls back to Excel only.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.config.settings import RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_analysis_excel(
    hotel_name: str,
    analysis: Dict,
    impact_table: pd.DataFrame,
    action_plan: Optional[Dict] = None,
    roi_data: Optional[Dict] = None,
) -> str:
    """Export a full hotel analysis report to an Excel workbook.

    Creates one worksheet per section:
    - **Summary**: Overall metrics
    - **Topic Stats**: Per-topic complaint frequencies and counts
    - **Impact Table**: Rating impact per category
    - **Action Plan**: Recommended actions by timeframe (if provided)
    - **ROI**: Financial projections (if provided)

    Args:
        hotel_name: Hotel name (used in the filename).
        analysis: Output of ``analyze_hotel()``.
        impact_table: DataFrame from ``build_impact_table()``.
        action_plan: Optional plan from ``ActionPlanGenerator.generate_plan()``.
        roi_data: Optional dict from ``calculate_roi()``.

    Returns:
        Absolute path to the generated ``.xlsx`` file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in hotel_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.xlsx"
    filepath = os.path.join(RESULTS_DIR, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Sheet 1 – Summary
        summary_data = {
            "Metric": [
                "Hotel Name",
                "Total Reviews",
                "Average Rating",
                "Average Sentiment",
            ],
            "Value": [
                analysis.get("hotel_name", hotel_name),
                analysis.get("total_reviews", 0),
                analysis.get("avg_rating", 0.0),
                analysis.get("avg_sentiment", 0.0),
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2 – Topic Stats
        topic_stats = analysis.get("topic_stats", {})
        if topic_stats:
            topic_rows = [
                {"Topic": k, "Count": v.get("count", 0), "Percentage (%)": v.get("percentage", 0)}
                for k, v in topic_stats.items()
            ]
            pd.DataFrame(topic_rows).to_excel(
                writer, sheet_name="Topic Stats", index=False
            )

        # Sheet 3 – Impact Table
        if not impact_table.empty:
            impact_table.to_excel(writer, sheet_name="Impact Table", index=False)

        # Sheet 4 – Action Plan
        if action_plan:
            actions_rows = []
            for timeframe in ("immediate_actions", "short_term_actions", "long_term_actions"):
                label = timeframe.replace("_actions", "").replace("_", " ").title()
                for action in action_plan.get(timeframe, []):
                    actions_rows.append(
                        {
                            "Timeframe": label,
                            "Action": action.get("description", ""),
                            "Cost": action.get("cost", ""),
                            "Timeline": action.get("timeline", ""),
                            "Expected Impact": action.get("expected_impact", ""),
                            "Root Cause": action.get("root_cause", ""),
                        }
                    )
            if actions_rows:
                pd.DataFrame(actions_rows).to_excel(
                    writer, sheet_name="Action Plan", index=False
                )

        # Sheet 5 – ROI
        if roi_data:
            roi_rows = [
                {"Metric": k.replace("_", " ").title(), "Value": v}
                for k, v in roi_data.items()
            ]
            pd.DataFrame(roi_rows).to_excel(writer, sheet_name="ROI", index=False)

    logger.info("Report exported to: %s", filepath)
    return filepath
