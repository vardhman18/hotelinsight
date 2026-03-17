#!/usr/bin/env python
"""
scripts/export_results.py
===========================
Batch-analyses one or more hotels and exports full Excel reports to
data/results/.

Usage:
    python scripts/export_results.py [--hotel "Hotel Name"] [--all] [--top N]

Flags:
    --hotel  "Hotel Name"  Export results for a single named hotel.
    --all                  Export for every hotel in the dataset (slow).
    --top N                Export for the N hotels with the most reviews (default: 10).
"""

import argparse
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def export_hotel(hotel_name: str) -> str | None:
    """Run analysis and export Excel for one hotel. Returns path or None on error."""
    try:
        from src.data_processing.data_loader import load_hotel_by_name
        from src.data_processing.data_cleaner import clean_reviews
        from src.analysis.pattern_detector import _ensure_topic_columns, analyze_hotel
        from src.analysis.impact_calculator import build_impact_table
        from src.analysis.root_cause_analyzer import infer_root_causes
        from src.planning.action_generator import ActionPlanGenerator
        from src.visualization.report_generator import export_analysis_excel
        from src.config.settings import COMPLAINT_CATEGORIES

        raw      = load_hotel_by_name(hotel_name)
        cleaned  = clean_reviews(raw)
        hotel_df = _ensure_topic_columns(cleaned)
        analysis = analyze_hotel(hotel_name)
        impact_tbl = build_impact_table(hotel_df)

        # Build action plan for top topic
        topic_stats = analysis.get("topic_stats", {})
        if topic_stats:
            top_topic = max(
                topic_stats,
                key=lambda t: topic_stats[t].get("complaint_rate", 0),
            )
        else:
            top_topic = COMPLAINT_CATEGORIES[0]

        root_causes = infer_root_causes(hotel_df, top_topic)
        generator   = ActionPlanGenerator()
        action_plan = generator.generate_plan(hotel_name, top_topic, root_causes)

        path = export_analysis_excel(hotel_name, analysis, impact_tbl, action_plan)
        return path
    except Exception as exc:
        print(f"  [ERROR] {hotel_name}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export HotelInsight Excel reports")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--hotel", type=str, help="Single hotel name")
    grp.add_argument("--all",   action="store_true", help="All hotels")
    grp.add_argument("--top",   type=int, default=10, help="Top N hotels by review count")
    args = parser.parse_args()

    from src.data_processing.data_loader import get_hotel_list
    from src.data_processing.data_loader import load_hotel_reviews
    from src.data_processing.data_cleaner import clean_reviews

    if args.hotel:
        hotels = [args.hotel]
    elif args.all:
        hotels = get_hotel_list()
    else:
        df = load_hotel_reviews()
        df = clean_reviews(df)
        top_hotels = (
            df.groupby("Hotel_Name").size().sort_values(ascending=False).head(args.top).index.tolist()
        )
        hotels = top_hotels

    print(f"Exporting reports for {len(hotels)} hotel(s)…")
    t0      = time.time()
    success = 0
    for i, hotel in enumerate(hotels, 1):
        print(f"  [{i}/{len(hotels)}] {hotel} … ", end="", flush=True)
        path = export_hotel(hotel)
        if path:
            print(f"saved → {path}")
            success += 1
        else:
            print("FAILED")

    elapsed = time.time() - t0
    print(f"\nDone: {success}/{len(hotels)} exported in {elapsed:.1f}s.")
    return 0 if success == len(hotels) else 1


if __name__ == "__main__":
    sys.exit(main())
