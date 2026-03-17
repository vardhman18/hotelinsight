# HotelInsight — User Manual

## Overview

HotelInsight is a **5-page Streamlit analytics app** that uses AI to help hotel managers
understand what guests complain about, why issues occur, and what concrete actions to take.

---

## Page-by-Page Guide

### 1. Home

The landing page describes the system capabilities and quick-start steps.
No interaction is required; navigate using the **sidebar**.

---

### 2. Hotel Selection

**Purpose:** Choose which hotel to analyse.

| Step | Action |
|------|--------|
| 1    | Type part of a hotel name in the search box |
| 2    | Select the hotel from the drop-down |
| 3    | Review the preview stats (reviews, avg rating, date range) |
| 4    | Click **"Analyse Hotel →"** |

The selected hotel is stored in session state and available to all other pages.

---

### 3. Dashboard

**Purpose:** High-level overview of the hotel's performance.

#### KPI Row
| Metric | Description |
|--------|-------------|
| Health Score | Composite score 0–100 (higher = better) |
| Average Rating | Mean guest rating (1–5 stars) |
| Total Reviews | Number of analysed reviews |
| Avg Sentiment | Mean AI sentiment score (−1 = very negative, +1 = very positive) |

#### Charts
- **Rating Trend** — monthly average rating over time
- **Complaint Frequency** — how often each category appears in negative reviews (%)
- **Impact Heatmap** — the rating drop caused by each complaint category

#### Issue Cards
Each identified complaint category appears as an expandable card showing:
- Priority badge (Critical / High / Medium / Low)
- Trend direction (🔴 Worsening / 🟢 Improving / 🔵 Stable)
- Sample bad reviews
- A button to navigate directly to the Action Plans page for that topic

---

### 4. Detailed Analysis

**Purpose:** Deep-dive into a single complaint category.

1. Select a **topic** from the drop-down (e.g. "cleanliness")
2. The page shows:
   - KPI row: complaint count, complaint rate, avg rating in affected reviews
   - A 2-panel chart: rating histogram + monthly trend
   - Root cause breakdown with confidence levels and evidence snippets
   - Sample guest reviews with star display and date

---

### 5. Action Plans

**Purpose:** View AI-generated remedial actions with cost estimates.

1. Select a **topic** and enter the **number of rooms** in your hotel
2. A cost summary banner displays the investment required
3. Actions are grouped into three timeframes:
   - 🚨 **Immediate (this week)** — quick wins
   - ⚡ **Short-Term (2–6 weeks)** — process improvements
   - 🌱 **Long-Term (1–6 months)** — structural changes

#### ROI Calculator
Enter:
- Average nightly room rate (£)
- Current occupancy rate (%)

Click **"Calculate ROI"** to see:
- Projected rating improvement
- Occupancy uplift
- Monthly revenue increase
- 3-month net profit and ROI %
- A waterfall chart visualising the financial model

#### Export
Click **"Export Report to Excel"** to save a multi-sheet workbook to `data/results/`.

---

### 6. Progress Tracker

**Purpose:** Monitor complaint trends over time.

- **Rating Over Time** chart — monthly average rating trajectory
- **Complaint Trend Table** — category-by-category trend direction
- **Monthly Complaint Rate Drill-Down** — bar chart for a selected topic
- **Last-30-Days Snapshot** — recent KPIs

---

## Understanding Priority Scores

| Badge | Score Range | Meaning |
|-------|-------------|---------|
| 🔴 Critical | ≥ 60 | Urgent intervention required |
| 🟠 High | 40–59 | Address in the next sprint |
| 🟡 Medium | 20–39 | Plan for next quarter |
| 🟢 Low | < 20 | Monitor and review annually |

Priority score is based on complaint frequency, rating impact, and trend direction.

---

## Understanding Root Causes

| Code | Description |
|------|-------------|
| UNDERSTAFFING | Too few staff to deliver the expected service level |
| TRAINING_ISSUES | Staff lack the skills or procedures to handle the task |
| EQUIPMENT_ISSUES | Physical equipment is broken, missing, or outdated |
| TIME_PRESSURE | Rushed service due to high demand or poor scheduling |
| SUPPLY_CHAIN_ISSUES | Supplies (e.g. breakfast items, toiletries) not available |
| PROCESS_INEFFICIENCY | Slow or unclear internal processes |
| COMMUNICATION_BREAKDOWN | Poor information flow between teams or with guests |

---

## Cost Formula Variables

Action cost formulas use the following variables (configurable in `src/config/settings.py`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `rooms` | (your input) | Number of guest rooms |
| `TRAINING_COST_BASE` | £1,200 | Fixed cost per training session |
| `EQUIPMENT_COST_PER_ROOM` | £100 | Per-room equipment allowance |
| `STAFF_COST_PER_ROOM` | £30 | Monthly staff cost per room |

---

## Exporting Results via CLI

Batch export without opening the app:

```bash
# Export top-10 hotels by review count
python scripts/export_results.py --top 10

# Export a specific hotel
python scripts/export_results.py --hotel "Hotel Arena"

# Export all hotels (slow)
python scripts/export_results.py --all
```

Reports are saved as `.xlsx` files in `data/results/`.
