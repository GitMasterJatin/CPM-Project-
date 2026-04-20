"""
RoadOpt AI — Construction Project Management Suite
Pure construction PM features:
  1. Float/Slack Analysis
  2. Bill of Quantities (BOQ)
  3. Cash Flow Projection
  4. Equipment Utilization Report
  5. Contractor Payment Milestones
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from config import RESOURCE_POOLS
from data_generator import TASK_TEMPLATES


# ═══════════════════════════════════════════════════════════════════════
# 1. FLOAT / SLACK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_float_analysis(tasks, schedule: List[Dict]) -> pd.DataFrame:
    """
    Compute Total Float, Free Float, and criticality for each task.
    Total Float = LS - ES (how much a task can slip without delaying the project)
    Free Float = min(ES of successors) - EF (slip without delaying any successor)
    """
    sched_by_id = {s["task_id"]: s for s in schedule}

    # Build successor map
    successors = {t.id: [] for t in tasks}
    for t in tasks:
        for p in t.predecessors:
            successors[p].append(t.id)

    makespan = max(s["end_week"] for s in schedule)
    rows = []
    for t in tasks:
        s = sched_by_id.get(t.id)
        if not s:
            continue

        es = s["start_week"]
        ef = s["end_week"]
        total_float = t.latest_start - t.earliest_start

        # Free float
        if successors[t.id]:
            min_succ_es = min(sched_by_id[sid]["start_week"]
                              for sid in successors[t.id] if sid in sched_by_id)
            free_float = min_succ_es - ef
        else:
            free_float = makespan - ef

        free_float = max(0, free_float)

        if total_float == 0:
            status = "🔴 Critical (0 float)"
        elif total_float <= 2:
            status = "🟡 Near-Critical"
        elif total_float <= 5:
            status = "🟢 Has Buffer"
        else:
            status = "🔵 Flexible"

        rows.append({
            "task_id": t.id,
            "task_name": t.name,
            "duration": t.duration_weeks,
            "ES": es, "EF": ef,
            "LS": t.latest_start, "LF": t.latest_start + t.duration_weeks,
            "total_float": total_float,
            "free_float": free_float,
            "is_critical": total_float == 0,
            "status": status,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 2. BILL OF QUANTITIES (BOQ)
# ═══════════════════════════════════════════════════════════════════════

# Material requirements per task: {task_index: [(material, qty, unit, unit_rate_inr)]}
BOQ_DATA = {
    4:  [("Topsoil Removal", 5000, "cu.m", 120)],
    5:  [("PVC Drainage Pipes", 800, "m", 350), ("Geotextile Fabric", 2000, "sq.m", 85)],
    6:  [("Earthwork Fill", 25000, "cu.m", 180), ("Geotextile", 5000, "sq.m", 85)],
    7:  [("HDPE Pipes", 500, "m", 450), ("Electrical Cables", 2000, "m", 220)],
    8:  [("Embankment Fill Material", 40000, "cu.m", 200), ("Geotextile", 8000, "sq.m", 85)],
    9:  [("Earthwork (Cut)", 15000, "cu.m", 150), ("Earthwork (Fill)", 15000, "cu.m", 180)],
    10: [("Subgrade Soil", 12000, "cu.m", 160), ("Lime/Cement", 200, "MT", 6500)],
    11: [("Granular Sub-Base (GSB)", 8000, "cu.m", 950), ("Water", 500, "KL", 80)],
    12: [("Wet Mix Macadam (WMM)", 6000, "cu.m", 1400), ("Crushed Aggregate", 3000, "cu.m", 850)],
    13: [("Dense Bituminous Macadam", 3500, "cu.m", 8500), ("VG-30 Bitumen", 450, "MT", 42000)],
    14: [("Bituminous Concrete", 2500, "cu.m", 9200), ("Modified Bitumen", 300, "MT", 52000)],
    15: [("M-35 Concrete", 4000, "cu.m", 6500), ("TMT Rebar (Fe-500)", 500, "MT", 58000),
         ("Pile Materials", 200, "nos", 15000)],
    16: [("M-40 Concrete", 3000, "cu.m", 7200), ("Structural Steel", 350, "MT", 72000),
         ("Pre-stressed Cables", 80, "MT", 95000), ("Formwork", 4000, "sq.m", 450)],
    17: [("M-25 Concrete", 800, "cu.m", 5800), ("TMT Rebar", 60, "MT", 58000),
         ("RCC Hume Pipes", 150, "m", 3200)],
    18: [("M-30 Concrete", 1200, "cu.m", 6200), ("TMT Rebar", 100, "MT", 58000),
         ("Stone Masonry", 500, "cu.m", 4500)],
    19: [("Road Signs", 120, "nos", 3500), ("Metal Crash Barriers", 5000, "m", 2800),
         ("Thermoplastic Paint", 15000, "sq.m", 180), ("Cat Eyes/Reflectors", 500, "nos", 250)],
    20: [("Tree Saplings", 2000, "nos", 150), ("Grass Turf", 10000, "sq.m", 45),
         ("Topsoil", 3000, "cu.m", 120)],
}


def generate_boq(tasks) -> pd.DataFrame:
    """Generate Bill of Quantities for the entire project."""
    rows = []
    for t in tasks:
        materials = BOQ_DATA.get(t.id, [])
        if not materials:
            rows.append({
                "task_name": t.name,
                "material": "— (No materials)",
                "quantity": 0, "unit": "—",
                "unit_rate_inr": 0,
                "total_cost_inr": 0,
            })
        else:
            for mat_name, qty, unit, rate in materials:
                rows.append({
                    "task_name": t.name,
                    "material": mat_name,
                    "quantity": qty, "unit": unit,
                    "unit_rate_inr": rate,
                    "total_cost_inr": qty * rate,
                })
    return pd.DataFrame(rows)


def boq_summary(boq_df: pd.DataFrame) -> Dict:
    """Aggregate BOQ stats."""
    material_rows = boq_df[boq_df["total_cost_inr"] > 0]
    total_material_cost = material_rows["total_cost_inr"].sum()

    by_material = material_rows.groupby("material")["total_cost_inr"].sum() \
        .sort_values(ascending=False).to_dict()

    return {
        "total_material_cost": int(total_material_cost),
        "unique_materials": len(material_rows["material"].unique()),
        "top_materials": by_material,
        "tasks_with_materials": len(material_rows["task_name"].unique()),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. CASH FLOW PROJECTION
# ═══════════════════════════════════════════════════════════════════════

def compute_cash_flow(schedule: List[Dict], boq_df: pd.DataFrame,
                      makespan: int) -> Dict:
    """
    Weekly cash flow = resource costs + material costs spread over task duration.
    Returns weekly and cumulative cash flow data.
    """
    weekly_resource_cost = np.zeros(makespan)
    weekly_material_cost = np.zeros(makespan)

    for entry in schedule:
        # Resource cost per week
        res_cost_pw = sum(
            qty * RESOURCE_POOLS.get(res, {}).get("cost_per_week", 0)
            for res, qty in entry["resources"].items()
        )
        for w in range(entry["start_week"], min(entry["end_week"], makespan)):
            weekly_resource_cost[w] += res_cost_pw

        # Material cost spread evenly over task duration
        task_materials = boq_df[boq_df["task_name"] == entry["task_name"]]
        mat_total = task_materials["total_cost_inr"].sum()
        dur = entry["duration"]
        if dur > 0 and mat_total > 0:
            mat_per_week = mat_total / dur
            for w in range(entry["start_week"], min(entry["end_week"], makespan)):
                weekly_material_cost[w] += mat_per_week

    weekly_total = weekly_resource_cost + weekly_material_cost
    cumulative = np.cumsum(weekly_total)

    return {
        "weeks": list(range(makespan)),
        "weekly_resource": weekly_resource_cost.tolist(),
        "weekly_material": weekly_material_cost.tolist(),
        "weekly_total": weekly_total.tolist(),
        "cumulative": cumulative.tolist(),
        "total_project_cost": int(cumulative[-1]) if len(cumulative) > 0 else 0,
        "peak_weekly_spend": int(np.max(weekly_total)),
        "peak_week": int(np.argmax(weekly_total)),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. EQUIPMENT UTILIZATION REPORT
# ═══════════════════════════════════════════════════════════════════════

# Operating hours per week and idle threshold
HOURS_PER_WEEK = 48

def compute_equipment_utilization(
    schedule: List[Dict],
    resource_usage: Dict[int, Dict[str, int]],
    resource_caps: Dict[str, int],
    makespan: int,
) -> pd.DataFrame:
    """
    Compute per-equipment utilization: operating weeks, idle weeks,
    utilization %, estimated fuel consumption, and maintenance alerts.
    """
    equipment_list = [r for r in resource_caps if r != "labor_crew" and r != "surveyor_team"]

    rows = []
    for equip in equipment_list:
        cap = resource_caps[equip]
        active_weeks = 0
        total_units_used = 0
        peak_usage = 0
        idle_weeks = 0

        for w in range(makespan):
            usage = resource_usage.get(w, {}).get(equip, 0)
            if usage > 0:
                active_weeks += 1
                total_units_used += usage
                peak_usage = max(peak_usage, usage)
            else:
                idle_weeks += 1

        avg_usage = total_units_used / max(active_weeks, 1)
        utilization_pct = (active_weeks / makespan * 100) if makespan > 0 else 0
        operating_hours = active_weeks * HOURS_PER_WEEK
        idle_hours = idle_weeks * HOURS_PER_WEEK
        cost_per_week = RESOURCE_POOLS.get(equip, {}).get("cost_per_week", 0)
        total_cost = active_weeks * cost_per_week * avg_usage

        # Maintenance alert
        if operating_hours > 1500:
            maintenance = "🔴 Overdue"
        elif operating_hours > 1000:
            maintenance = "🟡 Due Soon"
        else:
            maintenance = "🟢 OK"

        rows.append({
            "equipment": equip.replace("_", " ").title(),
            "capacity": cap,
            "peak_usage": peak_usage,
            "active_weeks": active_weeks,
            "idle_weeks": idle_weeks,
            "utilization_%": round(utilization_pct, 1),
            "operating_hours": operating_hours,
            "idle_hours": idle_hours,
            "avg_units_deployed": round(avg_usage, 1),
            "total_cost_inr": int(total_cost),
            "maintenance_status": maintenance,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 5. CONTRACTOR PAYMENT MILESTONES
# ═══════════════════════════════════════════════════════════════════════

# NHAI-style payment structure: milestone → (linked task IDs, % of contract)
PAYMENT_MILESTONES = [
    {"milestone": "Mobilization Advance",
     "task_ids": [0, 1],
     "payment_pct": 10.0,
     "description": "Site setup, survey, environmental clearance"},
    {"milestone": "Site Preparation Complete",
     "task_ids": [3, 4, 5, 6, 7],
     "payment_pct": 15.0,
     "description": "Land acquired, cleared, utilities relocated"},
    {"milestone": "Earthwork & Subgrade Done",
     "task_ids": [8, 9, 10],
     "payment_pct": 15.0,
     "description": "Embankment, cut-fill, subgrade preparation"},
    {"milestone": "Pavement Layers Complete",
     "task_ids": [11, 12, 13, 14],
     "payment_pct": 25.0,
     "description": "GSB, WMM, DBM, BC surface layers"},
    {"milestone": "Structures Complete",
     "task_ids": [15, 16, 17, 18],
     "payment_pct": 20.0,
     "description": "Bridge, culverts, retaining walls"},
    {"milestone": "Finishing & Handover",
     "task_ids": [19, 20, 21, 22, 23],
     "payment_pct": 10.0,
     "description": "Road furniture, landscaping, testing, handover"},
    {"milestone": "Retention Release",
     "task_ids": [23],
     "payment_pct": 5.0,
     "description": "Released after defect liability period"},
]


def compute_payment_schedule(
    schedule: List[Dict],
    total_contract_value: float = None,
) -> pd.DataFrame:
    """
    Compute contractor payment schedule based on milestone completion.
    """
    sched_by_id = {s["task_id"]: s for s in schedule}

    # Estimate total contract value from resource costs if not provided
    if total_contract_value is None:
        total_contract_value = sum(
            sum(qty * RESOURCE_POOLS.get(r, {}).get("cost_per_week", 0)
                for r, qty in s["resources"].items()) * s["duration"]
            for s in schedule
        )
        total_contract_value *= 1.3  # Add 30% for materials, overheads, profit

    rows = []
    cumulative = 0
    for ms in PAYMENT_MILESTONES:
        # Find when all linked tasks are complete
        completion_week = 0
        all_done = True
        for tid in ms["task_ids"]:
            entry = sched_by_id.get(tid)
            if entry:
                completion_week = max(completion_week, entry["end_week"])
            else:
                all_done = False

        payment_amount = total_contract_value * ms["payment_pct"] / 100
        cumulative += payment_amount

        rows.append({
            "milestone": ms["milestone"],
            "linked_tasks": len(ms["task_ids"]),
            "completion_week": completion_week if all_done else "TBD",
            "payment_%": f"{ms['payment_pct']}%",
            "payment_amount_inr": int(payment_amount),
            "cumulative_inr": int(cumulative),
            "description": ms["description"],
        })

    return pd.DataFrame(rows), int(total_contract_value)
