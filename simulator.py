"""
RoadOpt AI — What-If Scenario Simulator
Compare multiple route plans / resource configurations under
different traffic, weather, and disruption scenarios.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_generator import generate_project_tasks, Task
from optimizer import compute_critical_path, solve_rcpsp, estimate_cost
from ai_predictor import predict_task_delays
from live_data import compute_weekly_external_risk
from constraints import run_all_constraint_checks, FuelBudget, ShiftPolicy
from config import RESOURCE_POOLS


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """A named what-if scenario with parameter overrides."""
    name: str
    description: str = ""
    # Resource shocks (% reduction, 0–100)
    labor_shock_pct: int = 0
    equipment_shock_pct: int = 0
    # Material / crew overrides
    material_availability: float = 0.85
    crew_experience: int = 3
    # Schedule overrides
    start_month: int = 4          # April default
    objective: str = "minimize_time"
    # Budget overrides
    fuel_budget_inr: float = 5_000_000.0
    # Weather scenario
    weather_severity: str = "normal"  # "normal", "mild", "severe"
    # Custom resource caps (if None, uses defaults with shocks applied)
    custom_resource_caps: Optional[Dict[str, int]] = None
    # Seed for reproducibility
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# PRE-BUILT SCENARIO LIBRARY
# ═══════════════════════════════════════════════════════════════════════

SCENARIO_LIBRARY = {
    "baseline": Scenario(
        name="📋 Baseline",
        description="Standard plan — no disruptions, default resources",
    ),
    "monsoon_delay": Scenario(
        name="🌧️ Monsoon Disruption",
        description="Start in June with heavy monsoon impact + material shortages",
        start_month=6,
        material_availability=0.60,
        weather_severity="severe",
        labor_shock_pct=15,
    ),
    "labor_crisis": Scenario(
        name="👷 Labor Shortage",
        description="40% labor reduction (migration / strike scenario)",
        labor_shock_pct=40,
        crew_experience=2,
    ),
    "equipment_failure": Scenario(
        name="🔧 Equipment Breakdown",
        description="30% equipment unavailability due to breakdowns",
        equipment_shock_pct=30,
    ),
    "budget_crunch": Scenario(
        name="💰 Budget Crunch",
        description="Fuel budget cut by 40%, cost-optimized scheduling",
        fuel_budget_inr=3_000_000.0,
        objective="minimize_cost",
    ),
    "fast_track": Scenario(
        name="🚀 Fast Track",
        description="Maximum resources, experienced crew, winter start",
        start_month=11,
        crew_experience=5,
        material_availability=0.95,
        objective="minimize_time",
    ),
    "worst_case": Scenario(
        name="⛔ Worst Case",
        description="Monsoon start + labor shortage + equipment issues",
        start_month=7,
        labor_shock_pct=30,
        equipment_shock_pct=25,
        material_availability=0.55,
        crew_experience=2,
        weather_severity="severe",
        fuel_budget_inr=3_500_000.0,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# SIMULATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _apply_resource_shocks(
    base_caps: Dict[str, int],
    labor_shock_pct: int,
    equipment_shock_pct: int,
) -> Dict[str, int]:
    """Apply % reductions to resource caps."""
    caps = dict(base_caps)
    if labor_shock_pct > 0:
        caps["labor_crew"] = max(1, int(caps["labor_crew"] * (1 - labor_shock_pct / 100)))
    if equipment_shock_pct > 0:
        for eq in ["excavator", "bulldozer", "asphalt_paver", "roller_compactor", "crane"]:
            if eq in caps:
                caps[eq] = max(1, int(caps[eq] * (1 - equipment_shock_pct / 100)))
    return caps


def run_scenario(
    scenario: Scenario,
    base_resource_caps: Dict[str, int] | None = None,
) -> Dict:
    """
    Execute a single scenario: generate tasks, solve, predict delays, compute costs.
    Returns a comprehensive result dictionary.
    """
    if base_resource_caps is None:
        base_resource_caps = {k: v["units"] for k, v in RESOURCE_POOLS.items()}

    # Apply resource shocks
    if scenario.custom_resource_caps:
        adjusted_caps = dict(scenario.custom_resource_caps)
    else:
        adjusted_caps = _apply_resource_shocks(
            base_resource_caps,
            scenario.labor_shock_pct,
            scenario.equipment_shock_pct,
        )

    # Generate & solve
    tasks = generate_project_tasks(seed=scenario.seed)
    tasks, cpm_makespan = compute_critical_path(tasks)
    result = solve_rcpsp(
        tasks,
        resource_caps=adjusted_caps,
        objective=scenario.objective,
    )

    if result["status"] == "INFEASIBLE":
        return {
            "scenario": scenario,
            "status": "INFEASIBLE",
            "makespan": -1,
            "cpm_makespan": cpm_makespan,
            "cost": {"total_cost": 0, "breakdown": {}},
            "risk_df": pd.DataFrame(),
            "external_risk": pd.DataFrame(),
            "constraint_report": None,
            "adjusted_caps": adjusted_caps,
        }

    cost_data = estimate_cost(result["schedule"])

    # Delay predictions
    risk_df = predict_task_delays(
        tasks,
        resource_utilization=0.7,
        start_month=scenario.start_month,
        crew_experience=scenario.crew_experience,
        material_availability=scenario.material_availability,
    )

    # External risk
    external_risk = compute_weekly_external_risk(
        start_month=scenario.start_month,
        num_weeks=max(result["makespan"], 4),
        seed=scenario.seed,
    )

    # Constraint checks
    fuel = FuelBudget(total_budget_inr=scenario.fuel_budget_inr)
    constraint_report = run_all_constraint_checks(
        result["schedule"],
        makespan=result["makespan"],
        fuel_budget=fuel,
    )

    # Estimated total delay
    total_predicted_delay = risk_df["predicted_delay_weeks"].sum() if len(risk_df) > 0 else 0
    high_risk_count = len(risk_df[risk_df["risk_level"].str.contains("High")]) if len(risk_df) > 0 else 0
    avg_external_risk = external_risk["combined_risk"].mean() if len(external_risk) > 0 else 0

    return {
        "scenario": scenario,
        "status": result["status"],
        "makespan": result["makespan"],
        "cpm_makespan": cpm_makespan,
        "schedule": result["schedule"],
        "resource_usage": result["resource_usage"],
        "cost": cost_data,
        "risk_df": risk_df,
        "external_risk": external_risk,
        "constraint_report": constraint_report,
        "adjusted_caps": adjusted_caps,
        "total_predicted_delay": round(total_predicted_delay, 1),
        "high_risk_tasks": high_risk_count,
        "avg_external_risk": round(avg_external_risk, 3),
    }


def run_comparison(
    scenarios: List[Scenario],
    base_resource_caps: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Run multiple scenarios and return a comparison DataFrame.
    """
    results = []
    for sc in scenarios:
        res = run_scenario(sc, base_resource_caps)
        row = {
            "Scenario": sc.name,
            "Description": sc.description,
            "Status": res["status"],
            "Makespan (weeks)": res["makespan"],
            "CPM Ideal (weeks)": res["cpm_makespan"],
            "Delay vs Ideal": res["makespan"] - res["cpm_makespan"] if res["makespan"] > 0 else "N/A",
            "Total Cost (₹)": f"₹{res['cost']['total_cost']:,.0f}" if res["cost"]["total_cost"] > 0 else "N/A",
            "Predicted Delay (weeks)": res.get("total_predicted_delay", 0),
            "High-Risk Tasks": res.get("high_risk_tasks", 0),
            "Avg External Risk": res.get("avg_external_risk", 0),
            "Constraint Health": res["constraint_report"]["summary"]["health"] if res["constraint_report"] else "N/A",
            "Penalties (₹)": f"₹{res['constraint_report']['summary']['total_penalties_inr']:,.0f}" if res["constraint_report"] else "N/A",
        }
        results.append(row)

    return pd.DataFrame(results)


def get_scenario_detail(
    scenario_key: str,
    base_resource_caps: Dict[str, int] | None = None,
) -> Dict:
    """Run a single named scenario from the library."""
    sc = SCENARIO_LIBRARY.get(scenario_key)
    if sc is None:
        raise ValueError(f"Unknown scenario: {scenario_key}. Options: {list(SCENARIO_LIBRARY.keys())}")
    return run_scenario(sc, base_resource_caps)


def build_custom_scenario(
    name: str,
    labor_shock: int = 0,
    equipment_shock: int = 0,
    material_avail: float = 0.85,
    crew_exp: int = 3,
    start_month: int = 4,
    objective: str = "minimize_time",
    fuel_budget: float = 5_000_000.0,
) -> Scenario:
    """Build a custom scenario from user inputs."""
    return Scenario(
        name=f"🔧 {name}",
        description=f"Custom: labor-{labor_shock}% equip-{equipment_shock}% mat={material_avail}",
        labor_shock_pct=labor_shock,
        equipment_shock_pct=equipment_shock,
        material_availability=material_avail,
        crew_experience=crew_exp,
        start_month=start_month,
        objective=objective,
        fuel_budget_inr=fuel_budget,
    )
