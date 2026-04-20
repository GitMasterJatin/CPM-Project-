"""
RoadOpt AI — Schedule Crashing & Fast-Tracking
Cost-time tradeoff analysis for project acceleration.
"""

import pandas as pd
from typing import Dict
from copy import deepcopy

from data_generator import generate_project_tasks, TASK_TEMPLATES
from optimizer import compute_critical_path, solve_rcpsp, estimate_cost
from config import RESOURCE_POOLS


# Crash data: task_index → {max weeks reducible, cost per week crashed}
CRASH_DATA = {
    0:  {"max_reduction": 1, "cost_per_week": 200_000},
    1:  {"max_reduction": 1, "cost_per_week": 250_000},
    3:  {"max_reduction": 2, "cost_per_week": 300_000},
    4:  {"max_reduction": 1, "cost_per_week": 180_000},
    7:  {"max_reduction": 1, "cost_per_week": 350_000},
    8:  {"max_reduction": 2, "cost_per_week": 280_000},
    9:  {"max_reduction": 1, "cost_per_week": 250_000},
    10: {"max_reduction": 1, "cost_per_week": 200_000},
    13: {"max_reduction": 1, "cost_per_week": 400_000},
    14: {"max_reduction": 1, "cost_per_week": 450_000},
    15: {"max_reduction": 2, "cost_per_week": 500_000},
    16: {"max_reduction": 2, "cost_per_week": 600_000},
    17: {"max_reduction": 1, "cost_per_week": 220_000},
    18: {"max_reduction": 1, "cost_per_week": 280_000},
}


def compute_crash_tradeoff(max_steps: int = 10, seed: int = 42) -> Dict:
    """
    Progressively crash cheapest critical tasks.
    Returns tradeoff curve and crash plan.
    """
    # Baseline
    tasks = generate_project_tasks(seed=seed)
    tasks, _ = compute_critical_path(tasks)
    base_result = solve_rcpsp(tasks)
    if base_result["status"] == "INFEASIBLE":
        return {"tradeoff_curve": pd.DataFrame(), "crash_plan": pd.DataFrame(),
                "base_makespan": -1, "best_makespan": -1, "total_crash_cost": 0}

    base_cost = estimate_cost(base_result["schedule"])["total_cost"]
    tradeoff = [{"step": 0, "makespan": base_result["makespan"],
                 "crash_cost": 0, "total_cost": base_cost, "crashed_task": "None"}]

    remaining = {k: v["max_reduction"] for k, v in CRASH_DATA.items()}
    crashed_so_far = {}
    cumulative_crash = 0

    for step in range(1, max_steps + 1):
        # Rebuild tasks with crashes applied
        tasks_s = generate_project_tasks(seed=seed)
        for tid, red in crashed_so_far.items():
            tasks_s[tid].duration_weeks = max(1, tasks_s[tid].duration_weeks - red)
        tasks_s, _ = compute_critical_path(tasks_s)
        critical_ids = {t.id for t in tasks_s if t.is_critical}

        # Find cheapest crashable critical task
        best_tid, best_cost = None, float("inf")
        for tid in critical_ids:
            if tid in remaining and remaining[tid] > 0:
                c = CRASH_DATA[tid]["cost_per_week"]
                if c < best_cost:
                    best_cost, best_tid = c, tid
        if best_tid is None:
            break

        remaining[best_tid] -= 1
        crashed_so_far[best_tid] = crashed_so_far.get(best_tid, 0) + 1
        cumulative_crash += best_cost

        # Re-solve
        tasks_c = generate_project_tasks(seed=seed)
        for tid, red in crashed_so_far.items():
            tasks_c[tid].duration_weeks = max(1, tasks_c[tid].duration_weeks - red)
        tasks_c, _ = compute_critical_path(tasks_c)
        res = solve_rcpsp(tasks_c)
        if res["status"] == "INFEASIBLE":
            break

        res_cost = estimate_cost(res["schedule"])["total_cost"]
        tradeoff.append({"step": step, "makespan": res["makespan"],
                         "crash_cost": cumulative_crash,
                         "total_cost": res_cost + cumulative_crash,
                         "crashed_task": TASK_TEMPLATES[best_tid][0]})

        if len(tradeoff) > 2 and tradeoff[-1]["makespan"] >= tradeoff[-2]["makespan"]:
            break

    # Crash plan summary
    plan = []
    for tid, data in CRASH_DATA.items():
        used = crashed_so_far.get(tid, 0)
        plan.append({
            "task_name": TASK_TEMPLATES[tid][0],
            "max_reduction": data["max_reduction"],
            "weeks_crashed": used,
            "cost_per_week": f"₹{data['cost_per_week']:,}",
            "total_crash_cost": f"₹{used * data['cost_per_week']:,}",
            "status": "✅ Crashed" if used > 0 else "⬜ Available",
        })

    return {
        "tradeoff_curve": pd.DataFrame(tradeoff),
        "crash_plan": pd.DataFrame(plan),
        "base_makespan": base_result["makespan"],
        "best_makespan": tradeoff[-1]["makespan"] if tradeoff else base_result["makespan"],
        "total_crash_cost": cumulative_crash,
    }
