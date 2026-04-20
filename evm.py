"""
RoadOpt AI — Earned Value Management (EVM)
Computes PV, EV, AC, CPI, SPI, EAC, ETC and generates S-curve data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from config import RESOURCE_POOLS


def compute_evm(
    schedule: List[Dict],
    makespan: int,
    progress_pct: float = 0.5,
    cost_variance_factor: float = 1.05,
) -> Dict:
    """
    Compute Earned Value Management metrics at a given progress point.
    Returns weekly PV/EV/AC arrays and all standard EVM KPIs.
    """
    current_week = int(makespan * progress_pct)

    # Planned cost per week
    weekly_planned = np.zeros(makespan)
    for entry in schedule:
        cost_pw = sum(
            qty * RESOURCE_POOLS.get(res, {}).get("cost_per_week", 0)
            for res, qty in entry["resources"].items()
        )
        for w in range(entry["start_week"], min(entry["end_week"], makespan)):
            weekly_planned[w] += cost_pw

    BAC = float(np.sum(weekly_planned))
    pv_cum = np.cumsum(weekly_planned)

    # Earned Value — work completed up to current_week
    weekly_earned = np.zeros(makespan)
    for entry in schedule:
        cost_pw = sum(
            qty * RESOURCE_POOLS.get(res, {}).get("cost_per_week", 0)
            for res, qty in entry["resources"].items()
        )
        for w in range(entry["start_week"], min(entry["end_week"], makespan)):
            if w <= current_week:
                weekly_earned[w] += cost_pw
    ev_cum = np.cumsum(weekly_earned)

    # Actual Cost — simulated with variance
    rng = np.random.RandomState(42)
    weekly_actual = np.zeros(makespan)
    for w in range(makespan):
        if w <= current_week:
            noise = rng.normal(cost_variance_factor, 0.03)
            weekly_actual[w] = weekly_planned[w] * max(0.9, noise)
    ac_cum = np.cumsum(weekly_actual)

    # KPIs at current_week
    idx = min(current_week, makespan - 1)
    PV = float(pv_cum[idx])
    EV = float(ev_cum[idx])
    AC = float(ac_cum[idx])

    CPI = EV / AC if AC > 0 else 1.0
    SPI = EV / PV if PV > 0 else 1.0
    CV = EV - AC
    SV = EV - PV
    EAC = BAC / CPI if CPI > 0 else BAC
    ETC = EAC - AC
    VAC = BAC - EAC

    cost_status = "🟢 Under Budget" if CPI > 1.0 else ("🟡 On Budget" if CPI > 0.95 else "🔴 Over Budget")
    sched_status = "🟢 Ahead" if SPI > 1.0 else ("🟡 On Track" if SPI > 0.95 else "🔴 Behind")

    return {
        "current_week": current_week,
        "BAC": round(BAC), "PV": round(PV), "EV": round(EV), "AC": round(AC),
        "CPI": round(CPI, 3), "SPI": round(SPI, 3),
        "CV": round(CV), "SV": round(SV),
        "EAC": round(EAC), "ETC": round(ETC), "VAC": round(VAC),
        "cost_status": cost_status, "schedule_status": sched_status,
        "pv_cumulative": pv_cum.tolist(),
        "ev_cumulative": ev_cum.tolist(),
        "ac_cumulative": ac_cum.tolist(),
        "weeks": list(range(makespan)),
    }
