"""
RoadOpt AI — Monte Carlo Simulation Engine
Probabilistic scheduling using PERT distributions.
Runs N simulations to produce P50/P80/P90 completion estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from data_generator import TASK_TEMPLATES, DEPENDENCY_MAP


def pert_sample(optimistic, most_likely, pessimistic, size=1, lam=4):
    """Sample from a PERT (modified Beta) distribution."""
    if pessimistic <= optimistic:
        return np.full(size, most_likely)
    mu = (optimistic + lam * most_likely + pessimistic) / (lam + 2)
    # Shape parameters
    denom = (most_likely - mu) * (pessimistic - optimistic)
    if abs(denom) < 1e-9:
        return np.full(size, most_likely)
    alpha = ((mu - optimistic) * (2 * most_likely - optimistic - pessimistic)) / denom
    alpha = max(0.5, alpha)
    beta_param = alpha * (pessimistic - mu) / max(1e-9, mu - optimistic)
    beta_param = max(0.5, beta_param)
    samples = np.random.beta(alpha, beta_param, size=size)
    return optimistic + samples * (pessimistic - optimistic)


def generate_pert_tasks():
    """Generate tasks with optimistic/most_likely/pessimistic durations."""
    tasks_pert = []
    for idx, (name, dur, comp, res) in enumerate(TASK_TEMPLATES):
        tasks_pert.append({
            "id": idx,
            "name": name,
            "optimistic": max(1, int(dur * 0.7)),
            "most_likely": dur,
            "pessimistic": int(dur * 1.6),
            "complexity": comp,
            "resource_requirements": dict(res),
            "predecessors": list(DEPENDENCY_MAP.get(idx, [])),
        })
    return tasks_pert


def _forward_pass(tasks_pert, durations):
    """CPM forward pass. Returns (earliest_starts, makespan)."""
    n = len(tasks_pert)
    es = [0] * n
    for t in tasks_pert:
        if t["predecessors"]:
            es[t["id"]] = max(es[p] + durations[p] for p in t["predecessors"])
    makespan = max(es[i] + durations[i] for i in range(n))
    return es, makespan


def run_monte_carlo(n_simulations: int = 1000, seed: int = 42) -> Dict:
    """
    Run Monte Carlo simulation.
    Returns makespans array, percentiles, per-task stats, criticality index.
    """
    np.random.seed(seed)
    tasks_pert = generate_pert_tasks()
    n_tasks = len(tasks_pert)

    makespans = np.zeros(n_simulations)
    all_durations = np.zeros((n_simulations, n_tasks))
    all_finishes = np.zeros((n_simulations, n_tasks))

    for sim in range(n_simulations):
        durations = np.zeros(n_tasks, dtype=int)
        for t in tasks_pert:
            dur = pert_sample(t["optimistic"], t["most_likely"], t["pessimistic"])[0]
            durations[t["id"]] = max(1, int(round(dur)))
        all_durations[sim] = durations
        es, makespan = _forward_pass(tasks_pert, durations)
        for t in tasks_pert:
            all_finishes[sim, t["id"]] = es[t["id"]] + durations[t["id"]]
        makespans[sim] = makespan

    # Percentiles
    percentiles = {
        "P10": int(np.percentile(makespans, 10)),
        "P25": int(np.percentile(makespans, 25)),
        "P50": int(np.percentile(makespans, 50)),
        "P75": int(np.percentile(makespans, 75)),
        "P80": int(np.percentile(makespans, 80)),
        "P90": int(np.percentile(makespans, 90)),
        "P95": int(np.percentile(makespans, 95)),
        "Mean": round(float(np.mean(makespans)), 1),
        "Std": round(float(np.std(makespans)), 1),
    }

    # Criticality index: % of simulations each task finishes at makespan
    criticality = np.zeros(n_tasks)
    for sim in range(n_simulations):
        ms = makespans[sim]
        for t in tasks_pert:
            if all_finishes[sim, t["id"]] >= ms - 0.5:
                criticality[t["id"]] += 1
    criticality = criticality / n_simulations * 100

    task_stats = []
    for t in tasks_pert:
        tid = t["id"]
        task_stats.append({
            "task_name": t["name"],
            "optimistic": t["optimistic"],
            "most_likely": t["most_likely"],
            "pessimistic": t["pessimistic"],
            "sim_mean": round(float(all_durations[:, tid].mean()), 1),
            "sim_std": round(float(all_durations[:, tid].std()), 1),
            "criticality_%": round(float(criticality[tid]), 1),
        })

    return {
        "makespans": makespans,
        "percentiles": percentiles,
        "task_stats": pd.DataFrame(task_stats),
        "n_simulations": n_simulations,
    }
