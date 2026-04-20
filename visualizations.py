"""
RoadOpt AI — Visualization Helpers
Plotly-based charts: Gantt, Resource Heatmap, Cost Breakdown, Risk Matrix
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List


def create_gantt_chart(schedule: List[Dict], title: str = "Project Schedule") -> go.Figure:
    """Interactive Gantt chart from solver schedule."""
    df = pd.DataFrame(schedule)
    df["color"] = df["is_critical"].map({True: "Critical Path", False: "Non-Critical"})

    fig = px.timeline(
        df,
        x_start=df["start_week"].apply(lambda w: f"2026-W{w+1:02d}"),
        x_end=df["end_week"].apply(lambda w: f"2026-W{w+1:02d}"),
        y="task_name",
        color="color",
        color_discrete_map={"Critical Path": "#e74c3c", "Non-Critical": "#3498db"},
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=700, xaxis_title="Week", yaxis_title="", showlegend=True)
    return fig


def create_gantt_chart_bar(schedule: List[Dict], title: str = "Project Schedule — Gantt") -> go.Figure:
    """Horizontal bar Gantt (works without date parsing issues)."""
    df = pd.DataFrame(schedule).sort_values("start_week")

    colors = ["#e74c3c" if row["is_critical"] else "#3498db" for _, row in df.iterrows()]

    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["duration"]],
            y=[row["task_name"]],
            base=[row["start_week"]],
            orientation="h",
            marker_color="#e74c3c" if row["is_critical"] else "#3498db",
            name="Critical" if row["is_critical"] else "Normal",
            showlegend=False,
            hovertemplate=(
                f"<b>{row['task_name']}</b><br>"
                f"Start: Week {row['start_week']}<br>"
                f"End: Week {row['end_week']}<br>"
                f"Duration: {row['duration']} weeks<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Project Week",
        yaxis_title="",
        height=max(500, len(df) * 28),
        barmode="stack",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_resource_heatmap(
    resource_usage: Dict[int, Dict[str, int]],
    resource_caps: Dict[str, int],
) -> go.Figure:
    """Heatmap: weeks × resources, color = utilization %."""
    weeks = sorted(resource_usage.keys())
    resources = sorted(resource_caps.keys())

    z = []
    for res in resources:
        row = []
        cap = resource_caps[res]
        for w in weeks:
            usage = resource_usage[w].get(res, 0)
            pct = round(usage / cap * 100, 1) if cap > 0 else 0
            row.append(pct)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"W{w+1}" for w in weeks],
        y=[r.replace("_", " ").title() for r in resources],
        colorscale="RdYlGn_r",
        zmin=0,
        zmax=100,
        colorbar_title="Utilization %",
        hovertemplate="Week %{x}<br>%{y}: %{z:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Resource Utilization Heatmap",
        xaxis_title="Project Week",
        height=450,
    )
    return fig


def create_cost_breakdown(cost_data: Dict) -> go.Figure:
    """Donut chart of cost by resource type."""
    labels = [k.replace("_", " ").title() for k in cost_data["breakdown"]]
    values = list(cost_data["breakdown"].values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
    )])
    fig.update_layout(
        title=f"Cost Breakdown — Total: ₹{cost_data['total_cost']:,.0f}",
        height=450,
    )
    return fig


def create_risk_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of predicted delay per task."""
    df = risk_df.sort_values("predicted_delay_weeks", ascending=True)

    colors = []
    for _, row in df.iterrows():
        if "Low" in row["risk_level"]:
            colors.append("#27ae60")
        elif "Medium" in row["risk_level"]:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig = go.Figure(go.Bar(
        x=df["predicted_delay_weeks"],
        y=df["task_name"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.1f} weeks delay<extra></extra>",
    ))
    fig.update_layout(
        title="AI-Predicted Delay Risk per Task",
        xaxis_title="Predicted Delay (weeks)",
        yaxis_title="",
        height=max(450, len(df) * 25),
    )
    return fig


def create_feature_importance_chart(feature_imp: Dict[str, float]) -> go.Figure:
    """Bar chart of ML feature importances."""
    sorted_feats = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].replace("_", " ").title() for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#8e44ad",
    ))
    fig.update_layout(
        title="Delay Prediction — Feature Importance",
        xaxis_title="Importance",
        height=350,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# NEW: Scenario Comparison Charts
# ═══════════════════════════════════════════════════════════════════════

def create_scenario_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing makespan & cost across scenarios."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Makespan (weeks)",
        x=comparison_df["Scenario"],
        y=comparison_df["Makespan (weeks)"],
        marker_color="#3498db",
        text=comparison_df["Makespan (weeks)"],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Predicted Delay (weeks)",
        x=comparison_df["Scenario"],
        y=comparison_df["Predicted Delay (weeks)"],
        marker_color="#e74c3c",
        text=comparison_df["Predicted Delay (weeks)"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Scenario Comparison — Makespan & Predicted Delay",
        barmode="group",
        xaxis_title="Scenario",
        yaxis_title="Weeks",
        height=450,
    )
    return fig


def create_scenario_risk_radar(comparison_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing risk dimensions across scenarios."""
    categories = ["Makespan", "Predicted Delay", "High-Risk Tasks", "External Risk"]

    fig = go.Figure()
    for _, row in comparison_df.iterrows():
        if row["Status"] == "INFEASIBLE":
            continue
        vals = [
            row["Makespan (weeks)"] / max(comparison_df["Makespan (weeks)"].max(), 1),
            row["Predicted Delay (weeks)"] / max(comparison_df["Predicted Delay (weeks)"].max(), 1),
            row["High-Risk Tasks"] / max(comparison_df["High-Risk Tasks"].max(), 1),
            row["Avg External Risk"] / max(comparison_df["Avg External Risk"].max(), 0.01),
        ]
        vals.append(vals[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Scenario"],
            opacity=0.6,
        ))

    fig.update_layout(
        title="Risk Radar — Scenario Comparison",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
        showlegend=True,
    )
    return fig


def create_external_risk_timeline(risk_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of weather / traffic / event risk over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risk_df["week"], y=risk_df["weather_risk"],
        mode="lines", name="Weather Risk",
        fill="tozeroy", line=dict(color="#3498db"),
    ))
    fig.add_trace(go.Scatter(
        x=risk_df["week"], y=risk_df["traffic_risk"],
        mode="lines", name="Traffic Risk",
        fill="tonexty", line=dict(color="#e67e22"),
    ))
    fig.add_trace(go.Scatter(
        x=risk_df["week"], y=risk_df["event_risk"],
        mode="lines", name="Event Risk",
        fill="tonexty", line=dict(color="#e74c3c"),
    ))
    fig.update_layout(
        title="External Risk Timeline (Weather + Traffic + Events)",
        xaxis_title="Project Week",
        yaxis_title="Risk Score",
        height=400,
        yaxis=dict(range=[0, 1.2]),
    )
    return fig


def create_fuel_budget_chart(fuel_data: Dict) -> go.Figure:
    """Weekly fuel cost bar chart with budget cap line."""
    weeks = sorted(fuel_data["weekly_fuel_cost"].keys())
    costs = [fuel_data["weekly_fuel_cost"][w] for w in weeks]

    colors = ["#e74c3c" if c > fuel_data["weekly_cap"] else "#27ae60" for c in costs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"W{w+1}" for w in weeks],
        y=costs,
        marker_color=colors,
        name="Fuel Cost",
    ))
    fig.add_hline(
        y=fuel_data["weekly_cap"],
        line_dash="dash", line_color="red",
        annotation_text=f"Weekly Cap: ₹{fuel_data['weekly_cap']:,.0f}",
    )
    fig.update_layout(
        title=f"Weekly Fuel Cost — {fuel_data['status']} (Total: ₹{fuel_data['total_fuel_cost']:,.0f})",
        xaxis_title="Week",
        yaxis_title="Cost (₹)",
        height=400,
    )
    return fig


def create_milestone_timeline(ms_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of milestones: deadline vs projected finish."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ms_df["milestone"],
        x=ms_df["deadline_week"],
        name="Deadline",
        orientation="h",
        marker_color="#3498db",
        opacity=0.5,
    ))
    fig.add_trace(go.Bar(
        y=ms_df["milestone"],
        x=ms_df["projected_finish"],
        name="Projected Finish",
        orientation="h",
        marker_color=["#e74c3c" if r > 0 else "#27ae60" for r in ms_df["weeks_late"]],
    ))
    fig.update_layout(
        title="Priority Milestones — Deadline vs Projected",
        xaxis_title="Project Week",
        height=350,
        barmode="overlay",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# ML Quality Charts
# ═══════════════════════════════════════════════════════════════════════

def create_model_benchmark_chart(bench_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing MAE & R² across models."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE (weeks)",
        x=bench_df["Model"],
        y=bench_df["MAE (weeks)"],
        marker_color="#e74c3c",
        text=bench_df["MAE (weeks)"],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="RMSE (weeks)",
        x=bench_df["Model"],
        y=bench_df["RMSE (weeks)"],
        marker_color="#f39c12",
        text=bench_df["RMSE (weeks)"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Model Benchmark — MAE & RMSE Comparison",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Error (weeks)",
        height=420,
    )
    return fig


def create_model_r2_chart(bench_df: pd.DataFrame) -> go.Figure:
    """Bar chart of R² scores."""
    colors = ["#27ae60" if r > 0.7 else ("#f39c12" if r > 0.4 else "#e74c3c") for r in bench_df["R²"]]
    fig = go.Figure(go.Bar(
        x=bench_df["Model"],
        y=bench_df["R²"],
        marker_color=colors,
        text=bench_df["R²"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Model Benchmark — R² Score",
        xaxis_title="Model",
        yaxis_title="R²",
        height=380,
        yaxis=dict(range=[0, 1]),
    )
    return fig


def create_uncertainty_chart(unc_df: pd.DataFrame) -> go.Figure:
    """Error bar chart showing confidence intervals per task."""
    df = unc_df.sort_values("delay_median", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["delay_median"],
        y=df["task_name"],
        mode="markers",
        marker=dict(size=10, color="#2980b9"),
        name="Median Prediction",
        error_x=dict(
            type="data",
            symmetric=False,
            array=(df["delay_upper_90"] - df["delay_median"]).tolist(),
            arrayminus=(df["delay_median"] - df["delay_lower_10"]).tolist(),
            color="#e74c3c",
            thickness=2,
            width=6,
        ),
    ))
    fig.update_layout(
        title="Delay Prediction with 80% Confidence Interval (P10–P90)",
        xaxis_title="Predicted Delay (weeks)",
        height=max(450, len(df) * 28),
    )
    return fig


def create_shap_bar_chart(global_importance: Dict[str, float]) -> go.Figure:
    """SHAP global importance bar chart."""
    sorted_feats = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].replace("_", " ").title() for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#8e44ad",
    ))
    fig.update_layout(
        title="SHAP — Global Feature Importance (mean |SHAP|)",
        xaxis_title="Mean |SHAP value|",
        height=380,
    )
    return fig


def create_shap_waterfall(explanation: Dict) -> go.Figure:
    """Waterfall chart for a single prediction's SHAP contributions."""
    contribs = explanation.get("contributions", {})
    if not contribs:
        return go.Figure()

    names = []
    values = []
    for feat, info in contribs.items():
        names.append(feat.replace("_", " ").title())
        if "shap_contribution" in info:
            values.append(info["shap_contribution"])
        else:
            values.append(info.get("importance", 0))

    # Sort by absolute value
    order = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
    ))
    base_val = explanation.get("base_value", 0)
    pred_val = explanation.get("prediction", 0)
    fig.update_layout(
        title=f"SHAP Explanation — {explanation['task_name']} (pred={pred_val:.1f}w, base={base_val:.1f}w)",
        xaxis_title="SHAP Contribution (weeks)",
        height=380,
    )
    return fig


def create_drift_chart(drift_df: pd.DataFrame) -> go.Figure:
    """Bar chart of PSI values per feature with drift thresholds."""
    colors = []
    for _, row in drift_df.iterrows():
        if "No Drift" in row["Status"]:
            colors.append("#27ae60")
        elif "Minor" in row["Status"]:
            colors.append("#f39c12")
        elif "Moderate" in row["Status"]:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")

    fig = go.Figure(go.Bar(
        x=drift_df["Feature"],
        y=drift_df["PSI"],
        marker_color=colors,
        text=drift_df["PSI"],
        textposition="auto",
    ))
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                  annotation_text="Minor drift")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="Significant drift")
    fig.update_layout(
        title="Data Drift Detection — PSI per Feature",
        xaxis_title="Feature",
        yaxis_title="Population Stability Index (PSI)",
        height=420,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Monte Carlo Charts
# ═══════════════════════════════════════════════════════════════════════

def create_monte_carlo_histogram(makespans, percentiles: Dict) -> go.Figure:
    """Histogram of simulated makespans with percentile lines."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=makespans, nbinsx=40, marker_color="#3498db",
        opacity=0.75, name="Simulated Makespans",
    ))
    colors = {"P50": "#27ae60", "P80": "#f39c12", "P90": "#e74c3c", "P95": "#8e44ad"}
    for label, col in colors.items():
        val = percentiles[label]
        fig.add_vline(x=val, line_dash="dash", line_color=col,
                      annotation_text=f"{label}: {val}w", annotation_position="top")
    fig.update_layout(
        title="Monte Carlo Simulation — Project Completion Distribution",
        xaxis_title="Makespan (weeks)", yaxis_title="Frequency", height=420,
    )
    return fig


def create_criticality_chart(task_stats) -> go.Figure:
    """Horizontal bar chart of criticality index per task."""
    df = task_stats.sort_values("criticality_%", ascending=True)
    colors = ["#e74c3c" if v > 60 else ("#f39c12" if v > 30 else "#27ae60")
              for v in df["criticality_%"]]
    fig = go.Figure(go.Bar(
        x=df["criticality_%"], y=df["task_name"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.1f}% critical<extra></extra>",
    ))
    fig.update_layout(
        title="Criticality Index — % of Simulations on Critical Path",
        xaxis_title="Criticality (%)", height=max(450, len(df) * 25),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# EVM Charts
# ═══════════════════════════════════════════════════════════════════════

def create_evm_scurve(evm_data: Dict) -> go.Figure:
    """S-Curve: PV, EV, AC over project weeks."""
    weeks = evm_data["weeks"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=evm_data["pv_cumulative"],
                             mode="lines", name="Planned Value (PV)",
                             line=dict(color="#3498db", width=3)))
    fig.add_trace(go.Scatter(x=weeks, y=evm_data["ev_cumulative"],
                             mode="lines", name="Earned Value (EV)",
                             line=dict(color="#27ae60", width=3)))
    fig.add_trace(go.Scatter(x=weeks, y=evm_data["ac_cumulative"],
                             mode="lines", name="Actual Cost (AC)",
                             line=dict(color="#e74c3c", width=3, dash="dot")))
    fig.add_vline(x=evm_data["current_week"], line_dash="dash",
                  line_color="grey", annotation_text="Current Week")
    fig.update_layout(
        title="Earned Value Management — S-Curve",
        xaxis_title="Project Week", yaxis_title="Cumulative Cost (₹)", height=450,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Crashing Charts
# ═══════════════════════════════════════════════════════════════════════

def create_crash_tradeoff_chart(tradeoff_df) -> go.Figure:
    """Cost-Time tradeoff curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tradeoff_df["makespan"], y=tradeoff_df["crash_cost"],
        mode="lines+markers", name="Crash Cost",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=10),
        text=tradeoff_df["crashed_task"],
        hovertemplate="Makespan: %{x}w<br>Crash Cost: ₹%{y:,.0f}<br>Crashed: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Schedule Crashing — Cost-Time Tradeoff",
        xaxis_title="Makespan (weeks)", yaxis_title="Cumulative Crash Cost (₹)",
        height=420, xaxis=dict(autorange="reversed"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# DAG Network Graph
# ═══════════════════════════════════════════════════════════════════════

def create_dag_chart(tasks, schedule=None) -> go.Figure:
    """Task dependency network graph using layered layout."""
    from data_generator import DEPENDENCY_MAP

    # Compute layers via topological ordering (BFS by depth)
    n = len(tasks)
    depth = [0] * n
    for t in tasks:
        for p in t.predecessors:
            depth[t.id] = max(depth[t.id], depth[p] + 1)

    # Position nodes: x = depth, y = spread within layer
    layer_counts = {}
    for i in range(n):
        d = depth[i]
        layer_counts[d] = layer_counts.get(d, 0) + 1

    layer_idx = {}
    for i in range(n):
        d = depth[i]
        layer_idx[i] = layer_idx.get(d, -1) if d not in layer_idx else layer_idx[d]
    # Re-compute y positions
    layer_pos = {d: 0 for d in layer_counts}
    x_pos, y_pos = {}, {}
    for i in range(n):
        d = depth[i]
        total_in_layer = layer_counts[d]
        x_pos[i] = d
        y_pos[i] = layer_pos[d] - total_in_layer / 2
        layer_pos[d] += 1

    # Determine critical status
    is_crit = {}
    if schedule:
        for s in schedule:
            is_crit[s["task_id"]] = s["is_critical"]

    # Draw edges
    edge_x, edge_y = [], []
    for t in tasks:
        for p in t.predecessors:
            edge_x += [x_pos[p], x_pos[t.id], None]
            edge_y += [y_pos[p], y_pos[t.id], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#999"), hoverinfo="none",
    ))

    # Nodes
    node_colors = []
    for t in tasks:
        if is_crit.get(t.id, t.is_critical):
            node_colors.append("#e74c3c")
        else:
            node_colors.append("#3498db")

    node_text = [f"{t.name[:25]}" for t in tasks]
    hover_text = [f"<b>{t.name}</b><br>Duration: {t.duration_weeks}w<br>"
                  f"ES: {t.earliest_start} | LS: {t.latest_start}<br>"
                  f"Critical: {'Yes' if is_crit.get(t.id, t.is_critical) else 'No'}"
                  for t in tasks]

    fig.add_trace(go.Scatter(
        x=[x_pos[i] for i in range(n)],
        y=[y_pos[i] for i in range(n)],
        mode="markers+text", text=node_text,
        textposition="top center", textfont=dict(size=8),
        hovertext=hover_text, hoverinfo="text",
        marker=dict(size=18, color=node_colors, line=dict(width=2, color="white")),
    ))

    fig.update_layout(
        title="Task Dependency Network (DAG) — Red = Critical Path",
        showlegend=False, height=550,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Dependency Depth"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Geospatial Map View
# ═══════════════════════════════════════════════════════════════════════

def create_map_view(tasks, schedule=None) -> go.Figure:
    """Plot tasks along a simulated NH-48 route on a map."""
    # Simulated chainage points along NH-48 (Delhi → Jaipur stretch)
    route_points = [
        (28.6139, 77.2090, 0),    # Delhi
        (28.4595, 77.0266, 25),   # Gurgaon
        (28.2500, 76.8500, 50),
        (28.0500, 76.6500, 75),
        (27.8000, 76.5000, 100),
        (27.5500, 76.3500, 125),
        (27.3000, 76.2000, 150),
        (27.0000, 76.0000, 175),
        (26.9124, 75.7873, 200),  # Jaipur
    ]

    n = len(tasks)
    lats, lons, labels, colors, sizes = [], [], [], [], []

    for i, t in enumerate(tasks):
        # Distribute tasks along the route
        route_idx = min(int(i / n * len(route_points)), len(route_points) - 1)
        lat, lon, km = route_points[route_idx]
        # Add small offset so tasks don't overlap
        lat += (i % 3 - 1) * 0.02
        lon += (i % 2) * 0.015

        crit = False
        if schedule:
            for s in schedule:
                if s["task_id"] == t.id:
                    crit = s["is_critical"]
                    break
        else:
            crit = t.is_critical

        lats.append(lat)
        lons.append(lon)
        labels.append(f"{t.name[:30]} (Km {km})")
        colors.append("#e74c3c" if crit else "#3498db")
        sizes.append(15 if crit else 10)

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, mode="markers+text",
        marker=dict(size=sizes, color=colors),
        text=[t.name[:20] for t in tasks],
        textposition="top right", textfont=dict(size=8),
        hovertext=labels, hoverinfo="text",
    ))
    fig.update_layout(
        title="Geospatial View — Tasks Along NH-48 Route",
        mapbox=dict(style="open-street-map",
                    center=dict(lat=27.8, lon=76.5), zoom=6.5),
        height=550, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Construction PM Charts
# ═══════════════════════════════════════════════════════════════════════

def create_float_chart(float_df) -> go.Figure:
    """Horizontal bar chart of Total Float per task, colored by criticality."""
    df = float_df.sort_values("total_float", ascending=True)
    colors = []
    for _, r in df.iterrows():
        if r["total_float"] == 0:
            colors.append("#e74c3c")
        elif r["total_float"] <= 2:
            colors.append("#f39c12")
        elif r["total_float"] <= 5:
            colors.append("#27ae60")
        else:
            colors.append("#3498db")

    fig = go.Figure(go.Bar(
        x=df["total_float"], y=df["task_name"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Total Float: %{x} weeks<extra></extra>",
    ))
    fig.update_layout(
        title="Float / Slack Analysis — Total Float per Task",
        xaxis_title="Total Float (weeks)",
        height=max(450, len(df) * 25),
    )
    return fig


def create_cash_flow_chart(cf_data: Dict) -> go.Figure:
    """Stacked area for weekly spend + line for cumulative (S-curve)."""
    weeks = cf_data["weeks"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weeks, y=cf_data["weekly_resource"], name="Resource Cost",
        marker_color="#3498db", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=weeks, y=cf_data["weekly_material"], name="Material Cost",
        marker_color="#e67e22", opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=cf_data["cumulative"], name="Cumulative (S-Curve)",
        mode="lines", line=dict(color="#2c3e50", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Cash Flow Projection — Weekly Spend & Cumulative S-Curve",
        xaxis_title="Project Week",
        yaxis=dict(title="Weekly Cost (₹)", side="left"),
        yaxis2=dict(title="Cumulative (₹)", side="right", overlaying="y"),
        barmode="stack", height=450, legend=dict(x=0.01, y=0.99),
    )
    return fig


def create_equipment_utilization_chart(equip_df) -> go.Figure:
    """Bar chart of equipment utilization % with maintenance status."""
    colors = ["#e74c3c" if u > 85 else ("#f39c12" if u > 60 else "#27ae60")
              for u in equip_df["utilization_%"]]
    fig = go.Figure(go.Bar(
        x=equip_df["equipment"], y=equip_df["utilization_%"],
        marker_color=colors, text=equip_df["utilization_%"],
        texttemplate="%{text:.0f}%", textposition="auto",
    ))
    fig.update_layout(
        title="Equipment Utilization Report",
        xaxis_title="Equipment", yaxis_title="Utilization %",
        height=420, yaxis=dict(range=[0, 100]),
    )
    return fig


def create_boq_cost_chart(boq_df) -> go.Figure:
    """Horizontal bar chart of top materials by cost."""
    material_costs = boq_df[boq_df["total_cost_inr"] > 0].groupby("material")["total_cost_inr"].sum()
    material_costs = material_costs.sort_values(ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=material_costs.values,
        y=material_costs.index,
        orientation="h",
        marker_color="#8e44ad",
        hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Bill of Quantities — Top Materials by Cost",
        xaxis_title="Total Cost (₹)", height=max(400, len(material_costs) * 30),
    )
    return fig


def create_payment_chart(payment_df) -> go.Figure:
    """Bar chart of payment milestones with cumulative line."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=payment_df["milestone"],
        y=payment_df["payment_amount_inr"],
        name="Payment Amount",
        marker_color="#27ae60",
        text=[f"₹{v:,.0f}" for v in payment_df["payment_amount_inr"]],
        textposition="auto",
    ))
    fig.add_trace(go.Scatter(
        x=payment_df["milestone"],
        y=payment_df["cumulative_inr"],
        name="Cumulative",
        mode="lines+markers",
        line=dict(color="#e74c3c", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Contractor Payment Schedule",
        yaxis=dict(title="Payment (₹)", side="left"),
        yaxis2=dict(title="Cumulative (₹)", side="right", overlaying="y"),
        height=420,
    )
    return fig
