"""
Metrics Report Generator
Creates Plotly visualizations and exports for RAG evaluation results.
"""

import json
import csv
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_radar_chart(metrics: dict) -> go.Figure:
    """Create a radar/spider chart of aggregate metrics."""
    categories = [
        "Precision@K",
        "Context\nRelevance",
        "Answer\nRelevance",
        "Faithfulness",
        "Chunk\nUtilization",
    ]
    values_keys = [
        "avg_precision_at_k",
        "avg_context_relevance",
        "avg_answer_relevance",
        "avg_faithfulness",
        "avg_chunk_utilization",
    ]

    values = [metrics.get(k, 0) or 0 for k in values_keys]
    values.append(values[0])  # Close the radar
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.25)',
        line=dict(color='rgb(99, 102, 241)', width=2),
        marker=dict(size=8, color='rgb(99, 102, 241)'),
        name='RAG Metrics',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10, color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)',
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#e2e8f0'),
                gridcolor='rgba(148, 163, 184, 0.2)',
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title=dict(text="RAG Performance Radar", font=dict(size=16, color='#f1f5f9')),
        margin=dict(l=80, r=80, t=60, b=40),
        height=450,
    )
    return fig


def create_metrics_bar_chart(metrics: dict) -> go.Figure:
    """Create a horizontal bar chart of all metrics."""
    labels = [
        "Precision@K",
        "Context Relevance",
        "Answer Relevance",
        "Faithfulness",
        "Chunk Utilization",
    ]
    keys = [
        "avg_precision_at_k",
        "avg_context_relevance",
        "avg_answer_relevance",
        "avg_faithfulness",
        "avg_chunk_utilization",
    ]
    values = [metrics.get(k, 0) or 0 for k in keys]

    colors = [
        '#6366f1', '#8b5cf6', '#a78bfa', '#c084fc',
        '#e879f9', '#f472b6', '#fb7185'
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=4,
        ),
        text=[f"{v:.2%}" for v in values],
        textposition='auto',
        textfont=dict(color='white', size=12),
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 1],
            title="Score",
            gridcolor='rgba(148, 163, 184, 0.15)',
            tickfont=dict(color='#94a3b8'),
        ),
        yaxis=dict(
            tickfont=dict(color='#e2e8f0', size=12),
            autorange='reversed',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title=dict(text="Metric Scores", font=dict(size=16, color='#f1f5f9')),
        margin=dict(l=140, r=40, t=60, b=40),
        height=400,
    )
    return fig


def create_latency_chart(metrics: dict) -> go.Figure:
    """Create a stacked bar showing retrieval vs generation latency."""
    per_query = metrics.get("per_query", [])
    if not per_query:
        return go.Figure()

    questions = [f"Q{i+1}" for i in range(len(per_query))]
    retrieval = [q["latency_retrieval"] for q in per_query]
    generation = [q["latency_generation"] for q in per_query]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Retrieval',
        x=questions,
        y=retrieval,
        marker_color='#6366f1',
        marker_cornerradius=2,
    ))
    fig.add_trace(go.Bar(
        name='Generation',
        x=questions,
        y=generation,
        marker_color='#f472b6',
        marker_cornerradius=2,
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(title="Query", tickfont=dict(color='#94a3b8')),
        yaxis=dict(title="Time (seconds)", gridcolor='rgba(148, 163, 184, 0.15)', tickfont=dict(color='#94a3b8')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title=dict(text="Latency Breakdown per Query", font=dict(size=16, color='#f1f5f9')),
        legend=dict(font=dict(color='#e2e8f0')),
        margin=dict(l=60, r=40, t=60, b=40),
        height=400,
    )
    return fig


def create_retrieval_heatmap(metrics: dict) -> go.Figure:
    """Create a heatmap of retrieval scores per query per chunk."""
    per_query = metrics.get("per_query", [])
    if not per_query:
        return go.Figure()

    max_sources = max(len(q.get("sources", [])) for q in per_query)
    questions = [f"Q{i+1}" for i in range(len(per_query))]

    z = []
    hover_text = []
    for q in per_query:
        row = []
        hover_row = []
        for j in range(max_sources):
            if j < len(q.get("sources", [])):
                score = q["sources"][j]["score"]
                row.append(score)
                text_preview = q["sources"][j]["text"][:60] + "..."
                hover_row.append(f"Score: {score:.3f}<br>{text_preview}")
            else:
                row.append(0)
                hover_row.append("No chunk")
        z.append(row)
        hover_text.append(hover_row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"Chunk {j+1}" for j in range(max_sources)],
        y=questions,
        colorscale=[
            [0, '#1e1b4b'],
            [0.25, '#312e81'],
            [0.5, '#4f46e5'],
            [0.75, '#818cf8'],
            [1.0, '#c7d2fe'],
        ],
        hovertext=hover_text,
        hoverinfo='text',
        colorbar=dict(title="Score", tickfont=dict(color='#94a3b8')),
    ))

    fig.update_layout(
        xaxis=dict(title="Retrieved Chunks", tickfont=dict(color='#94a3b8')),
        yaxis=dict(tickfont=dict(color='#e2e8f0'), autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title=dict(text="Retrieval Score Heatmap", font=dict(size=16, color='#f1f5f9')),
        margin=dict(l=60, r=40, t=60, b=40),
        height=400,
    )
    return fig


def create_per_query_comparison(metrics: dict) -> go.Figure:
    """Line chart comparing per-query metrics."""
    per_query = metrics.get("per_query", [])
    if not per_query:
        return go.Figure()

    questions = [f"Q{i+1}" for i in range(len(per_query))]

    fig = go.Figure()
    traces = [
        ("precision_at_k", "Precision@K", '#6366f1'),
        ("context_relevance", "Context Relevance", '#8b5cf6'),
        ("answer_relevance", "Answer Relevance", '#a78bfa'),
        ("faithfulness", "Faithfulness", '#e879f9'),
        ("chunk_utilization", "Chunk Utilization", '#f472b6'),
    ]

    for key, name, color in traces:
        values = [q.get(key, 0) or 0 for q in per_query]
        fig.add_trace(go.Scatter(
            x=questions, y=values,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        xaxis=dict(title="Query", tickfont=dict(color='#94a3b8')),
        yaxis=dict(
            title="Score", range=[0, 1.05],
            gridcolor='rgba(148, 163, 184, 0.15)',
            tickfont=dict(color='#94a3b8'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title=dict(text="Per-Query Metric Comparison", font=dict(size=16, color='#f1f5f9')),
        legend=dict(font=dict(color='#e2e8f0'), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=40, t=60, b=40),
        height=400,
    )
    return fig


def export_json(metrics: dict, path: str | Path) -> str:
    """Export metrics report to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove non-serializable data
    export_data = {k: v for k, v in metrics.items() if k != "per_query"}
    if "per_query" in metrics:
        export_data["per_query"] = []
        for q in metrics["per_query"]:
            q_clean = {k: v for k, v in q.items() if k != "sources"}
            export_data["per_query"].append(q_clean)

    with open(path, "w") as f:
        json.dump(export_data, f, indent=2)
    return str(path)


def export_csv(metrics: dict, path: str | Path) -> str:
    """Export per-query metrics to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    per_query = metrics.get("per_query", [])
    if not per_query:
        return str(path)

    fieldnames = [
        "question", "answer", "reference_answer",
        "precision_at_k", "context_relevance", "answer_relevance",
        "faithfulness", "chunk_utilization",
        "latency_retrieval", "latency_generation", "total_latency",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for q in per_query:
            writer.writerow(q)

    return str(path)
