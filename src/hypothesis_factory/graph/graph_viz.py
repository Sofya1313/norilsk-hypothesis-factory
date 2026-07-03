from __future__ import annotations


def graph_to_plotly_figure(graph):
    try:
        import networkx as nx
        import plotly.graph_objects as go

        if not hasattr(graph, "nodes"):
            return None
        positions = nx.spring_layout(graph, seed=7) if hasattr(nx, "spring_layout") else {}
        edge_x, edge_y = [], []
        for source, target in graph.edges:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_x = [positions[node][0] for node in graph.nodes]
        node_y = [positions[node][1] for node in graph.nodes]
        return go.Figure(
            data=[
                go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#9aa4b2"), hoverinfo="none"),
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=list(graph.nodes),
                    textposition="top center",
                    marker=dict(size=13, color="#2563eb"),
                ),
            ],
            layout=go.Layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False),
        )
    except Exception:
        return None


def graph_edges_table(graph) -> list[dict]:
    edges = graph.edges() if callable(getattr(graph, "edges", None)) else graph.edges
    rows = []
    for edge in edges:
        source, target = edge[:2]
        rows.append({"source": source, "target": target})
    return rows

