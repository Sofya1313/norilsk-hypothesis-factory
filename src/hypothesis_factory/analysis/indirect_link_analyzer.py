from __future__ import annotations

from hypothesis_factory.models import UncertaintyZone


def _nodes(graph):
    return list(graph.nodes() if callable(getattr(graph, "nodes", None)) else graph.nodes)


def _edges(graph):
    return list(graph.edges() if callable(getattr(graph, "edges", None)) else graph.edges)


def _has_edge(graph, source: str, target: str) -> bool:
    if hasattr(graph, "has_edge"):
        return bool(graph.has_edge(source, target))
    return (source, target) in _edges(graph)


def _paths(graph, target: str, max_path_length: int) -> list[list[str]]:
    if hasattr(graph, "shortest_paths_to"):
        return graph.shortest_paths_to(target, max_path_length)
    try:
        import networkx as nx

        paths = []
        for node in _nodes(graph):
            if node == target:
                continue
            try:
                path = nx.shortest_path(graph, node, target)
            except Exception:
                continue
            if 2 <= len(path) - 1 <= max_path_length:
                paths.append(path)
        return paths
    except Exception:
        return []


def find_indirect_links(graph, kpi_node: str, max_path_length: int = 3) -> list[UncertaintyZone]:
    nodes = _nodes(graph)
    target = next((node for node in nodes if kpi_node.lower() in node.lower()), kpi_node)
    zones: list[UncertaintyZone] = []
    for path in _paths(graph, target, max_path_length=max_path_length):
        if len(path) < 3:
            continue
        if _has_edge(graph, path[0], path[-1]):
            continue
        zones.append(
            UncertaintyZone(
                id=f"IND-{len(zones)+1:03d}",
                type="indirect_link",
                description=f"Есть косвенный путь {' -> '.join(path)}, но нет прямого claim {path[0]} -> {path[-1]}.",
                target_kpi=target,
                linked_entities=path,
                indirect_path=path,
                source_links=[],
                why_it_matters="Косвенная связь дает проверяемый механизм, но требует прямого эксперимента.",
                suggested_check=f"Проверить прямое влияние {path[0]} на {path[-1]} в пределах найденного механизма.",
                kpi_relevance=0.82,
                gap_severity=0.62,
                indirect_mechanism_strength=min(1.0, 0.35 + 0.18 * len(path)),
            )
        )
    return zones[:6]

