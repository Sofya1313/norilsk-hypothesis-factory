from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from hypothesis_factory.models import EvidenceClaim


@dataclass
class SimpleGraph:
    adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    edge_attrs: dict[tuple[str, str], dict] = field(default_factory=dict)

    def add_edge(self, source: str, target: str, **attrs) -> None:
        self.adjacency[source].add(target)
        self.adjacency.setdefault(target, set())
        self.edge_attrs[(source, target)] = attrs

    @property
    def nodes(self) -> list[str]:
        return list(self.adjacency.keys())

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.edge_attrs.keys())

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self.edge_attrs

    def shortest_paths_to(self, target: str, max_length: int = 3) -> list[list[str]]:
        paths: list[list[str]] = []
        for start in self.nodes:
            if start == target:
                continue
            queue = deque([[start]])
            while queue:
                path = queue.popleft()
                if len(path) - 1 > max_length:
                    continue
                node = path[-1]
                if node == target and len(path) > 2:
                    paths.append(path)
                    break
                for neighbor in self.adjacency.get(node, set()):
                    if neighbor not in path:
                        queue.append(path + [neighbor])
        return paths


def build_knowledge_graph(claims: Iterable[EvidenceClaim]):
    try:
        import networkx as nx

        graph = nx.DiGraph()
        for claim in claims:
            graph.add_edge(
                claim.subject,
                claim.object,
                relation=claim.relation,
                direction=claim.direction,
                claim_id=claim.id,
                quote=claim.quote,
            )
        return graph
    except Exception:
        graph = SimpleGraph()
        for claim in claims:
            graph.add_edge(
                claim.subject,
                claim.object,
                relation=claim.relation,
                direction=claim.direction,
                claim_id=claim.id,
                quote=claim.quote,
            )
        return graph

