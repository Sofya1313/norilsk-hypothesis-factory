from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class SourceDocument:
    id: str
    title: str
    text: str
    source_type: str = "demo"
    language: str = "ru"
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextChunk:
    id: str
    source_id: str
    text: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Entity:
    id: str
    name: str
    type: Literal[
        "Material",
        "Component",
        "Process",
        "Parameter",
        "Property",
        "KPI",
        "Experiment",
        "Source",
        "Factory",
        "TailingsType",
        "Element",
        "ParticleSizeClass",
        "MineralForm",
        "Equipment",
    ]
    source_id: str | None = None
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceClaim:
    id: str
    subject: str
    relation: str
    object: str
    direction: Literal["increases", "decreases", "affects", "no_effect", "unknown"]
    magnitude: str | None = None
    condition: str | None = None
    source_id: str = ""
    quote: str = ""
    confidence: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyZone:
    id: str
    type: Literal[
        "coverage_gap",
        "contradiction",
        "indirect_link",
        "mechanism_gap",
        "kpi_gap",
        "high_extractable_loss",
        "coarse_locked_loss",
        "fine_particle_loss",
        "missing_process_link",
        "expert_unvalidated",
        "tof_aggregation_gap",
    ]
    description: str
    target_kpi: str
    linked_entities: list[str] = field(default_factory=list)
    supporting_claims: list[str] = field(default_factory=list)
    conflicting_claims: list[str] = field(default_factory=list)
    indirect_path: list[str] = field(default_factory=list)
    source_links: list[str] = field(default_factory=list)
    why_it_matters: str = ""
    suggested_check: str = ""
    kpi_relevance: float = 0.5
    gap_severity: float = 0.5
    contradiction_strength: float = 0.0
    indirect_mechanism_strength: float = 0.0
    priority: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    id: str
    title: str
    hypothesis: str
    target_kpi: str
    origin_uncertainty_zone: dict[str, Any]
    what_to_change: str
    why_it_can_affect_kpi: str
    expected_effect: str
    mechanism: str
    evidence: list[dict[str, Any]]
    knowledge_gaps: list[str]
    contradictions: list[str]
    indirect_links: list[str]
    risks: list[str]
    minimal_experiment: list[str]
    verification_plan: list[str]
    success_criteria: str
    failure_criteria: str
    scores: dict[str, float]
    expert_feedback: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
