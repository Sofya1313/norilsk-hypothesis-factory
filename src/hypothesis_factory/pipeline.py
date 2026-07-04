from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hypothesis_factory.analysis.contradiction_analyzer import find_contradictions
from hypothesis_factory.analysis.coverage_analyzer import build_coverage_matrix, find_coverage_gaps
from hypothesis_factory.analysis.gap_analyzer import find_kpi_gaps, find_mechanism_gaps
from hypothesis_factory.analysis.indirect_link_analyzer import find_indirect_links
from hypothesis_factory.analysis.tailings_analyzer import build_tailings_coverage_matrix, find_tailings_uncertainty_zones
from hypothesis_factory.analysis.uncertainty_mapper import build_uncertainty_map
from hypothesis_factory.data_loaders.image_registry import ImageMetadata, build_image_registry
from hypothesis_factory.data_loaders.real_case_loader import (
    ExpertHypothesis,
    TailingsObservation,
    build_real_case_claims,
    build_real_case_documents,
    load_real_case_data,
)
from hypothesis_factory.extraction.claim_extractor import extract_claims
from hypothesis_factory.extraction.entity_extractor import extract_entities
from hypothesis_factory.generation.hypothesis_generator import generate_hypotheses
from hypothesis_factory.graph.graph_builder import build_knowledge_graph
from hypothesis_factory.ingestion.chunking import chunk_documents
from hypothesis_factory.models import Entity, EvidenceClaim, Hypothesis, SourceDocument, TextChunk, UncertaintyZone
from hypothesis_factory.storage import load_demo_documents, save_chunks, save_claims


@dataclass
class PipelineResult:
    documents: list[SourceDocument]
    chunks: list[TextChunk]
    entities: list[Entity]
    claims: list[EvidenceClaim]
    coverage_matrix: object
    graph: object
    zones: list[UncertaintyZone]
    hypotheses: list[Hypothesis]
    tailings_observations: list[TailingsObservation] | None = None
    expert_hypotheses: list[ExpertHypothesis] | None = None
    images: list[ImageMetadata] | None = None


def _use_legacy_pipeline(kpi: str, documents: list[SourceDocument] | None) -> bool:
    if documents:
        return True
    return False


def _run_legacy_pipeline(kpi: str, constraints: str = "", documents: list[SourceDocument] | None = None) -> PipelineResult:
    documents = documents or load_demo_documents()
    chunks = chunk_documents(documents)
    claims = extract_claims(documents)
    entities = extract_entities(documents, claims)
    coverage = build_coverage_matrix(claims, entities, kpi)
    graph = build_knowledge_graph(claims)
    coverage_gaps = find_coverage_gaps(coverage, kpi, constraints)
    contradictions = find_contradictions(claims)
    indirect_links = find_indirect_links(graph, "high-temperature strength", max_path_length=3)
    mechanism_gaps = find_mechanism_gaps(claims, kpi)
    kpi_gaps = find_kpi_gaps(claims, kpi)
    zones = build_uncertainty_map(coverage_gaps, contradictions, indirect_links, mechanism_gaps, kpi_gaps)
    hypotheses = generate_hypotheses(zones, claims, kpi, constraints)
    save_chunks(chunks)
    save_claims(claims)
    return PipelineResult(documents, chunks, entities, claims, coverage, graph, zones, hypotheses)


def _run_real_case_pipeline(kpi: str, constraints: str = "") -> PipelineResult:
    data = load_real_case_data()
    documents = build_real_case_documents(data)
    chunks = chunk_documents(documents)
    claims = build_real_case_claims(data)
    entities = extract_entities(documents, claims)
    coverage = build_tailings_coverage_matrix(data.observations)
    graph = build_knowledge_graph(claims)
    zones = find_tailings_uncertainty_zones(data.observations, data.expert_hypotheses, kpi, constraints)
    hypotheses = generate_hypotheses(zones, claims, kpi, constraints)
    save_chunks(chunks)
    save_claims(claims)
    return PipelineResult(
        documents=documents,
        chunks=chunks,
        entities=entities,
        claims=claims,
        coverage_matrix=coverage if not coverage.empty else pd.DataFrame(),
        graph=graph,
        zones=zones,
        hypotheses=hypotheses,
        tailings_observations=data.observations,
        expert_hypotheses=data.expert_hypotheses,
        images=build_image_registry(),
    )


def run_pipeline(kpi: str, constraints: str = "", documents: list[SourceDocument] | None = None) -> PipelineResult:
    if _use_legacy_pipeline(kpi, documents):
        return _run_legacy_pipeline(kpi, constraints, documents)
    return _run_real_case_pipeline(kpi, constraints)
