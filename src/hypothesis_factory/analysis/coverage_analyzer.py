from __future__ import annotations

import itertools
import re

import pandas as pd

from hypothesis_factory.models import Entity, EvidenceClaim, UncertaintyZone


def _kpi_terms(kpi: str) -> set[str]:
    terms = set(re.findall(r"[A-Za-zА-Яа-я+-]+", kpi.lower()))
    if "потер" in kpi.lower() or "хвост" in kpi.lower():
        terms.add("tailings")
        terms.add("loss")
        terms.add("element_28")
        terms.add("element_29")
    return terms


def _claim_relevant(claim: EvidenceClaim, terms: set[str]) -> bool:
    text = f"{claim.subject} {claim.object} {claim.condition or ''}".lower()
    return any(term in text for term in terms)


def build_coverage_matrix(claims: list[EvidenceClaim], entities: list[Entity], kpi: str) -> pd.DataFrame:
    factories = [e.name for e in entities if e.type == "Factory"] or ["КГМК", "НОФ Вкр", "НОФ мед", "ТОФ"]
    elements = [e.name for e in entities if e.type == "Element"] or ["element_28", "element_29"]
    processes = [e.name for e in entities if e.type == "Process"] or ["измельчение", "классификация", "флотация"]
    properties = [e.name for e in entities if e.type == "KPI"] or ["tailings loss"]
    terms = _kpi_terms(kpi)
    rows = []
    for factory, element, process, prop in itertools.product(factories[:4], elements[:6], processes[:5], properties[:7]):
        matching = [
            claim
            for claim in claims
            if element.lower() in claim.subject.lower()
            and (prop.lower() in claim.object.lower() or _claim_relevant(claim, terms))
            and (process.lower() in (claim.condition or "").lower() or process.lower() in claim.subject.lower())
        ]
        direct_count = len(matching)
        if direct_count >= 2:
            status = "well_covered"
        elif direct_count == 1:
            status = "weakly_covered"
        else:
            status = "uncovered"
        rows.append(
            {
                "factory": factory,
                "element": element,
                "process": process,
                "property": prop,
                "target_kpi": kpi,
                "direct_claims": direct_count,
                "status": status,
                "evidence_ids": ", ".join(claim.id for claim in matching),
            }
        )
    return pd.DataFrame(rows)


def find_coverage_gaps(coverage_matrix: pd.DataFrame, kpi: str, constraints: str = "") -> list[UncertaintyZone]:
    if coverage_matrix.empty:
        return []
    candidates = coverage_matrix[
        (coverage_matrix["status"].isin(["uncovered", "weakly_covered"]))
        & (coverage_matrix["element"].str.contains("element_28|element_29|Элемент 28|Элемент 29", case=False, regex=True))
    ].head(8)
    zones: list[UncertaintyZone] = []
    for index, row in enumerate(candidates.to_dict("records"), start=1):
        component = row["element"]
        prop = row["property"]
        process = row["process"]
        status = row["status"]
        zones.append(
            UncertaintyZone(
                id=f"GAP-{index:03d}",
                type="coverage_gap",
                description=f"Недостаточно прямых evidence для связки {component} + {process} -> {prop} при KPI: {kpi}. Статус покрытия: {status}.",
                target_kpi=kpi,
                linked_entities=[row["factory"], component, process, prop],
                supporting_claims=[cid.strip() for cid in str(row.get("evidence_ids", "")).split(",") if cid.strip()],
                source_links=[cid.strip() for cid in str(row.get("evidence_ids", "")).split(",") if cid.strip()],
                why_it_matters="Пробел может скрывать проверяемую технологическую возможность, релевантную KPI и ограничениям.",
                suggested_check=f"Проверить {component} при процессе {process}; сравнить потери в хвостах с базовым режимом.",
                kpi_relevance=0.82 if "loss" in prop.lower() or "потер" in prop.lower() else 0.65,
                gap_severity=0.9 if status == "uncovered" else 0.62,
            )
        )
    return zones
