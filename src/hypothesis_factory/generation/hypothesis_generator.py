from __future__ import annotations

from hypothesis_factory.analysis.experiment_planner import suggest_minimal_experiment
from hypothesis_factory.generation.templates import ZONE_TITLES
from hypothesis_factory.models import EvidenceClaim, Hypothesis, UncertaintyZone
from hypothesis_factory.scoring.scorer import calculate_uncertainty_importance, final_score

TAILINGS_ZONE_TYPES = {
    "high_extractable_loss",
    "coarse_locked_loss",
    "fine_particle_loss",
    "missing_process_link",
    "expert_unvalidated",
    "tof_aggregation_gap",
}


def _claim_map(claims: list[EvidenceClaim]) -> dict[str, EvidenceClaim]:
    return {claim.id: claim for claim in claims}


def _evidence_for_zone(zone: UncertaintyZone, claims: list[EvidenceClaim]) -> list[dict]:
    by_id = _claim_map(claims)
    ids = [*zone.supporting_claims, *zone.conflicting_claims]
    selected = [by_id[cid] for cid in ids if cid in by_id]
    if not selected:
        selected = [claim for claim in claims if any(ent.lower() in f"{claim.subject} {claim.object}".lower() for ent in zone.linked_entities[:2])][:2]
    return [
        {
            "claim_id": claim.id,
            "source_id": claim.source_id,
            "quote": claim.quote,
            "direction": claim.direction,
            "condition": claim.condition,
        }
        for claim in selected[:4]
    ]


def _scores(zone: UncertaintyZone, evidence_count: int) -> dict[str, float]:
    uncertainty = calculate_uncertainty_importance(zone)
    is_tailings = zone.type in TAILINGS_ZONE_TYPES
    scores = {
        "value": round(0.68 + 0.25 * zone.kpi_relevance, 3) if is_tailings else round(0.62 + 0.25 * zone.kpi_relevance, 3),
        "novelty": round(0.52 + 0.36 * zone.gap_severity, 3),
        "feasibility": 0.76 if is_tailings else 0.72 if zone.type != "kpi_gap" else 0.66,
        "evidence": min(1.0, 0.35 + evidence_count * 0.16),
        "uncertainty_importance": uncertainty,
        "risk": 0.68 if zone.type == "contradiction" else 0.48,
        "cost": 0.42 if is_tailings else 0.38,
    }
    scores["final_score"] = final_score(scores)
    return scores


def _tailings_change(zone: UncertaintyZone) -> str:
    entities = zone.linked_entities
    factory = entities[0] if len(entities) > 0 else "выбранная фабрика"
    element = entities[2] if len(entities) > 2 else "element_28/element_29"
    size_class = entities[3] if len(entities) > 3 else "целевой класс"
    if zone.type == "coarse_locked_loss":
        return (
            f"Для {factory} проверить доизмельчение или перенастройку классификации класса {size_class}, "
            f"где сосредоточены извлекаемые потери {element}."
        )
    if zone.type == "fine_particle_loss":
        return (
            f"Для {factory} проверить режим флотации тонкого класса {size_class}: плотность пульпы, воду, "
            f"время агитации и реагентный режим для снижения потерь {element}."
        )
    if zone.type == "missing_process_link":
        return (
            f"Привязать пик потерь {element} в классе {size_class} к конкретному узлу схемы "
            "и менять только один управляемый параметр оборудования."
        )
    if zone.type == "expert_unvalidated":
        return "Проверить экспертное изменение как A/B-режим против базовой схемы и связать результат с численными потерями."
    return f"Сфокусировать проверку на зоне потерь {factory} / {element} / {size_class}."


def _tailings_mechanism(zone: UncertaintyZone) -> str:
    if zone.type == "coarse_locked_loss":
        return "Механизм: часть элемента остается в грубых или недостаточно раскрытых частицах; доизмельчение и классификация должны повысить доступность для последующей флотации."
    if zone.type == "fine_particle_loss":
        return "Механизм: тонкие частицы хуже сталкиваются и закрепляются на пузырьках; режим пульпы, воды, агитации и реагентов может снизить унос элемента в хвосты."
    if zone.type == "missing_process_link":
        return "Механизм пока не локализован: численный пик потерь известен, но нужно подтвердить, какой узел схемы создает или усиливает эту потерю."
    if zone.type == "expert_unvalidated":
        return "Механизм задан экспертной идеей, но должен быть подтвержден через измеримое снижение потерь в хвостах и сопоставление с классом крупности."
    return "Механизм проверяется через изменение распределения элемента 28/29 между хвостами и концентратом."


def generate_hypotheses(
    zones: list[UncertaintyZone],
    claims: list[EvidenceClaim],
    kpi: str,
    constraints: str = "",
    min_count: int = 5,
) -> list[Hypothesis]:
    hypotheses: list[Hypothesis] = []
    source_zones = zones[:]
    if not source_zones:
        return []
    while len(source_zones) < min_count:
        source_zones.append(source_zones[len(source_zones) % len(zones)])
    for index, zone in enumerate(source_zones[: max(min_count, len(zones[:8]))], start=1):
        evidence = _evidence_for_zone(zone, claims)
        first = zone.linked_entities[0] if zone.linked_entities else "выбранный фактор"
        second = zone.linked_entities[-1] if zone.linked_entities else kpi
        title = f"{ZONE_TITLES.get(zone.type, 'Проверить зону неопределенности')}: {first}"
        change = f"Изменить/зафиксировать фактор {first} в связке {', '.join(zone.linked_entities[1:4]) or second}."
        if zone.type == "contradiction":
            change = f"Развести условия для {first}: концентрация, температура и процесс из конфликтующих источников."
        elif zone.type == "indirect_link":
            change = f"Проверить прямой переход {zone.indirect_path[0]} -> {zone.indirect_path[-1]} из косвенного пути."
        elif zone.type in TAILINGS_ZONE_TYPES:
            change = _tailings_change(zone)
        experiment = suggest_minimal_experiment(zone, constraints)
        scores = _scores(zone, len(evidence))
        is_tailings = zone.type in TAILINGS_ZONE_TYPES
        hypotheses.append(
            Hypothesis(
                id=f"HYP-{index:03d}",
                title=title,
                hypothesis=(
                    f"Кандидат на проверку: {change} Это может улучшить KPI '{kpi}', "
                    f"если зона неопределенности {zone.id} подтвердится экспериментально."
                ),
                target_kpi=kpi,
                origin_uncertainty_zone=zone.to_dict(),
                what_to_change=change,
                why_it_can_affect_kpi=zone.why_it_matters or "Зона связана с целевым KPI через evidence или графовую связь.",
                expected_effect=(
                    "Ожидается снижение массы элемента 28/29 в хвостах при сохранении технологической устойчивости."
                    if is_tailings
                    else "Ожидается рост целевого эффекта при контролируемом риске и проверке ограничений."
                ),
                mechanism=_tailings_mechanism(zone)
                if is_tailings
                else (
                    "Проверяемый механизм: изменение микроструктуры, дефектности, выделений или экономического параметра "
                    "должно проявиться вместе с KPI."
                ),
                evidence=evidence,
                knowledge_gaps=[zone.description] if zone.type in {"coverage_gap", "mechanism_gap", "kpi_gap"} else [],
                contradictions=[zone.description] if zone.type == "contradiction" else [],
                indirect_links=[" -> ".join(zone.indirect_path)] if zone.type == "indirect_link" else [],
                risks=[
                    "Результат может не перенестись с лабораторной проверки на промышленный режим.",
                    "Возможны рост циркуляционной нагрузки, ухудшение селективности или снижение качества концентрата.",
                ],
                minimal_experiment=experiment,
                verification_plan=[
                    "Подготовить базовый и экспериментальный режимы.",
                    "Измерить KPI, механизм и ограничивающие параметры.",
                    "Сравнить результат с критериями успеха и неуспеха.",
                ],
                success_criteria=(
                    "Масса элемента 28/29 в хвостах снижается минимум на 5-10% в целевом классе без ухудшения качества концентрата."
                    if is_tailings
                    else "KPI улучшается минимум на 10-15% без нарушения заданных ограничений."
                ),
                failure_criteria=(
                    "Потери в хвостах не снижаются воспроизводимо, эффект переносит потери в другой класс или ухудшает концентрат."
                    if is_tailings
                    else "Нет воспроизводимого прироста KPI, механизм не подтвержден или стоимость выходит за лимит."
                ),
                scores=scores,
            )
        )
    return sorted(hypotheses, key=lambda item: item.scores["final_score"], reverse=True)
