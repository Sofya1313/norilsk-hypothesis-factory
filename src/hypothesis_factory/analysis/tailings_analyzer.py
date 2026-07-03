from __future__ import annotations

from collections import defaultdict

import pandas as pd

from hypothesis_factory.data_loaders.real_case_loader import ExpertHypothesis, TailingsObservation
from hypothesis_factory.models import UncertaintyZone


COARSE_CLASSES = {"+125", "+71", "-125+71"}
FINE_CLASSES = {"-10"}


def build_tailings_coverage_matrix(observations: list[TailingsObservation]) -> pd.DataFrame:
    rows = []
    grouped: dict[tuple[str, str, str, str], list[TailingsObservation]] = defaultdict(list)
    for obs in observations:
        if obs.particle_size_class:
            grouped[(obs.factory, obs.tailings_type, obs.element, obs.particle_size_class)].append(obs)
    for (factory, tailings_type, element, size_class), items in grouped.items():
        has_loss = any(item.loss_mass_t is not None for item in items)
        has_extractability = any(item.extractable is not None for item in items)
        status = "well_covered" if has_loss and has_extractability else "weakly_covered" if has_loss else "uncovered"
        rows.append(
            {
                "factory": factory,
                "tailings_type": tailings_type,
                "element": element,
                "particle_size_class": size_class,
                "loss_mass_t": max((item.loss_mass_t or 0 for item in items), default=0),
                "loss_share_pct": max((item.loss_share_pct or 0 for item in items), default=0),
                "has_extractability": has_extractability,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _zone_id(prefix: str, zones: list[UncertaintyZone]) -> str:
    return f"{prefix}-{len(zones)+1:03d}"


def find_tailings_uncertainty_zones(
    observations: list[TailingsObservation],
    expert_hypotheses: list[ExpertHypothesis],
    kpi: str,
) -> list[UncertaintyZone]:
    zones: list[UncertaintyZone] = []
    distributions = [
        obs for obs in observations if obs.extractable is None and obs.loss_mass_t is not None and obs.particle_size_class
    ]
    extractable = [
        obs for obs in observations if obs.extractable is True and obs.loss_mass_t is not None and obs.particle_size_class
    ]
    for obs in sorted(extractable, key=lambda item: item.loss_mass_t or 0, reverse=True)[:8]:
        stage = "измельчение" if obs.particle_size_class in COARSE_CLASSES else "флотация"
        equipment = "гидроциклон" if obs.particle_size_class in COARSE_CLASSES else "флотомашина"
        zone_type = "coarse_locked_loss" if obs.particle_size_class in COARSE_CLASSES else "fine_particle_loss"
        zones.append(
            UncertaintyZone(
                id=_zone_id("TAIL", zones),
                type=zone_type,
                description=(
                    f"{obs.factory}: высокий извлекаемый металл {obs.element} в хвостах "
                    f"{obs.tailings_type}, класс {obs.particle_size_class}: {obs.loss_mass_t:.1f} т."
                ),
                target_kpi=kpi,
                linked_entities=[obs.factory, obs.tailings_type, obs.element, obs.particle_size_class or "", stage, equipment],
                supporting_claims=[],
                source_links=[obs.source_file, obs.row_ref or ""],
                why_it_matters="Это технологически доступная часть потерь: ее можно пытаться вернуть изменением классификации, измельчения или флотации.",
                suggested_check=f"Проверить режим {stage} / {equipment} на классе {obs.particle_size_class} с измерением потерь {obs.element}.",
                kpi_relevance=0.95,
                gap_severity=min(1.0, 0.55 + (obs.loss_mass_t or 0) / 9000),
            )
        )
    for obs in sorted(distributions, key=lambda item: item.loss_mass_t or 0, reverse=True)[:6]:
        zones.append(
            UncertaintyZone(
                id=_zone_id("LINK", zones),
                type="missing_process_link",
                description=(
                    f"{obs.factory}: есть численный пик потерь {obs.element} в классе "
                    f"{obs.particle_size_class}, но нет явной привязки к конкретному режиму оборудования."
                ),
                target_kpi=kpi,
                linked_entities=[obs.factory, obs.element, obs.particle_size_class or "", "process_stage", "equipment"],
                source_links=[obs.source_file, obs.row_ref or ""],
                why_it_matters="Без связи с узлом схемы гипотеза остается слишком общей; это главный пробел для проверки на фабрике.",
                suggested_check="Привязать пик потерь к участку схемы: классификация, измельчение, основная или контрольная флотация.",
                kpi_relevance=0.88,
                gap_severity=0.78,
            )
        )
    by_factory_element: dict[tuple[str, str], list[TailingsObservation]] = defaultdict(list)
    for obs in distributions:
        by_factory_element[(obs.factory, obs.element)].append(obs)
    for (factory, element), items in by_factory_element.items():
        top = sorted(items, key=lambda item: item.loss_share_pct or 0, reverse=True)[:2]
        if len(top) == 2 and top[0].particle_size_class != top[1].particle_size_class:
            zones.append(
                UncertaintyZone(
                    id=_zone_id("CON", zones),
                    type="contradiction",
                    description=(
                        f"Potential contradiction / requires validation: для {factory} и {element} "
                        f"приоритеты распределены между {top[0].particle_size_class} и {top[1].particle_size_class}; "
                        "нужна проверка, какой механизм доминирует."
                    ),
                    target_kpi=kpi,
                    linked_entities=[factory, element, top[0].particle_size_class or "", top[1].particle_size_class or ""],
                    source_links=[top[0].source_file, top[1].source_file],
                    why_it_matters="Разные максимумы потерь требуют разных вмешательств: доизмельчения/классификации или тонкой флотации.",
                    suggested_check="Сравнить две зоны потерь в одном минимальном плане испытаний и выбрать доминирующий механизм.",
                    kpi_relevance=0.8,
                    gap_severity=0.6,
                    contradiction_strength=0.72,
                )
            )
    for item in expert_hypotheses[:8]:
        zones.append(
            UncertaintyZone(
                id=_zone_id("EXP", zones),
                type="expert_unvalidated",
                description=f"{item.factory}: экспертная гипотеза требует проверки на численных потерях: {item.text}",
                target_kpi=kpi,
                linked_entities=[item.factory, "expert_brainstorm"],
                source_links=[item.source_file],
                why_it_matters="Экспертная идея полезна как ориентир, но должна быть проверена через потери по классам и минимальный эксперимент.",
                suggested_check="Сопоставить экспертную идею с top-loss классами и проверить на малом технологическом окне.",
                kpi_relevance=0.76,
                gap_severity=0.58,
            )
        )
    return zones
