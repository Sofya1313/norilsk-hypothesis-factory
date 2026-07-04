from __future__ import annotations

from collections import defaultdict

import pandas as pd

from hypothesis_factory.data_loaders.real_case_loader import ExpertHypothesis, TailingsObservation
from hypothesis_factory.models import UncertaintyZone


COARSE_CLASSES = {"+125", "+71", "-125+71"}
FINE_CLASSES = {"-10"}


def _context_preferences(kpi: str, constraints: str = "") -> dict[str, set[str]]:
    text = f"{kpi} {constraints}".lower().replace("ё", "е").replace("_", " ")
    elements: set[str] = set()
    if "элемент 28" in text or "элемента 28" in text or "element 28" in text:
        elements.add("element_28")
    if "элемент 29" in text or "элемента 29" in text or "element 29" in text:
        elements.add("element_29")

    size_focus: set[str] = set()
    if any(term in text for term in ("тонк", "мелк", "-10", "шлам")):
        size_focus.add("fine")
    if any(term in text for term in ("груб", "крупн", "+125", "+71", "доизмельч", "классификац", "раскрыт")):
        size_focus.add("coarse")

    process_focus: set[str] = set()
    if any(term in text for term in ("флотац", "реагент", "пульп", "агитац", "вод")):
        process_focus.add("flotation")
    if any(term in text for term in ("классификац", "доизмельч", "гидроциклон", "раскрыт")):
        process_focus.add("classification")
    if any(term in text for term in ("узл", "схем", "оборудован", "привяз", "вклад")):
        process_focus.add("scheme_link")
    return {"elements": elements, "size": size_focus, "process": process_focus}


def _observation_context_boost(obs: TailingsObservation, zone_type: str, preferences: dict[str, set[str]]) -> float:
    boost = 0.0
    elements = preferences["elements"]
    if elements:
        boost += 0.2 if obs.element in elements else -0.14

    size_focus = preferences["size"]
    is_fine = obs.particle_size_class in FINE_CLASSES or zone_type == "fine_particle_loss"
    is_coarse = obs.particle_size_class in COARSE_CLASSES or zone_type == "coarse_locked_loss"
    if "fine" in size_focus:
        boost += 0.22 if is_fine else -0.1
    if "coarse" in size_focus:
        boost += 0.22 if is_coarse else -0.1

    process_focus = preferences["process"]
    if "flotation" in process_focus:
        boost += 0.1 if is_fine or zone_type == "missing_process_link" else -0.04
    if "classification" in process_focus:
        boost += 0.1 if is_coarse or zone_type == "missing_process_link" else -0.04
    if "scheme_link" in process_focus:
        boost += 0.42 if zone_type == "missing_process_link" else -0.2
    return boost


def _expert_context_boost(item: ExpertHypothesis, preferences: dict[str, set[str]]) -> float:
    text = item.text.lower().replace("ё", "е").replace("_", " ")
    boost = 0.0
    if "element_28" in preferences["elements"] and ("элемент 28" in text or "element 28" in text or "ni" in text):
        boost += 0.15
    if "element_29" in preferences["elements"] and ("элемент 29" in text or "element 29" in text or "cu" in text):
        boost += 0.15
    if "fine" in preferences["size"] and any(term in text for term in ("тонк", "-10", "шлам")):
        boost += 0.12
    if "coarse" in preferences["size"] and any(term in text for term in ("груб", "+125", "+71", "доизмельч", "классификац")):
        boost += 0.12
    return boost


def _clamp(value: float, low: float = 0.2, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _finalize_zones(zones: list[UncertaintyZone]) -> list[UncertaintyZone]:
    for zone in zones:
        zone.priority = round(
            0.5 * zone.kpi_relevance
            + 0.35 * zone.gap_severity
            + 0.15 * max(zone.contradiction_strength, zone.indirect_mechanism_strength),
            3,
        )
    return sorted(zones, key=lambda item: item.priority, reverse=True)


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
    constraints: str = "",
) -> list[UncertaintyZone]:
    zones: list[UncertaintyZone] = []
    preferences = _context_preferences(kpi, constraints)
    distributions = [
        obs for obs in observations if obs.extractable is None and obs.loss_mass_t is not None and obs.particle_size_class
    ]
    extractable = [
        obs for obs in observations if obs.extractable is True and obs.loss_mass_t is not None and obs.particle_size_class
    ]
    def extractable_key(item: TailingsObservation) -> tuple[float, float]:
        zone_type = "coarse_locked_loss" if item.particle_size_class in COARSE_CLASSES else "fine_particle_loss"
        return (_observation_context_boost(item, zone_type, preferences), item.loss_mass_t or 0)

    for obs in sorted(extractable, key=extractable_key, reverse=True)[:8]:
        stage = "измельчение" if obs.particle_size_class in COARSE_CLASSES else "флотация"
        equipment = "гидроциклон" if obs.particle_size_class in COARSE_CLASSES else "флотомашина"
        zone_type = "coarse_locked_loss" if obs.particle_size_class in COARSE_CLASSES else "fine_particle_loss"
        context_boost = _observation_context_boost(obs, zone_type, preferences)
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
                kpi_relevance=_clamp(0.78 + context_boost),
                gap_severity=_clamp(0.55 + (obs.loss_mass_t or 0) / 9000 + max(context_boost, 0) * 0.25),
            )
        )
    for obs in sorted(
        distributions,
        key=lambda item: (_observation_context_boost(item, "missing_process_link", preferences), item.loss_mass_t or 0),
        reverse=True,
    )[:6]:
        context_boost = _observation_context_boost(obs, "missing_process_link", preferences)
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
                kpi_relevance=_clamp(0.72 + context_boost),
                gap_severity=_clamp(0.68 + max(context_boost, 0) * 0.2),
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
    for item in sorted(expert_hypotheses, key=lambda hyp: _expert_context_boost(hyp, preferences), reverse=True)[:8]:
        context_boost = _expert_context_boost(item, preferences)
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
                kpi_relevance=_clamp(0.62 + context_boost),
                gap_severity=_clamp(0.52 + context_boost * 0.4),
            )
        )
    return _finalize_zones(zones)
