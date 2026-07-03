from __future__ import annotations

from hypothesis_factory.models import UncertaintyZone


def score_uncertainty_zone(zone: UncertaintyZone) -> float:
    return round(
        0.35 * zone.kpi_relevance
        + 0.25 * zone.gap_severity
        + 0.20 * zone.contradiction_strength
        + 0.20 * zone.indirect_mechanism_strength,
        3,
    )


def build_uncertainty_map(
    coverage_gaps: list[UncertaintyZone],
    contradictions: list[UncertaintyZone],
    indirect_links: list[UncertaintyZone],
    mechanism_gaps: list[UncertaintyZone] | None = None,
    kpi_gaps: list[UncertaintyZone] | None = None,
) -> list[UncertaintyZone]:
    zones = [
        *coverage_gaps,
        *contradictions,
        *indirect_links,
        *(mechanism_gaps or []),
        *(kpi_gaps or []),
    ]
    for zone in zones:
        zone.priority = score_uncertainty_zone(zone)
    return sorted(zones, key=lambda item: item.priority, reverse=True)

