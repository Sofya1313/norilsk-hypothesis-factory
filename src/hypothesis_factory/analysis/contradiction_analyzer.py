from __future__ import annotations

import re

from hypothesis_factory.models import EvidenceClaim, UncertaintyZone

OPPOSITES = {
    "increases": {"decreases", "no_effect"},
    "decreases": {"increases", "no_effect"},
    "no_effect": {"increases", "decreases"},
}


def _norm(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9+]+", " ", value.lower()).strip()


def _family(value: str) -> str:
    value = _norm(value)
    if "tailings" in value or "loss" in value or "хвост" in value or "потер" in value:
        return "tailings loss"
    if "priority" in value or "intervention" in value or "приоритет" in value:
        return "priority intervention zone"
    return value


def compare_conditions(claim_a: EvidenceClaim, claim_b: EvidenceClaim) -> dict:
    condition_a = claim_a.condition or "not specified"
    condition_b = claim_b.condition or "not specified"
    hints = []
    for token in ["+125", "+71", "-125+71", "-71+45", "-45+20", "-20+10", "-10", "флотация", "классификация", "измельчение"]:
        if token in condition_a.lower() or token in condition_b.lower():
            hints.append(token)
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "possible_drivers": ", ".join(sorted(set(hints))) or "temperature, concentration, material, process, method",
    }


def find_contradictions(claims: list[EvidenceClaim]) -> list[UncertaintyZone]:
    zones: list[UncertaintyZone] = []
    for i, left in enumerate(claims):
        for right in claims[i + 1 :]:
            same_subject = _norm(left.subject) == _norm(right.subject)
            same_object = _family(left.object) == _family(right.object)
            if not (same_subject and same_object):
                continue
            if right.direction in OPPOSITES.get(left.direction, set()):
                diff = compare_conditions(left, right)
                zones.append(
                    UncertaintyZone(
                        id=f"CON-{len(zones)+1:03d}",
                        type="contradiction",
                        description=(
                            f"Противоречивые claims для {left.subject} -> {_family(left.object)}: "
                            f"{left.direction} против {right.direction}. Отличия условий: {diff['possible_drivers']}."
                        ),
                        target_kpi=_family(left.object),
                        linked_entities=[left.subject, _family(left.object)],
                        supporting_claims=[left.id],
                        conflicting_claims=[right.id],
                        source_links=[left.source_id, right.source_id],
                        why_it_matters="Противоречие можно превратить в проверяемый эксперимент вместо усреднения выводов.",
                        suggested_check="Развести условия конфликтующих источников: концентрация, температура, процесс и методика испытаний.",
                        kpi_relevance=0.86,
                        gap_severity=0.55,
                        contradiction_strength=0.95,
                    )
                )
    return zones
