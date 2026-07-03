from __future__ import annotations

from hypothesis_factory.models import EvidenceClaim, UncertaintyZone


def find_mechanism_gaps(claims: list[EvidenceClaim], kpi: str) -> list[UncertaintyZone]:
    zones: list[UncertaintyZone] = []
    for claim in claims:
        if claim.direction in {"increases", "decreases"} and "through" not in claim.quote.lower() and "за счет" not in claim.quote.lower():
            zones.append(
                UncertaintyZone(
                    id=f"MECH-{len(zones)+1:03d}",
                    type="mechanism_gap",
                    description=f"Эффект {claim.subject} -> {claim.object} описан, но механизм не подтвержден в цитате.",
                    target_kpi=kpi,
                    linked_entities=[claim.subject, claim.object],
                    supporting_claims=[claim.id],
                    source_links=[claim.source_id],
                    why_it_matters="Без проверки механизма гипотеза хуже переносится на другой состав или режим.",
                    suggested_check="Вместе с KPI измерить микроструктуру, фазовый состав и дефектность.",
                    kpi_relevance=0.7,
                    gap_severity=0.6,
                )
            )
    return zones[:3]


def find_kpi_gaps(claims: list[EvidenceClaim], kpi: str) -> list[UncertaintyZone]:
    zones: list[UncertaintyZone] = []
    for claim in claims:
        text = f"{claim.object} {claim.quote}".lower()
        if "cost" in text or "себесто" in text or "scaling" in text:
            zones.append(
                UncertaintyZone(
                    id=f"KPI-{len(zones)+1:03d}",
                    type="kpi_gap",
                    description=f"Есть сигнал по экономике/масштабированию для {claim.subject}, но связь с целевым KPI требует проверки.",
                    target_kpi=kpi,
                    linked_entities=[claim.subject, claim.object],
                    supporting_claims=[claim.id],
                    source_links=[claim.source_id],
                    why_it_matters="Хакатонный KPI обычно включает технический эффект и бизнес-ограничение одновременно.",
                    suggested_check="Добавить расчет себестоимости, энергии, выхода годного и ограничения печи в план проверки.",
                    kpi_relevance=0.78,
                    gap_severity=0.68,
                )
            )
    return zones[:3]

