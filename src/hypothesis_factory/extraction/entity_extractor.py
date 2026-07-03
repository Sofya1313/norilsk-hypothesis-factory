from __future__ import annotations

import re

from hypothesis_factory.models import Entity, EvidenceClaim, SourceDocument

ENTITY_RULES = {
    "Factory": ["КГМК", "НОФ Вкр", "НОФ мед", "ТОФ"],
    "TailingsType": ["Хвосты породные", "Хвосты пирротиновые", "Хвосты отвальные", "хвосты"],
    "Element": ["element_28", "element_29", "Элемент 28", "Элемент 29", "Ni", "Cu"],
    "ParticleSizeClass": ["+125", "+71", "-125+71", "-71+45", "-45+20", "-20+10", "-10"],
    "Process": [
        "дробление",
        "измельчение",
        "классификация",
        "флотация",
        "контрольная флотация",
        "перечистка",
        "сгущение",
    ],
    "Equipment": ["мельница", "классификатор", "гидроциклон", "грохот", "флотомашина", "насос", "зумпф"],
    "KPI": ["KPI", "потери", "хвосты", "снижение потерь", "извлекаемый металл"],
}


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def extract_entities(documents: list[SourceDocument], claims: list[EvidenceClaim] | None = None) -> list[Entity]:
    found: dict[tuple[str, str], Entity] = {}
    texts = [doc.text for doc in documents]
    if claims:
        texts.extend([f"{c.subject} {c.object} {c.condition or ''}" for c in claims])
    corpus = "\n".join(texts).lower()
    for entity_type, names in ENTITY_RULES.items():
        for name in names:
            if name.lower() in corpus:
                key = (entity_type, _normalize(name))
                found[key] = Entity(id=f"{entity_type[:3].upper()}-{len(found)+1:03d}", name=_normalize(name), type=entity_type)
    if claims:
        for claim in claims:
            for entity_type, value in [("Element", claim.subject), ("KPI", claim.object)]:
                key = (entity_type, _normalize(value))
                found.setdefault(key, Entity(id=f"{entity_type[:3].upper()}-{len(found)+1:03d}", name=_normalize(value), type=entity_type))
    return sorted(found.values(), key=lambda e: (e.type, e.name))
