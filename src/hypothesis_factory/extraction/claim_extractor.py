from __future__ import annotations

import re

from hypothesis_factory.models import EvidenceClaim, SourceDocument
from hypothesis_factory.storage import load_demo_claims


def extract_claims(documents: list[SourceDocument], prefer_demo_claims: bool = True) -> list[EvidenceClaim]:
    demo_ids = {doc.id for doc in documents if doc.id.startswith("SRC-")}
    if prefer_demo_claims and demo_ids:
        return [claim for claim in load_demo_claims() if claim.source_id in demo_ids]

    claims: list[EvidenceClaim] = []
    patterns = [
        (r"(?P<subject>element_28|element_29|элемент 28|элемент 29).*?(потер|хвост).*?(?P<object>класс\s*[+-]?\d+|tailings|хвост)", "affects"),
        (r"(?P<subject>флотац|классификац|измельч).*?(сниж|reduce).*?(?P<object>потер|tailings loss)", "decreases"),
        (r"(?P<subject>гидроциклон|мельниц|флотомашин).*?(влия|affect).*?(?P<object>потер|tailings loss)", "affects"),
    ]
    for doc in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", doc.text):
            for pattern, direction in patterns:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    claims.append(
                        EvidenceClaim(
                            id=f"{doc.id}-RC{len(claims)+1:02d}",
                            subject=match.group("subject"),
                            relation="affects",
                            object=match.group("object"),
                            direction=direction,
                            source_id=doc.id,
                            quote=sentence.strip(),
                            confidence=0.58,
                        )
                    )
    return claims
