from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .config import DB_PATH, DEMO_DIR
from .models import EvidenceClaim, SourceDocument, TextChunk


def load_demo_documents(path: Path | None = None) -> list[SourceDocument]:
    raw = json.loads((path or DEMO_DIR / "demo_documents.json").read_text(encoding="utf-8"))
    return [
        SourceDocument(
            id=item["id"],
            title=item["title"],
            text=item["text"],
            source_type=item.get("source_type", "demo"),
            language=item.get("language", "ru"),
            url=item.get("url"),
        )
        for item in raw
    ]


def load_demo_claims(path: Path | None = None) -> list[EvidenceClaim]:
    raw = json.loads((path or DEMO_DIR / "demo_documents.json").read_text(encoding="utf-8"))
    claims: list[EvidenceClaim] = []
    for item in raw:
        for index, claim in enumerate(item.get("claims", []), start=1):
            claims.append(
                EvidenceClaim(
                    id=f"{item['id']}-C{index:02d}",
                    subject=claim["subject"],
                    relation=claim.get("relation", "affects"),
                    object=claim["object"],
                    direction=claim.get("direction", "unknown"),
                    magnitude=claim.get("magnitude"),
                    condition=claim.get("condition"),
                    source_id=item["id"],
                    quote=claim.get("quote", item["text"]),
                    confidence=float(claim.get("confidence", 0.82)),
                )
            )
    return claims


def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source_id TEXT, position INTEGER, text TEXT)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, source_id TEXT, subject TEXT, relation TEXT, object TEXT, direction TEXT, quote TEXT)"
    )
    con.commit()
    return con


def save_chunks(chunks: Iterable[TextChunk], path: Path = DB_PATH) -> None:
    con = init_db(path)
    con.executemany(
        "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)",
        [(c.id, c.source_id, c.position, c.text) for c in chunks],
    )
    con.commit()
    con.close()


def save_claims(claims: Iterable[EvidenceClaim], path: Path = DB_PATH) -> None:
    con = init_db(path)
    con.executemany(
        "INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(c.id, c.source_id, c.subject, c.relation, c.object, c.direction, c.quote) for c in claims],
    )
    con.commit()
    con.close()

