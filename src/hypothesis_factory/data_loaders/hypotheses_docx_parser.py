from __future__ import annotations

import re
from pathlib import Path

from hypothesis_factory.config import REAL_CASE_ROOT


FACTORY_BY_FOLDER = {
    "Пример 1": "КГМК",
    "Пример 2": "НОФ Вкр",
    "Пример 3": "НОФ мед",
    "Пример 4": "ТОФ",
}


def _clean_numbered_item(text: str) -> str:
    return re.sub(r"^\s*\d+[\).\s-]+", "", text).strip(" \t\r\n-")


def parse_expert_hypotheses_docx(path: Path, factory: str) -> list[dict]:
    try:
        from docx import Document
    except Exception:
        return []

    try:
        document = Document(path)
    except Exception:
        return []

    hypotheses: list[dict] = []
    for paragraph in document.paragraphs:
        text = _clean_numbered_item(paragraph.text)
        if len(text) < 12:
            continue
        if text.lower().startswith(("гипотез", "вариант", "№")):
            continue
        hypotheses.append(
            {
                "factory": factory,
                "text": text,
                "source_file": str(path),
                "index": len(hypotheses) + 1,
            }
        )
    return hypotheses


def parse_expert_hypotheses_folder(root: Path | None = None) -> list[dict]:
    root = root or REAL_CASE_ROOT
    if not root.exists():
        return []
    items: list[dict] = []
    for folder_name, factory in FACTORY_BY_FOLDER.items():
        folder = root / folder_name
        if not folder.exists():
            continue
        for path in folder.glob("Гипотезы*.docx"):
            items.extend(parse_expert_hypotheses_docx(path, factory))
    return items
