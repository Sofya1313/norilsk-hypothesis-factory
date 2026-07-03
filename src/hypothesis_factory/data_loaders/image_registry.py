from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hypothesis_factory.config import PROCESSED_DIR, REAL_CASE_ROOT


MANUAL_TAGS = [
    {"keyword": "мельница", "stage": "измельчение", "equipment": "мельница"},
    {"keyword": "классификатор", "stage": "классификация", "equipment": "классификатор"},
    {"keyword": "гидроциклон", "stage": "классификация", "equipment": "гидроциклон"},
    {"keyword": "грохот", "stage": "грохочение", "equipment": "грохот"},
    {"keyword": "флотомашина", "stage": "флотация", "equipment": "флотомашина"},
    {"keyword": "реагент", "stage": "флотация", "equipment": "reagent_regime"},
    {"keyword": "вода", "stage": "флотация", "equipment": "water_addition"},
]


@dataclass
class ImageMetadata:
    file_path: str
    file_name: str
    folder: str
    inferred_type: str
    optional_factory: str | None = None
    optional_stage: str | None = None
    tags: list[dict[str, str]] = field(default_factory=list)
    ocr_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_type(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "типич" in name or "оборуд" in name:
        return "equipment_list"
    if "регламент" in name or "регламент" in parent:
        return "regulation"
    return "scheme"


def _tags_for(path: Path) -> list[dict[str, str]]:
    text = f"{path.name} {path.parent.name}".lower()
    return [tag for tag in MANUAL_TAGS if tag["keyword"].lower() in text]


def build_image_registry(root: Path | None = None, persist: bool = False) -> list[ImageMetadata]:
    root = root or REAL_CASE_ROOT
    if not root.exists():
        return []
    folders = [root / "Регламенты", root / "Схемы флотации"]
    images: list[ImageMetadata] = []
    for folder in folders:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.png")):
            tags = _tags_for(path) or MANUAL_TAGS[:2] if "схема" in path.name.lower() else _tags_for(path)
            images.append(
                ImageMetadata(
                    file_path=str(path),
                    file_name=path.name,
                    folder=path.parent.name,
                    inferred_type=_infer_type(path),
                    optional_stage=(tags[0]["stage"] if tags else None),
                    tags=tags,
                )
            )
    if persist:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "image_metadata.json").write_text(
            json.dumps([item.to_dict() for item in images], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return images
