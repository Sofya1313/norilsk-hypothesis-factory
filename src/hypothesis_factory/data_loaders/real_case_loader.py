from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypothesis_factory.config import REAL_CASE_KNOWLEDGE_BASE_PATH, REAL_CASE_SUMMARY_PATH
from hypothesis_factory.data_loaders.hypotheses_docx_parser import parse_expert_hypotheses_folder
from hypothesis_factory.models import EvidenceClaim, SourceDocument


ELEMENT_LABELS = {
    "element_28": "Элемент 28 (Ni)",
    "element_29": "Элемент 29 (Cu)",
}


@dataclass
class TailingsObservation:
    source_file: str
    factory: str
    tailings_type: str
    element: str
    particle_size_class: str | None = None
    mineral_form: str | None = None
    loss_share_pct: float | None = None
    loss_mass_t: float | None = None
    class_share_pct: float | None = None
    extractable: bool | None = None
    row_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "factory": self.factory,
            "tailings_type": self.tailings_type,
            "element": self.element,
            "particle_size_class": self.particle_size_class,
            "mineral_form": self.mineral_form,
            "loss_share_pct": self.loss_share_pct,
            "loss_mass_t": self.loss_mass_t,
            "class_share_pct": self.class_share_pct,
            "extractable": self.extractable,
            "row_ref": self.row_ref,
        }


@dataclass
class ExpertHypothesis:
    factory: str
    text: str
    source_file: str
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "factory": self.factory,
            "text": self.text,
            "source_file": self.source_file,
            "index": self.index,
        }


@dataclass
class RealCaseData:
    summary: dict[str, Any]
    observations: list[TailingsObservation] = field(default_factory=list)
    expert_hypotheses: list[ExpertHypothesis] = field(default_factory=list)


def load_real_case_summary(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or REAL_CASE_SUMMARY_PATH).read_text(encoding="utf-8"))


def load_real_case_knowledge_base(path: Path | None = None) -> str:
    file_path = path or REAL_CASE_KNOWLEDGE_BASE_PATH
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _observation_from_fact(example: dict[str, Any], fact: dict[str, Any], element: str) -> TailingsObservation:
    return TailingsObservation(
        source_file=example.get("tailings_file", ""),
        factory=example.get("factory", ""),
        tailings_type=fact.get("tailings_type", "Хвосты отвальные"),
        element=element,
        particle_size_class=fact.get("class"),
        loss_share_pct=_as_float(fact.get(f"{element}_share_pct") or fact.get(f"{element}_pct")),
        loss_mass_t=_as_float(fact.get(f"{element}_t")),
        class_share_pct=_as_float(fact.get("class_share_pct")),
        extractable=_extractable_flag(fact),
        row_ref=f"row {fact.get('row')}" if fact.get("row") else None,
    )


def _extractable_flag(fact: dict[str, Any]) -> bool | None:
    if fact.get("type") != "extractability_by_class":
        return None
    kind = str(fact.get("kind", "")).lower()
    if "не извлека" in kind:
        return False
    if "извлека" in kind:
        return True
    return None


def build_tailings_observations(summary: dict[str, Any]) -> list[TailingsObservation]:
    observations: list[TailingsObservation] = []
    for example in summary.get("examples", []):
        for fact in example.get("facts", []):
            if fact.get("type") not in {"class_distribution", "extractability_by_class", "extractability_total"}:
                continue
            for element in ("element_28", "element_29"):
                if fact.get(f"{element}_t") is not None or fact.get(f"{element}_share_pct") is not None:
                    observations.append(_observation_from_fact(example, fact, element))
    return observations


def load_real_case_data() -> RealCaseData:
    summary = load_real_case_summary()
    observations = build_tailings_observations(summary)
    expert_hypotheses = [
        ExpertHypothesis(**item) for item in parse_expert_hypotheses_folder()
    ]
    return RealCaseData(summary=summary, observations=observations, expert_hypotheses=expert_hypotheses)


def build_real_case_documents(data: RealCaseData) -> list[SourceDocument]:
    docs: list[SourceDocument] = [
        SourceDocument(
            id="REAL-KB",
            title="Реальная база знаний: флотационные хвосты",
            text=load_real_case_knowledge_base()[:12000],
            source_type="knowledge_base",
        )
    ]
    for example in data.summary.get("examples", []):
        factory = example.get("factory", "")
        facts = [obs for obs in data.observations if obs.factory == factory and obs.extractable is None]
        top = sorted(
            [obs for obs in facts if obs.loss_mass_t is not None],
            key=lambda obs: obs.loss_mass_t or 0,
            reverse=True,
        )[:8]
        lines = [
            f"{factory}: отчет по хвостам из файла {example.get('tailings_file')}.",
            "Ключевые зоны потерь:",
        ]
        for obs in top:
            lines.append(
                f"{ELEMENT_LABELS.get(obs.element, obs.element)}; {obs.tailings_type}; "
                f"класс {obs.particle_size_class}; потери {obs.loss_mass_t:.1f} т "
                f"({obs.loss_share_pct:.1f}% от потерь элемента)."
            )
        docs.append(
            SourceDocument(
                id=f"REAL-{factory}",
                title=f"{factory}: численные паттерны хвостов",
                text="\n".join(lines),
                source_type="tailings_xlsx",
            )
        )
    if data.expert_hypotheses:
        docs.append(
            SourceDocument(
                id="REAL-EXPERT",
                title="Экспертные гипотезы по фабрикам",
                text="\n".join(f"{item.factory}: {item.text}" for item in data.expert_hypotheses),
                source_type="expert_brainstorm",
            )
        )
    return docs


def build_real_case_claims(data: RealCaseData) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    by_mass = sorted(
        [obs for obs in data.observations if obs.extractable is None and obs.loss_mass_t is not None],
        key=lambda obs: obs.loss_mass_t or 0,
        reverse=True,
    )
    for index, obs in enumerate(by_mass, start=1):
        claims.append(
            EvidenceClaim(
                id=f"TAIL-{index:03d}",
                subject=f"{obs.factory} {obs.element} {obs.particle_size_class}",
                relation="has_tailings_loss",
                object=f"{obs.element} loss in {obs.tailings_type}",
                direction="affects",
                magnitude=f"{obs.loss_mass_t:.1f} т; {obs.loss_share_pct:.1f}%" if obs.loss_share_pct is not None else None,
                condition=f"{obs.tailings_type}; class={obs.particle_size_class}; {obs.row_ref}",
                source_id=obs.source_file,
                quote=(
                    f"{obs.factory}: в классе {obs.particle_size_class} зафиксированы потери "
                    f"{ELEMENT_LABELS.get(obs.element, obs.element)} {obs.loss_mass_t:.1f} т"
                ),
                confidence=0.9,
            )
        )
    for index, item in enumerate(data.expert_hypotheses, start=1):
        claims.append(
            EvidenceClaim(
                id=f"EXP-{index:03d}",
                subject=item.factory,
                relation="expert_suggests",
                object=item.text,
                direction="unknown",
                source_id=item.source_file,
                quote=item.text,
                confidence=0.78,
            )
        )
    return claims
