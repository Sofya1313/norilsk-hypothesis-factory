from __future__ import annotations

from hypothesis_factory.config import DEFAULT_WEIGHTS
from hypothesis_factory.models import Hypothesis, UncertaintyZone


def calculate_uncertainty_importance(zone: UncertaintyZone) -> float:
    return round(
        0.35 * zone.kpi_relevance
        + 0.25 * zone.gap_severity
        + 0.20 * zone.contradiction_strength
        + 0.20 * zone.indirect_mechanism_strength,
        3,
    )


def final_score(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    value = (
        w["value"] * scores.get("value", 0)
        + w["novelty"] * scores.get("novelty", 0)
        + w["feasibility"] * scores.get("feasibility", 0)
        + w["evidence"] * scores.get("evidence", 0)
        + w["uncertainty"] * scores.get("uncertainty_importance", 0)
        + w.get("expert_alignment", 0) * scores.get("expert_alignment", 0)
        - w["risk"] * scores.get("risk", 0)
        - w["cost"] * scores.get("cost", 0)
    )
    return round(value, 3)


def rescore_hypothesis(hypothesis: Hypothesis, weights: dict[str, float] | None = None) -> Hypothesis:
    hypothesis.scores["final_score"] = final_score(hypothesis.scores, weights)
    return hypothesis


def rank_hypotheses(hypotheses: list[Hypothesis], weights: dict[str, float] | None = None) -> list[Hypothesis]:
    return sorted((rescore_hypothesis(item, weights) for item in hypotheses), key=lambda item: item.scores["final_score"], reverse=True)
