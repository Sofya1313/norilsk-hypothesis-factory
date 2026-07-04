from __future__ import annotations

from copy import deepcopy


WEIGHT_KEYS = (
    "value",
    "novelty",
    "feasibility",
    "evidence",
    "uncertainty",
    "expert_alignment",
    "risk",
    "cost",
)

SCORING_PROFILES = {
    "balanced": {
        "label": "Сбалансированный",
        "description": "Сбалансированный: учитывает эффект, риски, стоимость и качество доказательств без сильного перекоса в один критерий.",
        "weights": {
            "value": 0.8,
            "novelty": 0.6,
            "feasibility": 0.7,
            "evidence": 0.7,
            "uncertainty": 0.7,
            "expert_alignment": 0.5,
            "risk": 0.5,
            "cost": 0.5,
        },
    },
    "max_kpi_effect": {
        "label": "Максимальный эффект на KPI",
        "description": "Максимальный эффект на KPI: сильнее поднимает гипотезы с ожидаемым влиянием на целевой показатель и важные зоны неопределенности.",
        "weights": {
            "value": 1.0,
            "novelty": 0.5,
            "feasibility": 0.6,
            "evidence": 0.7,
            "uncertainty": 0.8,
            "expert_alignment": 0.5,
            "risk": 0.4,
            "cost": 0.3,
        },
    },
    "low_risk": {
        "label": "Минимальный риск",
        "description": "Минимальный риск: отдает приоритет технически понятным гипотезам с сильной доказательной базой и экспертным доверием.",
        "weights": {
            "value": 0.7,
            "novelty": 0.4,
            "feasibility": 0.9,
            "evidence": 0.9,
            "uncertainty": 0.5,
            "expert_alignment": 0.7,
            "risk": 1.0,
            "cost": 0.5,
        },
    },
    "fast_validation": {
        "label": "Быстрая проверка",
        "description": "Быстрая проверка: выше ранжирует гипотезы, которые проще быстро проверить без тяжелого эксперимента.",
        "weights": {
            "value": 0.7,
            "novelty": 0.5,
            "feasibility": 1.0,
            "evidence": 0.6,
            "uncertainty": 0.7,
            "expert_alignment": 0.5,
            "risk": 0.6,
            "cost": 0.8,
        },
    },
    "low_cost": {
        "label": "Минимальная стоимость",
        "description": "Минимальная стоимость: предпочитает гипотезы с дешевой проверкой и высокой практической реализуемостью.",
        "weights": {
            "value": 0.6,
            "novelty": 0.4,
            "feasibility": 0.9,
            "evidence": 0.6,
            "uncertainty": 0.5,
            "expert_alignment": 0.5,
            "risk": 0.6,
            "cost": 1.0,
        },
    },
    "high_novelty": {
        "label": "Максимальная новизна",
        "description": "Максимальная новизна: сильнее выделяет исследовательски перспективные и нестандартные гипотезы.",
        "weights": {
            "value": 0.8,
            "novelty": 1.0,
            "feasibility": 0.5,
            "evidence": 0.5,
            "uncertainty": 0.9,
            "expert_alignment": 0.4,
            "risk": 0.4,
            "cost": 0.3,
        },
    },
}


def profile_options() -> list[str]:
    return list(SCORING_PROFILES)


def profile_label(profile_key: str) -> str:
    return SCORING_PROFILES[profile_key]["label"]


def profile_description(profile_key: str) -> str:
    return SCORING_PROFILES[profile_key]["description"]


def scoring_profile_weights(profile_key: str) -> dict[str, float]:
    return deepcopy(SCORING_PROFILES[profile_key]["weights"])


def _checked_weight(value: float) -> float:
    weight = float(value)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Scoring weights must be between 0.0 and 1.0")
    return weight


def scoring_weights(profile_key: str, expert_weights: dict[str, float] | None = None) -> dict[str, float]:
    weights = scoring_profile_weights(profile_key)
    for key, value in (expert_weights or {}).items():
        if key in WEIGHT_KEYS:
            weights[key] = _checked_weight(value)
    return weights
