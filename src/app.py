from __future__ import annotations

import json
import sys

import pandas as pd

from hypothesis_factory.config import DEFAULT_CONSTRAINTS, DEFAULT_KPI, DEFAULT_WEIGHTS, PROCESSED_DIR
from hypothesis_factory.export.export_json import hypotheses_to_csv, hypotheses_to_json
from hypothesis_factory.export.report_docx import build_docx_report
from hypothesis_factory.graph.graph_viz import graph_edges_table, graph_to_plotly_figure
from hypothesis_factory.ingestion.parsers import parse_uploaded_file
from hypothesis_factory.pipeline import PipelineResult, run_pipeline
from hypothesis_factory.scoring.scorer import rank_hypotheses


def _load_streamlit():
    try:
        import streamlit as st

        return st
    except Exception:
        print("Streamlit is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        raise


def _as_df(items) -> pd.DataFrame:
    return pd.DataFrame([item.to_dict() if hasattr(item, "to_dict") else item for item in items])


def _valid_default(options, preferred=None):
    options = list(options)
    if preferred is None:
        return options
    preferred = list(preferred)
    option_set = set(options)
    selected = [item for item in preferred if item in option_set]
    return selected or options


def _format_list(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(_display_text(item) for item in value if _display_text(item).strip())
    return _display_text(value)


def _display_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return _format_list(value)
    if isinstance(value, bool):
        return "Да" if value else "Нет"
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    for source, target in DISPLAY_REPLACEMENTS:
        text = text.replace(source, target)
    return _polish_sentence(text)


def _polish_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text
    technical_markers = ("\\", ".json", ".png", ".jpg", ".jpeg", ".xlsx", ".xls", ".csv", ".docx", ".pdf")
    if any(marker in stripped.lower() for marker in technical_markers):
        return text
    if len(stripped) < 45 or len(stripped.split()) < 5:
        return text
    first_alpha_index = next((index for index, char in enumerate(stripped) if char.isalpha()), None)
    if first_alpha_index is None:
        return text
    polished = stripped[:first_alpha_index] + stripped[first_alpha_index].upper() + stripped[first_alpha_index + 1 :]
    if polished[-1] not in ".!?…:":
        polished += "."
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    return f"{leading}{polished}{trailing}"


def _localize_display_df(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    for column in localized.columns:
        if localized[column].dtype == object or localized[column].dtype == bool:
            localized[column] = localized[column].map(_display_text)
    return localized


FEEDBACK_PATH = PROCESSED_DIR / "expert_feedback.json"

DISPLAY_REPLACEMENTS = (
    ("element_28/element_29", "Элементы 28/29"),
    ("element_28/29", "Элементы 28/29"),
    ("element_28", "Элемент 28"),
    ("element_29", "Элемент 29"),
    ("tailings_loss", "Потери в хвостах"),
    ("tailings loss", "Потери в хвостах"),
    ("loss in", "Потери в"),
    ("tailings", "Хвосты"),
    ("process_stage", "Стадия процесса"),
    ("expert_brainstorm", "Экспертные гипотезы"),
    ("knowledge_base", "База знаний"),
    ("tailings_xlsx_summary", "Сводка XLSX по хвостам"),
    ("tailings_xlsx", "Данные XLSX по хвостам"),
    ("water_addition", "Добавка воды"),
    ("reagent_regime", "Реагентный режим"),
    ("regulation", "Регламент"),
    ("scheme", "Схема"),
    ("class=", "Класс "),
    ("Factory", "Фабрика"),
    ("Element", "Элемент"),
    ("Equipment", "Оборудование"),
    ("Process", "Процесс"),
    ("Property", "Показатель"),
    ("Material", "Материал"),
    ("Stage", "Стадия"),
    ("KPI", "Целевой показатель"),
)

WEIGHT_LABELS = {
    "value": "Ценность",
    "novelty": "Новизна",
    "feasibility": "Реализуемость",
    "evidence": "Доказательная база",
    "uncertainty": "Важность неопределенности",
    "expert_alignment": "Согласие эксперта",
    "risk": "Штраф за риск",
    "cost": "Штраф за стоимость",
}

PRIORITY_PROFILES = {
    "Сбалансированный": {
        "description": "Равный фокус на ценности, реализуемости, неопределенности и рисках.",
        "weights": DEFAULT_WEIGHTS,
    },
    "Максимальный эффект на KPI": {
        "description": "Выше поднимает гипотезы с сильной связью с целевым показателем.",
        "weights": {
            "value": 2.0,
            "novelty": 0.8,
            "feasibility": 0.5,
            "evidence": 0.8,
            "uncertainty": 1.3,
            "expert_alignment": 0.6,
            "risk": 0.1,
            "cost": 0.05,
        },
    },
    "Минимальный риск": {
        "description": "Выше поднимает проверенные и технологически осторожные гипотезы.",
        "weights": {
            "value": 0.9,
            "novelty": 0.45,
            "feasibility": 1.35,
            "evidence": 1.25,
            "uncertainty": 0.85,
            "expert_alignment": 0.8,
            "risk": 1.6,
            "cost": 0.75,
        },
    },
    "Быстрая проверка": {
        "description": "Выше поднимает гипотезы, которые проще проверить малым экспериментом.",
        "weights": {
            "value": 0.95,
            "novelty": 0.5,
            "feasibility": 1.65,
            "evidence": 1.0,
            "uncertainty": 0.8,
            "expert_alignment": 0.7,
            "risk": 0.75,
            "cost": 1.15,
        },
    },
    "Минимальная стоимость": {
        "description": "Выше поднимает гипотезы без дорогих изменений оборудования и схемы.",
        "weights": {
            "value": 0.85,
            "novelty": 0.45,
            "feasibility": 1.2,
            "evidence": 0.9,
            "uncertainty": 0.8,
            "expert_alignment": 0.6,
            "risk": 0.85,
            "cost": 1.75,
        },
    },
    "Максимальная новизна": {
        "description": "Выше поднимает более исследовательские и менее очевидные гипотезы.",
        "weights": {
            "value": 0.8,
            "novelty": 2.0,
            "feasibility": 0.2,
            "evidence": 0.65,
            "uncertainty": 1.3,
            "expert_alignment": 0.45,
            "risk": 0.05,
            "cost": 0.05,
        },
    },
}

ZONE_LABELS = {
    "fine_particle_loss": "Потери в тонких классах",
    "coarse_locked_loss": "Потери в грубых классах",
    "missing_process_link": "Нет связи с узлом схемы",
    "expert_unvalidated": "Экспертная идея без проверки",
    "contradiction": "Требуется валидация противоречия",
    "coverage_gap": "Пробел покрытия",
    "indirect_link": "Косвенная связь",
    "mechanism_gap": "Пробел механизма",
    "kpi_gap": "Пробел связи с целевым показателем",
}

STATUS_LABELS = {
    "well_covered": "Хорошо покрыто",
    "weakly_covered": "Слабо покрыто",
    "uncovered": "Не покрыто",
}

SCORE_LABELS = {
    "value": "Ценность",
    "novelty": "Новизна",
    "feasibility": "Реализуемость",
    "evidence": "Доказательная база",
    "uncertainty_importance": "Важность неопределенности",
    "expert_alignment": "Согласие эксперта",
    "risk": "Риск",
    "cost": "Стоимость",
    "final_score": "Итоговый балл",
}

DECISION_TO_INTERNAL = {"На рассмотрении": "pending", "Принять": "accept", "Отклонить": "reject"}
DECISION_FROM_INTERNAL = {value: key for key, value in DECISION_TO_INTERNAL.items()}

SOURCE_TYPE_LABELS = {
    "knowledge_base": "База знаний",
    "tailings_xlsx": "Данные XLSX по хвостам",
    "expert_brainstorm": "Экспертные гипотезы",
    "tailings_xlsx_summary": "Сводка XLSX по хвостам",
    "demo": "Демо",
}

RELATION_LABELS = {
    "has_tailings_loss": "Есть потери в хвостах",
    "expert_suggests": "Эксперт предлагает",
    "requires_validation": "Требует проверки",
    "concentrated_in": "Сконцентрировано в",
    "points_to": "Указывает на",
}

DIRECTION_LABELS = {
    "increases": "Увеличивает",
    "decreases": "Снижает",
    "affects": "Влияет",
    "no_effect": "Нет эффекта",
    "unknown": "Неизвестно",
}


def _load_feedback() -> dict:
    if not FEEDBACK_PATH.exists():
        return {}
    try:
        return json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_feedback(feedback: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")


def _feedback_adjustment(decision: str) -> float:
    if decision == "accept":
        return 1.0
    if decision == "reject":
        return -0.8
    return 0.0


def _ru_zone(value: str) -> str:
    return ZONE_LABELS.get(str(value), str(value))


def _ru_status(value: str) -> str:
    return STATUS_LABELS.get(str(value), str(value))


def _score_df(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{SCORE_LABELS.get(key, key): value for key, value in scores.items()}])


def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={key: value for key, value in mapping.items() if key in df.columns})


def _localize_source_type(value: str) -> str:
    return SOURCE_TYPE_LABELS.get(str(value), str(value))


def _apply_feedback_to_hypotheses(hypotheses, feedback: dict):
    for item in hypotheses:
        item_feedback = feedback.get(item.id, {})
        item.expert_feedback = item_feedback or None
        item.scores["expert_alignment"] = _feedback_adjustment(item_feedback.get("decision", "pending"))
    return hypotheses


def _profile_weights(profile: str) -> dict[str, float]:
    return {**DEFAULT_WEIGHTS, **PRIORITY_PROFILES[profile]["weights"]}


def _weights_sidebar(st) -> tuple[str, dict[str, float]]:
    st.sidebar.header("Приоритизация гипотез")
    st.sidebar.caption("Выберите, что сейчас важнее для исследования.")
    mode = st.sidebar.radio(
        "Режим приоритизации",
        ["Готовый профиль", "Ручная настройка"],
        horizontal=False,
    )
    if mode == "Готовый профиль":
        profile = st.sidebar.selectbox("Профиль приоритизации", list(PRIORITY_PROFILES.keys()))
        st.sidebar.caption(PRIORITY_PROFILES[profile]["description"])
        weights = _profile_weights(profile)
        return profile, weights

    profile = "Ручная настройка"
    st.sidebar.caption("Настройте веса самостоятельно. Эти значения применятся после запуска обработки.")
    weights = {}
    for key, default in DEFAULT_WEIGHTS.items():
        weights[key] = st.sidebar.slider(
            WEIGHT_LABELS.get(key, key),
            0.0,
            2.0,
            float(default),
            0.05,
            key=f"manual_weight_{key}",
        )
    return profile, weights


def _parse_uploads(st, enabled: bool) -> list:
    if not enabled:
        return []
    uploaded = st.file_uploader(
        "Дополнительные файлы базы знаний",
        type=["txt", "pdf", "docx", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    docs = []
    errors = []
    for file in uploaded or []:
        try:
            docs.append(parse_uploaded_file(file, file.name))
        except ValueError as exc:
            errors.append(str(exc))
    for error in errors:
        st.warning(error)
    return docs


def _run(
    st,
    kpi: str,
    constraints: str,
    uploaded_docs: list,
    priority_profile: str,
    weights: dict[str, float],
) -> PipelineResult:
    st.session_state.result = run_pipeline(kpi, constraints, uploaded_docs or None)
    st.session_state.active_kpi = kpi
    st.session_state.active_constraints = constraints
    st.session_state.active_uploaded_titles = [doc.title for doc in uploaded_docs]
    st.session_state.active_priority_profile = priority_profile
    st.session_state.active_weights = dict(weights)
    return st.session_state.result


def _draft_changed(
    st,
    kpi: str,
    constraints: str,
    uploaded_docs: list,
    priority_profile: str,
    weights: dict[str, float],
) -> bool:
    if "result" not in st.session_state:
        return False
    return (
        st.session_state.get("active_kpi") != kpi
        or st.session_state.get("active_constraints") != constraints
        or st.session_state.get("active_uploaded_titles", []) != [doc.title for doc in uploaded_docs]
        or st.session_state.get("active_priority_profile") != priority_profile
        or st.session_state.get("active_weights", {}) != weights
    )


def task_setup_tab(
    st,
    result: PipelineResult,
    weights: dict[str, float],
    kpi: str,
    constraints: str,
    priority_profile: str,
) -> None:
    st.subheader("Постановка задачи")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Источники", len(result.documents))
    c2.metric("Факты / утверждения", len(result.claims))
    c3.metric("Зоны неопределенности", len(result.zones))
    c4.metric("Гипотезы", len(result.hypotheses))
    if result.tailings_observations:
        c5, c6, c7 = st.columns(3)
        c5.metric("Наблюдения по хвостам", len(result.tailings_observations))
        c6.metric("Экспертные гипотезы", len(result.expert_hypotheses or []))
        c7.metric("PNG-схемы и регламенты", len(result.images or []))
    st.caption("Показаны параметры последнего запуска пайплайна.")
    st.text_area("Текущий целевой показатель", _polish_sentence(kpi), disabled=True)
    st.text_area("Ограничения", _polish_sentence(constraints), disabled=True)
    st.text_input("Профиль приоритизации", priority_profile, disabled=True)
    st.dataframe(pd.DataFrame([{WEIGHT_LABELS.get(key, key): value for key, value in weights.items()}]), use_container_width=True, hide_index=True)


def knowledge_base_tab(st, result: PipelineResult) -> None:
    st.subheader("База знаний")
    docs = _rename_columns(
        _as_df(result.documents)[["id", "title", "source_type", "language"]],
        {"id": "ID", "title": "Название", "source_type": "Тип источника", "language": "Язык"},
    )
    docs["Тип источника"] = docs["Тип источника"].map(_localize_source_type)
    st.dataframe(docs, use_container_width=True, hide_index=True)
    st.caption("Фрагменты сохраняются в локальный SQLite, чтобы цепочка обработки оставалась прослеживаемой.")
    chunks = _rename_columns(
        _as_df(result.chunks),
        {"id": "ID", "source_id": "Источник", "text": "Текст", "position": "Позиция"},
    )
    st.dataframe(_localize_display_df(chunks), use_container_width=True, hide_index=True)


def tailings_data_tab(st, result: PipelineResult) -> None:
    st.subheader("Реальные данные по хвостам")
    if not result.tailings_observations:
        st.info("Реальный слой хвостов не активен: используется загруженная вручную база знаний.")
        return
    df = pd.DataFrame([item.to_dict() for item in result.tailings_observations])
    factories = sorted(df["factory"].dropna().unique())
    elements = sorted(df["element"].dropna().unique())
    selected_factories = st.multiselect("Фабрика", factories, default=_valid_default(factories))
    selected_elements = st.multiselect("Элемент", elements, default=_valid_default(elements), format_func=_display_text)
    view = df[df["factory"].isin(selected_factories) & df["element"].isin(selected_elements)]
    dist = view[view["extractable"].isna() & view["loss_mass_t"].notna()]
    display_view = _rename_columns(
        view,
        {
            "source_file": "Файл-источник",
            "factory": "Фабрика",
            "tailings_type": "Тип хвостов",
            "element": "Элемент",
            "particle_size_class": "Класс крупности",
            "mineral_form": "Минеральная форма",
            "loss_share_pct": "Доля потерь, %",
            "loss_mass_t": "Масса потерь, т",
            "class_share_pct": "Доля класса, %",
            "extractable": "Извлекаемый",
            "row_ref": "Строка",
        },
    )
    st.dataframe(_localize_display_df(display_view), use_container_width=True, hide_index=True)
    if not dist.empty:
        chart = dist.groupby(["factory", "particle_size_class"], as_index=False)["loss_mass_t"].sum()
        chart = chart.rename(columns={"factory": "Фабрика", "particle_size_class": "Класс крупности", "loss_mass_t": "Масса потерь, т"})
        st.bar_chart(chart, x="Класс крупности", y="Масса потерь, т", color="Фабрика")


def schemes_tab(st, result: PipelineResult) -> None:
    st.subheader("Схемы флотации и регламенты")
    images = result.images or []
    if not images:
        st.info("PNG-схемы не найдены или реальный слой не активен.")
        return
    image_df = _rename_columns(
        pd.DataFrame([item.to_dict() for item in images]),
        {
            "file_path": "Путь к файлу",
            "file_name": "Имя файла",
            "folder": "Папка",
            "inferred_type": "Тип",
            "optional_factory": "Фабрика",
            "optional_stage": "Стадия",
            "tags": "Теги",
            "ocr_text": "Текст OCR",
        },
    )
    st.dataframe(_localize_display_df(image_df), use_container_width=True, hide_index=True)
    image_types = sorted({item.inferred_type for item in images})
    selected_type = st.multiselect("Тип изображения", image_types, default=_valid_default(image_types), format_func=_display_text)
    filtered = [item for item in images if item.inferred_type in selected_type]
    for row_start in range(0, len(filtered), 3):
        cols = st.columns(3)
        for col, item in zip(cols, filtered[row_start : row_start + 3]):
            with col:
                st.image(item.file_path, caption=f"{item.file_name} / {_display_text(item.inferred_type)}", use_container_width=True)


def entities_claims_tab(st, result: PipelineResult) -> None:
    st.subheader("Сущности")
    entities = _rename_columns(
        _as_df(result.entities),
        {"id": "ID", "name": "Название", "type": "Тип", "source_id": "Источник", "confidence": "Уверенность"},
    )
    st.dataframe(_localize_display_df(entities), use_container_width=True, hide_index=True)
    st.subheader("Факты и утверждения")
    claims = _rename_columns(
        _as_df(result.claims),
        {
            "id": "ID",
            "subject": "Субъект",
            "relation": "Связь",
            "object": "Объект",
            "direction": "Направление",
            "magnitude": "Величина",
            "condition": "Условие",
            "source_id": "Источник",
            "quote": "Цитата / факт",
            "confidence": "Уверенность",
        },
    )
    if "Связь" in claims.columns:
        claims["Связь"] = claims["Связь"].map(lambda value: RELATION_LABELS.get(str(value), str(value)))
    if "Направление" in claims.columns:
        claims["Направление"] = claims["Направление"].map(lambda value: DIRECTION_LABELS.get(str(value), str(value)))
    st.dataframe(_localize_display_df(claims), use_container_width=True, hide_index=True)


def coverage_tab(st, result: PipelineResult) -> None:
    st.subheader("Карта изученности")
    if result.coverage_matrix.empty:
        st.info("Карта изученности пока пуста.")
        return
    status = result.coverage_matrix["status"].map(_ru_status).value_counts().reset_index()
    status.columns = ["Статус", "Количество"]
    st.bar_chart(status.set_index("Статус"))
    status_options = sorted(result.coverage_matrix["status"].dropna().unique())
    status_display = [_ru_status(item) for item in status_options]
    selected_status_display = st.multiselect(
        "Статус покрытия",
        status_display,
        default=_valid_default(status_display, [_ru_status("uncovered"), _ru_status("weakly_covered")]),
    )
    selected_status = [raw for raw in status_options if _ru_status(raw) in selected_status_display]
    view = result.coverage_matrix[result.coverage_matrix["status"].isin(selected_status)].copy()
    view["status"] = view["status"].map(_ru_status)
    view = _rename_columns(
        view,
        {
            "factory": "Фабрика",
            "tailings_type": "Тип хвостов",
            "element": "Элемент",
            "particle_size_class": "Класс крупности",
            "loss_mass_t": "Масса потерь, т",
            "loss_share_pct": "Доля потерь, %",
            "has_extractability": "Есть данные об извлекаемости",
            "status": "Статус",
        },
    )
    st.dataframe(_localize_display_df(view.head(300)), use_container_width=True, hide_index=True)


def zones_tab(st, result: PipelineResult) -> None:
    st.subheader("Пробелы, противоречия и зоны неопределенности")
    zones_df = _as_df(result.zones)
    if zones_df.empty:
        st.info("Зоны неопределенности пока не найдены.")
        return
    view = pd.DataFrame(
        {
            "ID": zones_df["id"],
            "Тип зоны": zones_df["type"].map(_ru_zone),
            "Наблюдаемый паттерн": zones_df["description"],
            "Связанные сущности": zones_df["linked_entities"].apply(_format_list),
            "Источники": zones_df["source_links"].apply(_format_list),
            "Почему важно": zones_df["why_it_matters"],
            "Минимальная проверка": zones_df["suggested_check"],
            "Приоритет": zones_df["priority"],
        }
    )
    st.dataframe(_localize_display_df(view), use_container_width=True, hide_index=True)
    st.caption("Гипотезы во вкладке «Гипотезы» рождаются из этих зон неопределенности, а не из свободного текста запроса.")


def hypotheses_tab(st, result: PipelineResult, weights: dict[str, float]) -> None:
    st.subheader("Ранжированные гипотезы")
    feedback = st.session_state.setdefault("expert_feedback", _load_feedback())
    ranked = rank_hypotheses(_apply_feedback_to_hypotheses(result.hypotheses, feedback), weights)
    zone_types = sorted({item.origin_uncertainty_zone.get("type") for item in ranked})
    zone_display = [_ru_zone(item) for item in zone_types]
    selected_display = st.multiselect("Тип зоны неопределенности", zone_display, default=_valid_default(zone_display))
    selected = [raw for raw in zone_types if _ru_zone(raw) in selected_display]
    for item in [hyp for hyp in ranked if hyp.origin_uncertainty_zone.get("type") in selected]:
        with st.container(border=True):
            cols = st.columns([4, 1])
            cols[0].markdown(f"### {item.id}. {_display_text(item.title)}")
            cols[1].metric("Итоговый балл", item.scores["final_score"])
            st.write(_display_text(item.hypothesis))
            st.markdown(
                f"**Источник неопределенности:** `{item.origin_uncertainty_zone.get('id')}` / {_ru_zone(item.origin_uncertainty_zone.get('type'))}"
            )
            st.write(_display_text(item.origin_uncertainty_zone.get("description")))
            left, right = st.columns(2)
            with left:
                st.markdown("**Доказательная база**")
                for evidence in item.evidence:
                    st.write(f"- `{_display_text(evidence.get('source_id'))}`: {_display_text(evidence.get('quote'))}")
            with right:
                st.markdown("**Минимальный эксперимент**")
                for step in item.minimal_experiment:
                    st.write(f"- {_display_text(step)}")
            st.dataframe(_score_df(item.scores), use_container_width=True, hide_index=True)


def _display_node_label(kind: str, label: str) -> str:
    zone_labels = {
        "fine_particle_loss": "Тонкие потери",
        "coarse_locked_loss": "Грубые потери",
        "missing_process_link": "Нет связи с узлом",
        "expert_unvalidated": "Эксперт без проверки",
        "contradiction": "Нужна валидация",
    }
    element_labels = {"element_28": "Элемент 28", "element_29": "Элемент 29", "element_28/29": "Элементы 28/29"}
    if kind == "element":
        return element_labels.get(label, label)
    if kind == "zone":
        return zone_labels.get(label, label)
    return label


def _zone_action(zone_type: str) -> str:
    if zone_type == "fine_particle_loss":
        return "Тонкая флотация"
    if zone_type == "coarse_locked_loss":
        return "Раскрытие/классификация"
    if zone_type == "missing_process_link":
        return "Привязка к узлу"
    if zone_type == "expert_unvalidated":
        return "Проверка экспертной идеи"
    if zone_type == "contradiction":
        return "Валидация конфликта"
    return "Проверка"


def _real_case_graph_rows(result: PipelineResult) -> pd.DataFrame:
    rows = []
    for zone in result.zones:
        entities = zone.linked_entities
        factory = entities[0] if entities else ""
        element = next((str(item) for item in entities if str(item).startswith("element_")), "")
        size_class = next((str(item) for item in entities if str(item).startswith(("+", "-"))), "")
        rows.append(
            {
                "Фабрика": factory,
                "Элемент": _display_node_label("element", element or "element_28/29"),
                "Класс": size_class or "Не указан",
                "Зона неопределенности": _display_node_label("zone", zone.type),
                "Действие проверки": _zone_action(zone.type),
                "Источники": _format_list(zone.source_links),
            }
        )
    return pd.DataFrame(rows)


def _build_graphviz_dot(rows: pd.DataFrame) -> str:
    safe_rows = rows.drop_duplicates(
        subset=["Фабрика", "Элемент", "Класс", "Зона неопределенности", "Действие проверки"]
    )
    columns = [
        ("Фабрика", "Фабрика", "#2563eb"),
        ("Элемент", "Элемент", "#16a34a"),
        ("Класс", "Класс", "#f59e0b"),
        ("Зона неопределенности", "Зона", "#dc2626"),
        ("Действие проверки", "Проверка", "#7c3aed"),
    ]
    node_ids: dict[tuple[str, str], str] = {}
    lines = [
        "digraph G {",
        "  graph [rankdir=LR, bgcolor=\"transparent\", pad=\"0.25\", nodesep=\"0.45\", ranksep=\"0.65\"];",
        "  node [shape=box, style=\"rounded,filled\", color=\"#e5e7eb\", fontname=\"Arial\", fontsize=12, fontcolor=\"#ffffff\", margin=\"0.12,0.08\"];",
        "  edge [color=\"#94a3b8\", arrowsize=0.7, penwidth=1.4];",
    ]

    def quote(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace("\"", "\\\"")

    def node_id(column: str, value: str, color: str) -> str:
        key = (column, value)
        if key not in node_ids:
            node_name = f"n{len(node_ids)}"
            node_ids[key] = node_name
            lines.append(f"  {node_name} [label=\"{quote(value)}\", fillcolor=\"{color}\"];")
        return node_ids[key]

    edges: set[tuple[str, str]] = set()
    for _, row in safe_rows.iterrows():
        chain = []
        for column, _, color in columns:
            value = str(row[column])
            chain.append(node_id(column, value, color))
        for source, target in zip(chain, chain[1:]):
            edges.add((source, target))
    for source, target in sorted(edges):
        lines.append(f"  {source} -> {target};")

    for column, _, color in columns:
        rank_nodes = [node_ids[(column, str(value))] for value in dict.fromkeys(safe_rows[column].astype(str).tolist())]
        if rank_nodes:
            lines.append("  { rank=same; " + "; ".join(rank_nodes) + "; }")
    lines.append("}")
    return "\n".join(lines)


def graph_tab(st, result: PipelineResult) -> None:
    st.subheader("Карта связей базы знаний")
    if result.tailings_observations:
        rows = _real_case_graph_rows(result)
        c1, c2, c3 = st.columns(3)
        c1.metric("Связей", len(rows))
        c2.metric("Фабрик", rows["Фабрика"].nunique())
        c3.metric("Типов зон", rows["Зона неопределенности"].nunique())
        st.markdown("**Фабрика -> Элемент -> Класс крупности -> Зона неопределенности -> Действие проверки**")
        st.graphviz_chart(_build_graphviz_dot(rows), use_container_width=True)
        st.dataframe(_localize_display_df(rows), use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Где сконцентрированы зоны**")
            zone_counts = rows.groupby(["Фабрика", "Элемент"], as_index=False).size().rename(columns={"size": "Количество"})
            st.bar_chart(zone_counts, x="Фабрика", y="Количество", color="Элемент")
        with right:
            st.markdown("**Какие проверки нужны чаще всего**")
            action_counts = rows["Действие проверки"].value_counts().reset_index()
            action_counts.columns = ["Действие проверки", "Количество"]
            st.bar_chart(action_counts, x="Действие проверки", y="Количество")
        st.caption("Граф построен с фиксированными уровнями, поэтому подписи не накладываются друг на друга.")
    else:
        fig = graph_to_plotly_figure(result.graph)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Визуализация графа недоступна в этой среде; показываю таблицу связей.")
        edge_table = _rename_columns(pd.DataFrame(graph_edges_table(result.graph)), {"source": "Источник", "target": "Цель"})
        st.dataframe(_localize_display_df(edge_table), use_container_width=True, hide_index=True)
    indirect = [zone for zone in result.zones if zone.type == "indirect_link"]
    if indirect:
        st.subheader("Косвенные пути")
        indirect_df = _rename_columns(
            _as_df(indirect)[["id", "indirect_path", "description", "priority"]],
            {"indirect_path": "Косвенный путь", "description": "Описание", "priority": "Приоритет"},
        )
        st.dataframe(_localize_display_df(indirect_df), use_container_width=True, hide_index=True)


def tz_check_tab(st, result: PipelineResult) -> None:
    st.subheader("Чеклист соответствия ТЗ")
    checks = [
        {
            "Требование": "Прием базы знаний и разнородных данных",
            "Статус": "Закрыто",
            "Как реализовано": "Загрузка файлов популярных форматов плюс сводка реального кейса, экспертные гипотезы и реестр изображений.",
        },
        {
            "Требование": "Извлечение сущностей и связей",
            "Статус": "Закрыто",
            "Как реализовано": f"{len(result.entities)} сущностей, {len(result.claims)} фактов/утверждений, граф связей.",
        },
        {
            "Требование": "Выявление паттернов и пробелов",
            "Статус": "Закрыто",
            "Как реализовано": f"{len(result.zones)} зон неопределенности: пики потерь, отсутствие связи с узлом схемы, противоречия, непроверенные экспертные идеи.",
        },
        {
            "Требование": "Проверяемые гипотезы с обоснованием",
            "Статус": "Закрыто",
            "Как реализовано": f"{len(result.hypotheses)} гипотез с механизмом, доказательной базой, рисками и критериями успеха/провала.",
        },
        {
            "Требование": "Ранжирование по новизне, реализуемости, эффекту, рискам",
            "Статус": "Закрыто",
            "Как реализовано": "Интерпретируемый скоринг с весами в боковой панели и пересчетом рейтинга.",
        },
        {
            "Требование": "Визуализация связей",
            "Статус": "Закрыто",
            "Как реализовано": "Читаемая карта связей, группировки по фабрикам/элементам и действиям проверки.",
        },
        {
            "Требование": "Экспорт и экспертная обратная связь",
            "Статус": "Закрыто",
            "Как реализовано": "Эксперт может принять, отклонить, отредактировать и прокомментировать гипотезу; результат выгружается в машинный, табличный и документный форматы.",
        },
        {
            "Требование": "Экспертная настройка и обучение на обратной связи",
            "Статус": "Закрыто",
            "Как реализовано": "Экспертная обратная связь сохраняется локально и влияет на фактор согласия эксперта в последующем ранжировании.",
        },
    ]
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)


def review_export_tab(st, result: PipelineResult, weights: dict[str, float]) -> None:
    st.subheader("Экспертная проверка")
    feedback = st.session_state.setdefault("expert_feedback", _load_feedback())
    ranked = rank_hypotheses(_apply_feedback_to_hypotheses(result.hypotheses, feedback), weights)
    accepted = sum(1 for item in feedback.values() if item.get("decision") == "accept")
    rejected = sum(1 for item in feedback.values() if item.get("decision") == "reject")
    c1, c2, c3 = st.columns(3)
    c1.metric("Принято", accepted)
    c2.metric("Отклонено", rejected)
    c3.metric("Файл обратной связи", "Локальный JSON")
    for item in ranked:
        with st.expander(f"{item.id}: {_display_text(item.title)}"):
            status = st.radio(
                "Решение",
                ["На рассмотрении", "Принять", "Отклонить"],
                horizontal=True,
                key=f"decision-{item.id}",
                index=["На рассмотрении", "Принять", "Отклонить"].index(
                    DECISION_FROM_INTERNAL.get(feedback.get(item.id, {}).get("decision", "pending"), "На рассмотрении")
                ),
            )
            comment = st.text_area("Комментарий", key=f"comment-{item.id}", value=feedback.get(item.id, {}).get("comment", ""))
            edited = st.text_area("Экспертная редактура", key=f"edit-{item.id}", value=_display_text(feedback.get(item.id, {}).get("edit", item.hypothesis)))
            feedback[item.id] = {"decision": DECISION_TO_INTERNAL[status], "comment": comment, "edit": edited}
            item.expert_feedback = feedback[item.id]

    st.session_state.expert_feedback = feedback
    if st.button("Сохранить обратную связь и пересчитать рейтинг", type="primary"):
        _save_feedback(feedback)
        st.success("Обратная связь сохранена.")
        st.rerun()

    st.subheader("Экспорт")
    reranked = rank_hypotheses(_apply_feedback_to_hypotheses(result.hypotheses, feedback), weights)
    json_data = hypotheses_to_json(reranked)
    csv_data = hypotheses_to_csv(reranked)
    st.download_button("Скачать JSON", json_data, "hypotheses.json", "application/json")
    st.download_button("Скачать CSV", csv_data, "hypotheses.csv", "text/csv")
    try:
        docx = build_docx_report(reranked)
        st.download_button(
            "Скачать DOCX",
            docx,
            "hypothesis_factory_report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except RuntimeError as exc:
        st.warning(str(exc))


def main() -> None:
    st = _load_streamlit()
    st.set_page_config(page_title="Фабрика гипотез", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
            display: none !important;
        }
        #MainMenu, footer {
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Фабрика гипотез для снижения потерь в хвостах")
    st.caption("Генерируем гипотезы из пробелов, противоречий и пиков потерь в реальной базе знаний по флотационному обогащению.")

    with st.sidebar:
        st.header("Постановка")
        kpi = _polish_sentence(st.text_area("Целевой показатель", _polish_sentence(DEFAULT_KPI), height=90, key="kpi_input_polished_v2"))
        constraints = _polish_sentence(st.text_area("Ограничения", _polish_sentence(DEFAULT_CONSTRAINTS), height=120, key="constraints_input_polished_v2"))
        enable_uploads = st.checkbox("Добавить свои файлы", value=False)
        uploaded_docs = _parse_uploads(st, enable_uploads)
        priority_profile, weights = _weights_sidebar(st)
        run_button = st.button("Запустить обработку", type="primary", use_container_width=True)
        st.caption("Изменения KPI, ограничений, файлов и профиля применятся после запуска обработки.")

    if run_button or "result" not in st.session_state:
        result = _run(st, kpi, constraints, uploaded_docs, priority_profile, weights)
    else:
        result = st.session_state.result

    active_kpi = st.session_state.get("active_kpi", kpi)
    active_constraints = st.session_state.get("active_constraints", constraints)
    active_priority_profile = st.session_state.get("active_priority_profile", priority_profile)
    active_weights = st.session_state.get("active_weights", weights)
    if _draft_changed(st, kpi, constraints, uploaded_docs, priority_profile, weights):
        st.info("Параметры в боковой панели изменены, но результат еще не пересчитан. Нажмите «Запустить обработку», чтобы применить изменения.")

    tabs = st.tabs(
        [
            "Постановка",
            "Чеклист ТЗ",
            "База знаний",
            "Хвосты",
            "Схемы",
            "Сущности и факты",
            "Карта изученности",
            "Зоны неопределенности",
            "Гипотезы",
            "Граф знаний",
            "Эксперт и экспорт",
        ]
    )
    with tabs[0]:
        task_setup_tab(st, result, active_weights, active_kpi, active_constraints, active_priority_profile)
    with tabs[1]:
        tz_check_tab(st, result)
    with tabs[2]:
        knowledge_base_tab(st, result)
    with tabs[3]:
        tailings_data_tab(st, result)
    with tabs[4]:
        schemes_tab(st, result)
    with tabs[5]:
        entities_claims_tab(st, result)
    with tabs[6]:
        coverage_tab(st, result)
    with tabs[7]:
        zones_tab(st, result)
    with tabs[8]:
        hypotheses_tab(st, result, active_weights)
    with tabs[9]:
        graph_tab(st, result)
    with tabs[10]:
        review_export_tab(st, result, active_weights)


if __name__ == "__main__":
    main()
