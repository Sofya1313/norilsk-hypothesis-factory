from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "01_Data"
DEMO_DIR = DATA_DIR / "demo"
PROCESSED_DIR = DATA_DIR / "processed_runtime"
EXPORT_DIR = PROJECT_ROOT / "exports"
DB_PATH = DATA_DIR / "hypothesis_factory.sqlite"
REAL_CASE_ROOT = PROJECT_ROOT / "01_Data" / "raw" / "Задача 1. Фабрика гипотез" / "Задача 1"
REAL_CASE_SUMMARY_PATH = PROJECT_ROOT / "01_Data" / "processed" / "nornickel_real_case_data_summary.json"
REAL_CASE_KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "01_Data" / "processed" / "nornickel_real_case_knowledge_base.md"

DEFAULT_KPI = "Снизить потери элемента 28 / элемента 29 в хвостах обогатительных фабрик."
DEFAULT_CONSTRAINTS = (
    "Использовать текущие схемы флотации, классификации и измельчения; "
    "не предлагать гипотезы без минимального проверочного эксперимента; "
    "учитывать классы крупности, минеральные формы и доступное оборудование."
)

DEFAULT_WEIGHTS = {
    "value": 1.0,
    "novelty": 0.8,
    "feasibility": 0.9,
    "evidence": 0.7,
    "uncertainty": 1.0,
    "expert_alignment": 0.6,
    "risk": 0.6,
    "cost": 0.5,
}
