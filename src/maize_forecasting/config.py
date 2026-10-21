"""Rutas y archivos de entrada del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIZE_DATA_DIR = PROJECT_ROOT / "AR" / "maize"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "standardized"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "star_schema"
QUALITY_REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
SOURCE_METADATA_PATH = PROJECT_ROOT / "metadata" / "data_dictionary.json"
FEATURE_REGISTRY_PATH = PROJECT_ROOT / "metadata" / "feature_registry.json"
FEATURE_REPORT_DIR = PROJECT_ROOT / "reports" / "feature_engineering"
MODELS_DIR = PROJECT_ROOT / "models"
MODELING_REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"
FEATURE_CUTOFFS = (0.25, 0.5, 0.75)
STAR_SCHEMA_DB = PROCESSED_DIR / "maize_star_schema.duckdb"

INPUT_FILES = {
    "yield": "yield_maize_AR.csv",
    "meteorology": "meteo_maize_AR.csv",
    "soil_moisture": "soil_moisture_maize_AR.csv",
    "ndvi": "ndvi_maize_AR.csv",
    "fpar": "fpar_maize_AR.csv",
    "soil": "soil_maize_AR.csv",
    "crop_calendar": "crop_calendar_maize_AR.csv",
    "location": "location_maize_AR.csv",
    "crop_mask": "crop_mask_maize_AR.csv",
}
