import json
from datetime import date

import pandas as pd
import pytest

from maize_forecasting import (
    data_preparation,
    data_quality,
    feature_engineering,
    star_schema,
)
from maize_forecasting.config import SOURCE_METADATA_PATH


def _source_frames():
    return {
        "yield": pd.DataFrame(
            {
                "crop_name": [" MAIZE ", " MAIZE "],
                "country_code": ["ar", "ar"],
                "adm_id": [" ar001 ", " ar002 "],
                "planting_year": ["2020/21", "2020/21"],
                "planting_date": ["2020/21-05", "2020/21-05"],
                "harvest_year": [2021, 2021],
                "harvest_date": ["2021-01", "2021-01"],
                "yield": [5.2, 4.8],
                "production": [520.0, 480.0],
                "harvest_area": [100.0, 100.0],
                "planted_area": [120.0, 110.0],
            }
        ),
        "meteorology": pd.DataFrame(
            {
                "crop_name": ["MAIZE", "MAIZE"],
                "adm_id": ["AR001", "AR002"],
                "date": [20210105, 20210106],
                "tmin": [10.0, 4.0],
                "tmax": [5.0, 8.0],
                "prec": [0.0, 2.0],
                "rad": [100.0, 110.0],
                "tavg": [7.0, 6.0],
                "et0": [1.0, 1.1],
                "vpd": [2.0, 2.1],
                "cwb": [-1.0, 0.9],
            }
        ),
        "soil_moisture": pd.DataFrame(
            {"crop_name": ["maize"], "adm_id": ["AR001"], "date": [20210105], "ssm": [2.0], "rsm": [100.0]}
        ),
        "ndvi": pd.DataFrame(
            {"crop_name": ["maize"], "adm_id": ["AR001"], "date": [20210105], "ndvi": [0.5]}
        ),
        "fpar": pd.DataFrame(
            {"crop_name": ["maize"], "adm_id": ["AR001"], "date": [20210105], "fpar": [50.0]}
        ),
        "soil": pd.DataFrame(
            {
                "crop_name": ["maize"], "adm_id": ["AR001"],
                "awc": [15.0], "bulk_density": [1.4], "drainage_class": [4],
            }
        ),
        "crop_calendar": pd.DataFrame(
            {"crop_name": ["maize"], "adm_id": ["AR001"], "sos": [300.0], "eos": [120.0]}
        ),
        "location": pd.DataFrame(
            {
                "crop_name": ["maize"], "adm_id": ["AR001"],
                "latitude": [-35.0], "longitude": [-60.0], "region_area": [1000],
            }
        ),
        "crop_mask": pd.DataFrame(
            {
                "crop_name": ["maize"], "adm_id": ["AR001"],
                "crop_area": [100.0], "crop_area_percentage": [10.0],
            }
        ),
    }


@pytest.fixture
def miniature_star_workspace(tmp_path, monkeypatch):
    source_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    metadata_path = tmp_path / "data_dictionary.json"
    source_dir.mkdir()
    metadata_path.write_bytes(SOURCE_METADATA_PATH.read_bytes())
    files = {}
    raw_bytes = {}
    for dataset_name, frame in _source_frames().items():
        filename = f"{dataset_name}.csv"
        path = source_dir / filename
        frame.to_csv(path, index=False)
        files[dataset_name] = filename
        raw_bytes[dataset_name] = path.read_bytes()

    for module in (data_preparation, data_quality, star_schema):
        monkeypatch.setattr(module, "INPUT_FILES", files)
        monkeypatch.setattr(module, "INTERIM_DIR", interim_dir)
    for module in (data_quality, star_schema):
        monkeypatch.setattr(module, "QUALITY_REPORT_DIR", reports_dir)
        monkeypatch.setattr(module, "SOURCE_METADATA_PATH", metadata_path)
    monkeypatch.setattr(feature_engineering, "SOURCE_METADATA_PATH", metadata_path)
    monkeypatch.setattr(
        feature_engineering, "FEATURE_REPORT_DIR", reports_dir / "feature_engineering"
    )
    monkeypatch.setattr(data_preparation, "MAIZE_DATA_DIR", source_dir)
    monkeypatch.setattr(star_schema, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(star_schema, "STAR_SCHEMA_DB", processed_dir / "star.duckdb")
    monkeypatch.setattr(feature_engineering, "STAR_SCHEMA_DB", processed_dir / "star.duckdb")

    return {
        "files": files,
        "source_dir": source_dir,
        "raw_bytes": raw_bytes,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
        "reports_dir": reports_dir,
        "metadata_path": metadata_path,
        "database": processed_dir / "star.duckdb",
    }


def test_prepare_audit_and_build_source_grain_star(miniature_star_workspace):
    workspace = miniature_star_workspace

    manifest = data_preparation.prepare_standardized_sources()
    assert sum(item["rows"] for item in manifest) == 11
    prepared_meteorology = pd.read_parquet(
        workspace["interim_dir"] / "meteorology.parquet"
    )
    assert prepared_meteorology["crop_name"].tolist() == ["maize", "maize"]
    assert prepared_meteorology["adm_id"].tolist() == ["AR001", "AR002"]
    assert all(isinstance(value, date) for value in prepared_meteorology["date"])
    assert (workspace["source_dir"] / workspace["files"]["yield"]).read_bytes() == (
        workspace["raw_bytes"]["yield"]
    )

    audit = data_quality.audit_data_quality()
    assert audit["semantic_checks"]["meteorology_tmin_above_tmax"] == 1
    assert audit["semantic_checks"]["meteorology_cwb_formula"]["validation_status"] == "VALIDATED"
    assert audit["semantic_checks"]["crop_calendar_crosses_calendar_year"] == 1
    assert audit["semantic_checks"]["crop_mask_percentage_vs_region_area"][
        "validation_status"
    ] == "CONSISTENT_ON_OVERLAP"
    assert audit["management_scenario"]["status"] == "assumed"
    assert audit["datasets"]["yield"]["candidate_key_duplicates"] == {
        "duplicate_groups": 0,
        "excess_rows": 0,
    }

    report = star_schema.build_star_schema()
    assert report["training_mart_status"] == "BLOCKED"
    assert report["facts"]["fact_yield"] == 2
    assert report["facts"]["fact_meteorology"] == 2
    assert report["management_scenario"]["code"] == "mm0"
    assert report["dataset_bundle"]["release"] == "1.10"
    assert report["administrative_dimension"] == {
        "units": 2,
        "with_location_attributes": 1,
        "without_location_attributes": 1,
    }
    assert report["dimension_reference_unmatched_rows"]
    assert all(value == 0 for value in report["dimension_reference_unmatched_rows"].values())
    season_report = report["season_alignment"]
    assert season_report["checks"]["yield_rows"] == 2
    assert season_report["checks"]["mapped_yield_rows"] == 1
    assert season_report["checks"]["missing_crop_calendar_yield_rows"] == 1
    assert season_report["checks"]["overlapping_campaign_intervals"] == 0
    administrative_dimension = star_schema.query_star(
        "SELECT adm_id, location_attributes_available "
        "FROM dim_administrative_unit ORDER BY adm_id",
        database_path=workspace["database"],
    )
    assert administrative_dimension["adm_id"].tolist() == ["AR001", "AR002"]
    assert administrative_dimension["location_attributes_available"].tolist() == [
        True,
        False,
    ]
    staged_weather = star_schema.query_star(
        """SELECT adm.adm_id, stage.observation_date, stage.season_phase,
                  stage.observation_on_or_before_cutoff,
                  stage.observation_mapping_status
           FROM vw_meteorology_season_stage AS stage
           INNER JOIN dim_administrative_unit AS adm USING (administrative_unit_key)
           ORDER BY adm.adm_id, stage.observation_date""",
        database_path=workspace["database"],
    )
    assert len(staged_weather) == 2
    assert staged_weather.loc[0, "season_phase"] == "mid"
    assert staged_weather.loc[0, "observation_on_or_before_cutoff"]
    assert pd.isna(staged_weather.loc[1, "season_phase"])
    assert staged_weather.loc[1, "observation_mapping_status"] == (
        "CROP_CALENDAR_UNAVAILABLE"
    )
    assert (workspace["reports_dir"] / "season_alignment_report.json").is_file()

    yield_columns = star_schema.query_star(
        "PRAGMA table_info('fact_yield')", database_path=workspace["database"]
    )["name"].tolist()
    assert "season_key" in yield_columns
    assert "date_key" not in yield_columns
    calendar_flags = star_schema.query_star(
        "SELECT crosses_calendar_year FROM fact_crop_calendar",
        database_path=workspace["database"],
    )
    assert calendar_flags["crosses_calendar_year"].tolist() == [True]
    soil_columns = star_schema.query_star(
        "PRAGMA table_info('fact_soil')", database_path=workspace["database"]
    )["name"].tolist()
    assert "drainage_class_key" in soil_columns
    assert "drainage_class" not in soil_columns
    crop_area_unit = star_schema.query_star(
        "SELECT unit FROM source_variable_metadata "
        "WHERE source_dataset = 'crop_mask' AND variable_name = 'crop_area'",
        database_path=workspace["database"],
    ).iloc[0, 0]
    assert crop_area_unit == "km2"

    candidate_tables = feature_engineering.build_candidate_feature_tables(
        database_path=workspace["database"]
    )
    assert [summary["cutoff_fraction"] for summary in candidate_tables] == [
        0.25,
        0.5,
        0.75,
    ]
    assert all(summary["rows"] == 1 for summary in candidate_tables)
    assert all(summary["cutoff_violations"] == 0 for summary in candidate_tables)
    candidate_columns = star_schema.query_star(
        "PRAGMA table_info('candidate_features_cutoff_50')",
        database_path=workspace["database"],
    )["name"].tolist()
    assert {
        "available_water_capacity_cm_m",
        "latitude_deg",
        "longitude_deg",
        "drainage_class_code",
    }.issubset(candidate_columns)
    early_features = star_schema.query_star(
        "SELECT cutoff_date, tavg_mean_to_cutoff, latest_ndvi "
        "FROM candidate_features_cutoff_25",
        database_path=workspace["database"],
    )
    midpoint_features = star_schema.query_star(
        "SELECT cutoff_date, tavg_mean_to_cutoff, latest_ndvi, latest_fpar "
        "FROM candidate_features_cutoff_50",
        database_path=workspace["database"],
    )
    assert pd.isna(early_features.loc[0, "latest_ndvi"])
    assert midpoint_features.loc[0, "latest_ndvi"] == 0.5
    assert midpoint_features.loc[0, "latest_fpar"] == 50.0
    assert midpoint_features.loc[0, "tavg_mean_to_cutoff"] == 7.0
    assert (workspace["reports_dir"] / "data_quality_report.json").is_file()
    assert (workspace["reports_dir"] / "integration_report.json").is_file()

    model_input_report = feature_engineering.prepare_model_input_candidates(
        cutoff_pct=50,
        selected_features=["season_duration_days", "cwb_sum_to_cutoff"],
    )
    assert model_input_report["model_training_performed"] is False
    assert model_input_report["rows"] == 1
    assert model_input_report["additional_candidate_columns_not_evaluated"] == [
        "available_water_capacity_cm_m",
        "latitude_deg",
        "longitude_deg",
        "drainage_class_code",
    ]
    prepared_model_row = star_schema.query_star(
        """SELECT target_yield, available_water_capacity_cm_m,
                  drainage_class_code, latitude_deg, longitude_deg
           FROM model_input_candidates_cutoff_50""",
        database_path=workspace["database"],
    ).iloc[0]
    assert prepared_model_row["target_yield"] == 5.2
    assert prepared_model_row["available_water_capacity_cm_m"] == 15.0
    assert prepared_model_row["drainage_class_code"] == "4"
    assert prepared_model_row["latitude_deg"] == -35.0
    assert prepared_model_row["longitude_deg"] == -60.0
    awc_metadata = star_schema.query_star(
        "SELECT unit, unit_status FROM source_variable_metadata "
        "WHERE source_dataset = 'soil' AND variable_name = 'awc'",
        database_path=workspace["database"],
    ).iloc[0]
    assert awc_metadata["unit"] == "cm m-1"
    assert awc_metadata["unit_status"] == "documented"


def test_preparation_refuses_to_overwrite_without_explicit_request(
    miniature_star_workspace,
):
    data_preparation.prepare_standardized_sources()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        data_preparation.prepare_standardized_sources()


def test_star_build_rejects_quality_report_from_previous_preparation(
    miniature_star_workspace,
):
    data_preparation.prepare_standardized_sources()
    data_quality.audit_data_quality()
    data_preparation.prepare_standardized_sources(overwrite=True)

    with pytest.raises(ValueError, match="no corresponde a la preparacion vigente"):
        star_schema.build_star_schema()


def test_star_build_rejects_quality_report_from_previous_metadata(
    miniature_star_workspace,
):
    data_preparation.prepare_standardized_sources()
    data_quality.audit_data_quality()
    metadata_path = miniature_star_workspace["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metadata_version"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="no corresponde al diccionario de datos vigente"):
        star_schema.build_star_schema()
