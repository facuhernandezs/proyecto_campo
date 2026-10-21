"""Construccion del modelo estrella analitico con granos fuente preservados."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

from maize_forecasting.config import (
    INPUT_FILES,
    INTERIM_DIR,
    PROCESSED_DIR,
    QUALITY_REPORT_DIR,
    SOURCE_METADATA_PATH,
    STAR_SCHEMA_DB,
)
from maize_forecasting.data_quality import GRAIN_KEYS, SOURCE_GRAINS
from maize_forecasting.season_calendar import build_season_alignment

TEMPORAL_DATASETS = ("meteorology", "soil_moisture", "ndvi", "fpar")
FACTS = {
    "yield": ("fact_yield", "season"),
    "meteorology": ("fact_meteorology", "date"),
    "soil_moisture": ("fact_soil_moisture", "date"),
    "ndvi": ("fact_ndvi", "date"),
    "fpar": ("fact_fpar", "date"),
    "soil": ("fact_soil", "static"),
    "crop_calendar": ("fact_crop_calendar", "static"),
    "crop_mask": ("fact_crop_mask", "static"),
}

def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _read_parquet(dataset_name: str) -> str:
    return f"read_parquet({_sql_string(INTERIM_DIR / f'{dataset_name}.parquet')})"


def _create_dimensions(connection: duckdb.DuckDBPyConnection) -> None:
    crop_selects = " UNION ALL ".join(
        f"SELECT crop_name FROM {_read_parquet(dataset_name)}"
        for dataset_name in INPUT_FILES
    )
    connection.execute(
        f"""
        CREATE TABLE dim_crop AS
        SELECT ROW_NUMBER() OVER (ORDER BY crop_name)::INTEGER AS crop_key,
               crop_name
        FROM (SELECT DISTINCT crop_name FROM ({crop_selects}))
        """
    )

    admin_selects = " UNION ALL ".join(
        f"SELECT adm_id FROM {_read_parquet(dataset_name)}"
        for dataset_name in INPUT_FILES
    )
    source_metadata = json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))
    bundle = source_metadata["dataset_bundle"]
    country_code = bundle["country_code"]
    connection.execute(
        f"""
        CREATE TABLE dim_administrative_unit AS
        WITH all_units AS (
            SELECT DISTINCT adm_id FROM ({admin_selects})
        ), location AS (
            SELECT adm_id, latitude, longitude, region_area
            FROM {_read_parquet('location')}
        )
        SELECT ROW_NUMBER() OVER (ORDER BY units.adm_id)::BIGINT
                   AS administrative_unit_key,
               {_sql_string(country_code)} AS country_code,
               units.adm_id,
               location.latitude,
               location.longitude,
               location.region_area,
               location.adm_id IS NOT NULL AS location_attributes_available,
               CAST(NULL AS VARCHAR) AS coordinate_reference_system
        FROM all_units AS units
        LEFT JOIN location USING (adm_id)
        """
    )

    date_selects = " UNION ALL ".join(
        f"SELECT date FROM {_read_parquet(dataset_name)}"
        for dataset_name in TEMPORAL_DATASETS
    )
    connection.execute(
        f"""
        CREATE TABLE dim_date AS
        WITH limits AS (
            SELECT MIN(date)::TIMESTAMP AS first_date,
                   MAX(date)::TIMESTAMP AS last_date
            FROM ({date_selects})
        ), calendar AS (
            SELECT CAST(calendar_date AS DATE) AS calendar_date
            FROM limits,
                 generate_series(first_date, last_date, INTERVAL 1 DAY) AS dates(calendar_date)
        )
        SELECT CAST(strftime(calendar_date, '%Y%m%d') AS INTEGER) AS date_key,
               calendar_date,
               EXTRACT(YEAR FROM calendar_date)::SMALLINT AS year,
               EXTRACT(MONTH FROM calendar_date)::TINYINT AS month,
               EXTRACT(DAY FROM calendar_date)::TINYINT AS day,
               EXTRACT(DOY FROM calendar_date)::SMALLINT AS day_of_year,
               EXTRACT(ISODOW FROM calendar_date)::TINYINT AS iso_day_of_week
        FROM calendar
        """
    )

    season_relation = _read_parquet("yield")
    conflicts = connection.execute(
        f"""
        SELECT planting_year
        FROM {season_relation}
        GROUP BY planting_year
        HAVING COUNT(DISTINCT harvest_year) > 1
            OR COUNT(DISTINCT planting_date) > 1
            OR COUNT(DISTINCT harvest_date) > 1
        """
    ).fetchall()
    if conflicts:
        raise ValueError(
            "No se puede crear dim_season: etiquetas de campana ambiguas en yield: "
            + ", ".join(str(row[0]) for row in conflicts)
        )

    connection.execute(
        f"""
        CREATE TABLE dim_season AS
        SELECT ROW_NUMBER() OVER (ORDER BY crop_key, season_label)::INTEGER AS season_key,
               crop_key,
               season_label,
               harvest_year,
               planting_date AS planting_date_source_label,
               harvest_date AS harvest_date_source_label
        FROM (
            SELECT DISTINCT crop.crop_key,
                   source.planting_year AS season_label,
                   source.harvest_year,
                   source.planting_date,
                   source.harvest_date
            FROM {season_relation} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
        )
        """
    )

    connection.execute(
        f"""
        CREATE TABLE dim_drainage_class AS
        SELECT ROW_NUMBER() OVER (ORDER BY source_class_code)::INTEGER
                   AS drainage_class_key,
               source_class_code,
               CAST(NULL AS VARCHAR) AS category_label,
               'codebook_unresolved' AS label_status
        FROM (
            SELECT DISTINCT drainage_class AS source_class_code
            FROM {_read_parquet('soil')}
        )
        """
    )

    scenario = source_metadata["management_scenario"]
    source_rows = []
    for dataset_name, filename in INPUT_FILES.items():
        dataset_metadata = source_metadata["datasets"].get(dataset_name, {})
        source_rows.append(
            (
                dataset_name,
                filename,
                SOURCE_GRAINS[dataset_name],
                dataset_metadata.get("temporal_resolution"),
                dataset_metadata.get("source"),
                bundle["release"],
                bundle["crop"],
                bundle["country_code"],
                scenario["code"],
                scenario["description"],
                scenario["status"],
            )
        )
    connection.execute(
        """
        CREATE TABLE dim_source_dataset (
            source_dataset VARCHAR,
            source_file VARCHAR,
            grain_hypothesis VARCHAR,
            temporal_resolution VARCHAR,
            source_reference VARCHAR,
            dataset_release VARCHAR,
            crop_name VARCHAR,
            country_code VARCHAR,
            management_scenario_code VARCHAR,
            management_scenario_description VARCHAR,
            management_scenario_status VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO dim_source_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        source_rows,
    )

    variable_rows = []
    for dataset_name, dataset_metadata in source_metadata["datasets"].items():
        for variable_name, details in dataset_metadata.get("variables", {}).items():
            variable_rows.append(
                (
                    dataset_name,
                    variable_name,
                    details.get("definition"),
                    details.get("unit"),
                    details.get("unit_status", "not_documented"),
                    dataset_metadata.get("source"),
                    json.dumps(details),
                )
            )
    connection.execute(
        """
        CREATE TABLE source_variable_metadata (
            source_dataset VARCHAR,
            variable_name VARCHAR,
            definition VARCHAR,
            unit VARCHAR,
            unit_status VARCHAR,
            source_reference VARCHAR,
            metadata_json VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO source_variable_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
        variable_rows,
    )


def _create_facts(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    prepared_counts = {
        name: connection.execute(f"SELECT COUNT(*) FROM {_read_parquet(name)}").fetchone()[0]
        for name in INPUT_FILES
    }

    for dataset_name, key_columns in GRAIN_KEYS.items():
        relation = _read_parquet(dataset_name)
        group_columns = ", ".join(key_columns)
        duplicate_count = connection.execute(
            f"""SELECT COUNT(*) FROM (
                SELECT {group_columns} FROM {relation}
                GROUP BY {group_columns} HAVING COUNT(*) > 1
            )"""
        ).fetchone()[0]
        if duplicate_count:
            raise ValueError(
                f"El grano candidato de {dataset_name} tiene "
                f"{duplicate_count} claves duplicadas; no se materializa el hecho."
            )

    fact_specs = {
        "yield": """
            SELECT crop.crop_key, admin.administrative_unit_key, season.season_key,
                   source.country_code,
                   source.planting_date AS planting_date_source_label,
                   source.harvest_date AS harvest_date_source_label,
                   source."yield" AS yield,
                   source.production,
                   source.harvest_area,
                   source.planted_area
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
            INNER JOIN dim_season AS season
              ON season.crop_key = crop.crop_key
             AND season.season_label = source.planting_year
        """,
        "meteorology": """
            SELECT crop.crop_key, admin.administrative_unit_key, calendar.date_key,
                   source.tmin, source.tmax, source.prec, source.rad,
                   source.tavg, source.et0, source.vpd, source.cwb
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
            INNER JOIN dim_date AS calendar ON calendar.calendar_date = source.date
        """,
        "soil_moisture": """
            SELECT crop.crop_key, admin.administrative_unit_key, calendar.date_key,
                   source.ssm, source.rsm
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
            INNER JOIN dim_date AS calendar ON calendar.calendar_date = source.date
        """,
        "ndvi": """
            SELECT crop.crop_key, admin.administrative_unit_key, calendar.date_key, source.ndvi
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
            INNER JOIN dim_date AS calendar ON calendar.calendar_date = source.date
        """,
        "fpar": """
            SELECT crop.crop_key, admin.administrative_unit_key, calendar.date_key, source.fpar
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
            INNER JOIN dim_date AS calendar ON calendar.calendar_date = source.date
        """,
        "soil": """
            SELECT crop.crop_key, admin.administrative_unit_key,
                   source.awc, source.bulk_density, drainage.drainage_class_key
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
            INNER JOIN dim_drainage_class AS drainage
              ON drainage.source_class_code = source.drainage_class
        """,
        "crop_calendar": """
            SELECT crop.crop_key, admin.administrative_unit_key, source.sos, source.eos,
                   source.sos > source.eos AS crosses_calendar_year
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
        """,
        "crop_mask": """
            SELECT crop.crop_key, admin.administrative_unit_key,
                   source.crop_area, source.crop_area_percentage
            FROM {source} AS source
            INNER JOIN dim_crop AS crop USING (crop_name)
            INNER JOIN dim_administrative_unit AS admin
              ON admin.adm_id = source.adm_id
        """,
    }

    fact_counts: dict[str, int] = {}
    for dataset_name, select_sql in fact_specs.items():
        fact_table = FACTS[dataset_name][0]
        connection.execute(
            f"CREATE TABLE {fact_table} AS "
            + select_sql.format(source=_read_parquet(dataset_name))
        )
        actual_count = connection.execute(
            f"SELECT COUNT(*) FROM {fact_table}"
        ).fetchone()[0]
        if actual_count != prepared_counts[dataset_name]:
            raise ValueError(
                f"La materializacion de {fact_table} cambio el conteo: "
                f"{prepared_counts[dataset_name]} -> {actual_count}."
            )
        fact_counts[fact_table] = actual_count

    location_count = connection.execute(
        "SELECT COUNT(*) FROM dim_administrative_unit WHERE location_attributes_available"
    ).fetchone()[0]
    if location_count != prepared_counts["location"]:
        raise ValueError("La dimension location perdio filas durante su materializacion.")

    return fact_counts


def _create_source_coverage(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, dict[str, object]]:
    connection.execute(
        """
        CREATE TABLE source_coverage (
            source_dataset VARCHAR,
            rows BIGINT,
            distinct_adm_id BIGINT,
            date_min DATE,
            date_max DATE,
            distinct_dates BIGINT,
            harvest_year_min INTEGER,
            harvest_year_max INTEGER,
            distinct_seasons BIGINT
        )
        """
    )
    coverage: dict[str, dict[str, object]] = {}
    for dataset_name in INPUT_FILES:
        relation = _read_parquet(dataset_name)
        if dataset_name in TEMPORAL_DATASETS:
            values = connection.execute(
                f"""SELECT COUNT(*), COUNT(DISTINCT adm_id), MIN(date), MAX(date),
                           COUNT(DISTINCT date), NULL, NULL, NULL
                    FROM {relation}"""
            ).fetchone()
        elif dataset_name == "yield":
            values = connection.execute(
                f"""SELECT COUNT(*), COUNT(DISTINCT adm_id), NULL, NULL, NULL,
                           MIN(harvest_year), MAX(harvest_year),
                           COUNT(DISTINCT planting_year)
                    FROM {relation}"""
            ).fetchone()
        else:
            values = connection.execute(
                f"""SELECT COUNT(*), COUNT(DISTINCT adm_id), NULL, NULL, NULL,
                           NULL, NULL, NULL
                    FROM {relation}"""
            ).fetchone()

        connection.execute(
            "INSERT INTO source_coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [dataset_name, *values],
        )
        coverage[dataset_name] = {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in zip(
                (
                    "rows",
                    "distinct_adm_id",
                    "date_min",
                    "date_max",
                    "distinct_dates",
                    "harvest_year_min",
                    "harvest_year_max",
                    "distinct_seasons",
                ),
                values,
            )
        }
    return coverage


def _validate_foreign_keys(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    results = {}
    for table_name, _ in FACTS.values():
        for key, dimension, dimension_key in (
            ("crop_key", "dim_crop", "crop_key"),
            (
                "administrative_unit_key",
                "dim_administrative_unit",
                "administrative_unit_key",
            ),
        ):
            count = connection.execute(
                f"""SELECT COUNT(*) FROM {table_name} AS fact
                    LEFT JOIN {dimension} AS dim
                      ON fact.{key} = dim.{dimension_key}
                    WHERE dim.{dimension_key} IS NULL"""
            ).fetchone()[0]
            results[f"{table_name}.{key}"] = count

    for table_name in ("fact_meteorology", "fact_soil_moisture", "fact_ndvi", "fact_fpar"):
        count = connection.execute(
            f"""SELECT COUNT(*) FROM {table_name} AS fact
                LEFT JOIN dim_date AS dim ON fact.date_key = dim.date_key
                WHERE dim.date_key IS NULL"""
        ).fetchone()[0]
        results[f"{table_name}.date_key"] = count

    count = connection.execute(
        """SELECT COUNT(*) FROM fact_yield AS fact
            LEFT JOIN dim_season AS dim ON fact.season_key = dim.season_key
            WHERE dim.season_key IS NULL"""
    ).fetchone()[0]
    results["fact_yield.season_key"] = count

    count = connection.execute(
        """SELECT COUNT(*) FROM fact_soil AS fact
            LEFT JOIN dim_drainage_class AS dim
              ON fact.drainage_class_key = dim.drainage_class_key
            WHERE dim.drainage_class_key IS NULL"""
    ).fetchone()[0]
    results["fact_soil.drainage_class_key"] = count
    if any(results.values()):
        raise ValueError(f"Referencias dimensionales sin correspondencia: {results}")
    return results


def build_star_schema(*, overwrite: bool = False) -> dict[str, object]:
    """Materializa dimensiones/facts source-grain en un archivo DuckDB.

    La dimension administrativa comun usa el alcance/correccion de IDs de
    CY-Bench v1.10 y conserva atributos de ubicacion opcionales.
    """
    missing_sources = [
        str(INTERIM_DIR / f"{name}.parquet")
        for name in INPUT_FILES
        if not (INTERIM_DIR / f"{name}.parquet").is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Faltan copias estandarizadas; ejecuta prepare_standardized_sources primero: "
            + ", ".join(missing_sources)
        )

    manifest_path = INTERIM_DIR / "preparation_manifest.json"
    quality_report_path = QUALITY_REPORT_DIR / "data_quality_report.json"
    if (
        not manifest_path.is_file()
        or not quality_report_path.is_file()
        or not SOURCE_METADATA_PATH.is_file()
    ):
        raise FileNotFoundError(
            "Falta preparacion, reporte de calidad o metadata registry; "
            "ejecuta primero el flujo de preparacion y auditoria."
        )
    preparation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    source_metadata = json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))
    if quality_report.get("preparation_id") != preparation_manifest.get("preparation_id"):
        raise ValueError(
            "El reporte de calidad no corresponde a la preparacion vigente; "
            "vuelve a ejecutar audit_data_quality()."
        )
    metadata_sha256 = hashlib.sha256(SOURCE_METADATA_PATH.read_bytes()).hexdigest()
    if quality_report.get("source_metadata_sha256") != metadata_sha256:
        raise ValueError(
            "El reporte de calidad no corresponde al diccionario de datos vigente; "
            "vuelve a ejecutar audit_data_quality()."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if STAR_SCHEMA_DB.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe la base {STAR_SCHEMA_DB}; usa overwrite=True para regenerarla."
        )

    temporary_db = STAR_SCHEMA_DB.with_name(
        f".{STAR_SCHEMA_DB.stem}.{uuid4().hex}.duckdb"
    )
    connection = duckdb.connect(str(temporary_db))
    try:
        _create_dimensions(connection)
        fact_counts = _create_facts(connection)
        source_coverage = _create_source_coverage(connection)
        season_alignment = build_season_alignment(connection)
        foreign_key_checks = _validate_foreign_keys(connection)
        location_available_count = connection.execute(
            """SELECT COUNT(*) FROM dim_administrative_unit
               WHERE location_attributes_available"""
        ).fetchone()[0]
        table_names = [
            row[0]
            for row in connection.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                   ORDER BY table_name"""
            ).fetchall()
        ]
        table_counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in table_names
        }
    except Exception:
        connection.close()
        temporary_db.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    os.replace(temporary_db, STAR_SCHEMA_DB)
    report = {
        "integration_status": "PASS_WITH_WARNINGS",
        "preparation_id": preparation_manifest["preparation_id"],
        "database": str(STAR_SCHEMA_DB),
        "storage": "DuckDB analytical database",
        "raw_sources_modified": False,
        "source_facts_preserved_at_native_grain": True,
        "dataset_bundle": source_metadata["dataset_bundle"],
        "version_history": source_metadata["version_history"],
        "dimensions": [
            "dim_crop",
            "dim_date",
            "dim_season",
            "dim_source_dataset",
            "dim_administrative_unit",
            "dim_drainage_class",
        ],
        "metadata_tables": ["source_variable_metadata", "source_coverage"],
        "season_alignment_tables": ["bridge_administrative_season"],
        "season_alignment_views": [
            "vw_yield_midseason_calendar",
            *[
                f"vw_{dataset_name}_season_stage"
                for dataset_name in TEMPORAL_DATASETS
            ],
        ],
        "facts": fact_counts,
        "datasets": [
            {
                "name": name,
                "grain": SOURCE_GRAINS[name],
                "candidate_key": GRAIN_KEYS[name],
                "spatial_key": ["crop_name", "adm_id"],
                "temporal_key": (
                    "planting_year (source season label)"
                    if name == "yield"
                    else "date"
                    if name in TEMPORAL_DATASETS
                    else None
                ),
                "temporal_resolution": source_metadata["datasets"][name].get(
                    "temporal_resolution"
                ),
                "source_reference": source_metadata["datasets"][name].get("source"),
                "rows": int(source_coverage[name]["rows"]),
                "distinct_adm_id": int(source_coverage[name]["distinct_adm_id"]),
                "missing_cells": sum(
                    quality_report["datasets"][name]["missing_values"].values()
                ),
                "duplicate_groups": quality_report["datasets"][name][
                    "candidate_key_duplicates"
                ]["duplicate_groups"],
                "unit_status": quality_report["datasets"][name]["unit_status"],
                "variable_metadata": quality_report["datasets"][name]["metadata"][
                    "variables"
                ],
            }
            for name in INPUT_FILES
        ],
        "source_coverage": source_coverage,
        "administrative_dimension": {
            "units": table_counts["dim_administrative_unit"],
            "with_location_attributes": location_available_count,
            "without_location_attributes": (
                table_counts["dim_administrative_unit"] - location_available_count
            ),
        },
        "table_counts": table_counts,
        "dimension_reference_unmatched_rows": foreign_key_checks,
        "geography_strategy": "common adm_id dimension per CY-Bench v1.10; all source IDs map to one administrative unit, location attributes remain optional",
        "spatial_alignment_status": {
            "admin_id": "conformed_by_version_notes_1.1_1.2",
            "country_scope": "AR only",
            "coordinate_reference_system": "unresolved",
            "polygon_geometry": "not present in local CSV extract",
        },
        "metadata_registry": str(SOURCE_METADATA_PATH),
        "source_metadata_sha256": metadata_sha256,
        "management_scenario": source_metadata["management_scenario"],
        "schema_reconciliation": quality_report.get("schema_reconciliation", {}),
        "transformations_with_domain_metadata": [
            "cwb formula verified against prec - et0 within tolerance 0.002",
            "sos > eos represented as crosses_calendar_year using documented day_of_year semantics",
            "drainage_class modeled as a categorical code with unresolved labels",
        ],
        "temporal_strategy": {
            "dated_sources": "DATE values in dim_date; facts remain at source observation grain",
            "yield": "planting_year label is the season id; its start year agrees with harvest_year = start year + 1 for all labels",
            "coverage_difference": "all rows retained; 2001-2023 calendar-year overlap is not treated as crop-season alignment",
            "campaign_window": "static SOS/EOS projected onto the start year in planting_year; EOS remains growing-season end, not harvest date",
            "prediction_cutoff": "midpoint of the local static SOS/EOS window, rounded to a whole date",
            "observation_phase": "generic thirds early/mid/late; no named phenological stages inferred",
            "availability": "source date is used as availability proxy; no separate publication timestamp exists",
            "aggregation": "none; seasonal stage views expose dated observations without creating features",
        },
        "prediction_specification": {
            "target": "final yield",
            "one_prediction_per": "administrative unit x crop season",
            "forecast_checkpoint": "midpoint of local static crop-calendar window",
            "information_cutoff": "calendar date at local midpoint; observations are flagged by date <= cutoff",
        },
        "season_alignment": season_alignment,
        "unit_status": "variable-level; see source_variable_metadata",
        "joins": [
            {
                "relationship": "N:1",
                "from": "each fact to dim_crop and dim_administrative_unit",
                "validation": "all foreign-key lookups matched; fact row counts preserved",
            },
            {
                "relationship": "N:1",
                "from": "dated facts to dim_date",
                "validation": "all date keys matched",
            },
            {
                "relationship": "N:1",
                "from": "fact_yield to dim_season",
                "validation": "season labels unambiguous in source and all keys matched",
            },
            {
                "relationship": "1:N",
                "from": "bridge_administrative_season to dated environmental observations by admin_id/date interval",
                "validation": "no overlapping mapped seasons; views preserve each source fact row",
            },
        ],
        "training_mart_status": "BLOCKED",
        "training_mart_blockers": [
            "variable selection and feature aggregation have not been performed",
            "228 yield admin-season rows lack crop_calendar and have no midseason cutoff",
            "some source units and category codebooks remain provisional or unresolved",
            "publication timestamps are unavailable; observation dates are used as an availability proxy",
        ],
    }

    QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = QUALITY_REPORT_DIR / "integration_report.json"
    temporary_report = report_path.with_name(f".{report_path.stem}.{uuid4().hex}.tmp")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    season_report_path = QUALITY_REPORT_DIR / "season_alignment_report.json"
    temporary_season_report = season_report_path.with_name(
        f".{season_report_path.stem}.{uuid4().hex}.tmp"
    )
    temporary_season_report.write_text(
        json.dumps(season_alignment, indent=2), encoding="utf-8"
    )
    temporary_season_report.replace(season_report_path)
    return report


def refresh_metadata_catalog() -> dict[str, object]:
    """Sync variable metadata into DuckDB without rebuilding large fact tables."""
    if not STAR_SCHEMA_DB.is_file():
        raise FileNotFoundError(f"No existe la base DuckDB: {STAR_SCHEMA_DB}")
    quality_path = QUALITY_REPORT_DIR / "data_quality_report.json"
    manifest_path = INTERIM_DIR / "preparation_manifest.json"
    if not quality_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Se requiere el reporte de calidad y el manifest vigentes.")

    source_metadata = json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))
    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    preparation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_sha256 = hashlib.sha256(SOURCE_METADATA_PATH.read_bytes()).hexdigest()
    if quality_report.get("preparation_id") != preparation_manifest.get("preparation_id"):
        raise ValueError("El reporte de calidad no corresponde a la preparacion vigente.")
    if quality_report.get("source_metadata_sha256") != metadata_sha256:
        raise ValueError("El reporte de calidad no corresponde a la metadata vigente.")

    scenario = source_metadata["management_scenario"]
    bundle = source_metadata["dataset_bundle"]
    variable_rows = [
        (
            dataset_name,
            variable_name,
            details.get("definition"),
            details.get("unit"),
            details.get("unit_status", "not_documented"),
            dataset_metadata.get("source"),
            json.dumps(details),
        )
        for dataset_name, dataset_metadata in source_metadata["datasets"].items()
        for variable_name, details in dataset_metadata.get("variables", {}).items()
    ]

    with duckdb.connect(str(STAR_SCHEMA_DB)) as connection:
        connection.execute("DELETE FROM source_variable_metadata")
        connection.executemany(
            "INSERT INTO source_variable_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            variable_rows,
        )
        for dataset_name, dataset_metadata in source_metadata["datasets"].items():
            connection.execute(
                """UPDATE dim_source_dataset
                   SET temporal_resolution = ?, source_reference = ?,
                       dataset_release = ?, crop_name = ?, country_code = ?,
                       management_scenario_code = ?,
                       management_scenario_description = ?,
                       management_scenario_status = ?
                   WHERE source_dataset = ?""",
                [
                    dataset_metadata.get("temporal_resolution"),
                    dataset_metadata.get("source"),
                    bundle["release"],
                    bundle["crop"],
                    bundle["country_code"],
                    scenario["code"],
                    scenario["description"],
                    scenario["status"],
                    dataset_name,
                ],
            )
        metadata_count = connection.execute(
            "SELECT COUNT(*) FROM source_variable_metadata"
        ).fetchone()[0]

    integration_path = QUALITY_REPORT_DIR / "integration_report.json"
    integration_report = json.loads(integration_path.read_text(encoding="utf-8"))
    integration_report["metadata_registry"] = str(SOURCE_METADATA_PATH)
    integration_report["source_metadata_sha256"] = metadata_sha256
    integration_report["dataset_bundle"] = bundle
    integration_report["version_history"] = source_metadata["version_history"]
    integration_report["management_scenario"] = scenario
    integration_report["schema_reconciliation"] = quality_report.get(
        "schema_reconciliation", {}
    )
    for dataset in integration_report["datasets"]:
        quality_dataset = quality_report["datasets"][dataset["name"]]
        dataset["unit_status"] = quality_dataset["unit_status"]
        dataset["variable_metadata"] = quality_dataset["metadata"]["variables"]
        dataset["source_reference"] = source_metadata["datasets"][
            dataset["name"]
        ].get("source")
    integration_report["table_counts"]["source_variable_metadata"] = metadata_count

    temporary_report = integration_path.with_name(
        f".{integration_path.stem}.{uuid4().hex}.tmp"
    )
    temporary_report.write_text(json.dumps(integration_report, indent=2), encoding="utf-8")
    temporary_report.replace(integration_path)
    return {
        "metadata_sha256": metadata_sha256,
        "source_variable_metadata_rows": metadata_count,
        "integration_status": integration_report["integration_status"],
    }


def query_star(query: str, *, database_path: Path = STAR_SCHEMA_DB) -> pd.DataFrame:
    """Ejecuta una consulta SQL de solo lectura sobre el modelo estrella."""
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(query).df()
