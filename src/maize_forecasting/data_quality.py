"""Auditorias de calidad sobre copias estandarizadas por fuente."""

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import duckdb

from maize_forecasting.config import (
    INPUT_FILES,
    INTERIM_DIR,
    QUALITY_REPORT_DIR,
    SOURCE_METADATA_PATH,
)

GRAIN_KEYS = {
    "yield": ["crop_name", "adm_id", "planting_year"],
    "meteorology": ["crop_name", "adm_id", "date"],
    "soil_moisture": ["crop_name", "adm_id", "date"],
    "ndvi": ["crop_name", "adm_id", "date"],
    "fpar": ["crop_name", "adm_id", "date"],
    "soil": ["crop_name", "adm_id"],
    "crop_calendar": ["crop_name", "adm_id"],
    "location": ["crop_name", "adm_id"],
    "crop_mask": ["crop_name", "adm_id"],
}

SOURCE_GRAINS = {
    "yield": "adm_id x crop-season label (planting_year)",
    "meteorology": "adm_id x calendar date",
    "soil_moisture": "adm_id x calendar date",
    "ndvi": "adm_id x dated vegetation observation",
    "fpar": "adm_id x dated vegetation observation",
    "soil": "source-specific adm_id",
    "crop_calendar": "source-specific adm_id",
    "location": "source-specific adm_id",
    "crop_mask": "source-specific adm_id",
}


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _fetch_one(connection, query: str, parameters=None) -> dict[str, object]:
    cursor = connection.execute(query, parameters or [])
    values = cursor.fetchone()
    return {
        name: _json_value(value)
        for name, value in zip((item[0] for item in cursor.description), values)
    }


def _dataset_audit(connection, dataset_name: str, parquet_path: Path) -> dict[str, object]:
    relation = f"read_parquet({_sql_string(parquet_path)})"
    columns = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    column_names = [row[0] for row in columns]
    row_count = connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]

    missing_query = ", ".join(
        f"SUM(CASE WHEN {_sql_identifier(column)} IS NULL THEN 1 ELSE 0 END) "
        f"AS {_sql_identifier(column)}"
        for column in column_names
    )
    missing_values = _fetch_one(connection, f"SELECT {missing_query} FROM {relation}")

    key_columns = GRAIN_KEYS[dataset_name]
    key_sql = ", ".join(_sql_identifier(column) for column in key_columns)
    duplicate_stats = _fetch_one(
        connection,
        f"""
        SELECT COUNT(*) AS duplicate_groups,
               COALESCE(SUM(record_count - 1), 0) AS excess_rows
        FROM (
            SELECT COUNT(*) AS record_count
            FROM {relation}
            GROUP BY {key_sql}
            HAVING COUNT(*) > 1
        )
        """,
    )

    distinct_admins = None
    if "adm_id" in column_names:
        distinct_admins = connection.execute(
            f"SELECT COUNT(DISTINCT adm_id) FROM {relation}"
        ).fetchone()[0]

    temporal = None
    if "date" in column_names:
        temporal = _fetch_one(
            connection,
            f"""
            WITH ordered AS (
                SELECT adm_id, date,
                       date_diff('day', LAG(date) OVER (
                           PARTITION BY adm_id ORDER BY date
                       ), date) AS interval_days
                FROM {relation}
            )
            SELECT MIN(date) AS date_min,
                   MAX(date) AS date_max,
                   COUNT(DISTINCT date) AS distinct_dates,
                   QUANTILE_CONT(interval_days, 0.5) AS median_interval_days,
                   MAX(interval_days) AS max_interval_days,
                   SUM(CASE WHEN interval_days > 1 THEN 1 ELSE 0 END)
                       AS intervals_longer_than_one_day,
                   COUNT(DISTINCT adm_id) FILTER (WHERE interval_days > 30)
                       AS admins_with_gap_over_30_days,
                   COUNT(DISTINCT adm_id) FILTER (WHERE interval_days > 180)
                       AS admins_with_gap_over_180_days
            FROM ordered
            """,
        )

    numeric_ranges: dict[str, dict[str, object]] = {}
    for column_name, column_type, *_ in columns:
        if column_type.upper() not in {
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
        } and not column_type.upper().startswith("DECIMAL("):
            continue
        if column_name == "date":
            continue
        column_sql = _sql_identifier(column_name)
        numeric_ranges[column_name] = _fetch_one(
            connection,
            f"""
            SELECT MIN({column_sql}) AS min,
                   MAX({column_sql}) AS max,
                   SUM(CASE WHEN {column_sql} = 0 THEN 1 ELSE 0 END) AS zero_rows
            FROM {relation}
            """,
        )

    return {
        "dataset": dataset_name,
        "source_file": INPUT_FILES[dataset_name],
        "rows": row_count,
        "columns": [
            {"name": name, "dtype": dtype} for name, dtype, *_ in columns
        ],
        "grain_hypothesis": SOURCE_GRAINS[dataset_name],
        "candidate_key": key_columns,
        "candidate_key_duplicates": duplicate_stats,
        "distinct_adm_id": distinct_admins,
        "missing_values": missing_values,
        "temporal_coverage": temporal,
        "numeric_ranges_and_zeros": numeric_ranges,
        "unit_status": "UNIT_UNVERIFIED",
    }


def audit_data_quality() -> dict[str, object]:
    """Valida granos candidatos, nulos, duplicados, rangos y cadencia temporal."""
    if not SOURCE_METADATA_PATH.is_file():
        raise FileNotFoundError(f"Falta el diccionario de datos: {SOURCE_METADATA_PATH}")
    source_metadata = json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))
    manifest_path = INTERIM_DIR / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Falta preparation_manifest.json; ejecuta prepare_standardized_sources primero."
        )
    preparation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audits: dict[str, dict[str, object]] = {}
    with duckdb.connect(database=":memory:") as connection:
        for dataset_name in INPUT_FILES:
            parquet_path = INTERIM_DIR / f"{dataset_name}.parquet"
            if not parquet_path.is_file():
                raise FileNotFoundError(
                    f"Falta la copia preparada de {dataset_name}: {parquet_path}"
                )
            audits[dataset_name] = _dataset_audit(
                connection, dataset_name, parquet_path
            )
            dataset_metadata = source_metadata["datasets"].get(dataset_name, {})
            observed_columns = {
                column["name"] for column in audits[dataset_name]["columns"]
            }
            documented_columns = set(dataset_metadata.get("documented_columns", []))
            declared_undocumented = set(
                dataset_metadata.get("observed_undocumented_columns", [])
            )
            variable_metadata = dataset_metadata.get("variables", {})
            audits[dataset_name]["metadata"] = {
                "description": dataset_metadata.get("description"),
                "source": dataset_metadata.get("source"),
                "temporal_resolution": dataset_metadata.get("temporal_resolution"),
                "variables": {
                    name: variable_metadata.get(name, {"unit_status": "not_documented"})
                    for name in sorted(observed_columns)
                },
                "observed_columns_not_in_supplied_schema": sorted(
                    observed_columns - documented_columns - declared_undocumented
                )
                if documented_columns
                else [],
            }
            statuses = {
                details.get("unit_status")
                for details in variable_metadata.values()
                if details.get("unit_status")
            }
            measurement_statuses = statuses - {"categorical_code", "derived"}
            if measurement_statuses and measurement_statuses == {"documented"}:
                audits[dataset_name]["unit_status"] = "DOCUMENTED"
            elif measurement_statuses:
                audits[dataset_name]["unit_status"] = "PARTIALLY_DOCUMENTED"
            else:
                audits[dataset_name]["unit_status"] = "UNIT_UNVERIFIED"

        meteorology_path = _sql_string(INTERIM_DIR / "meteorology.parquet")
        calendar_path = _sql_string(INTERIM_DIR / "crop_calendar.parquet")
        yield_path = _sql_string(INTERIM_DIR / "yield.parquet")
        fpar_path = _sql_string(INTERIM_DIR / "fpar.parquet")
        crop_mask_path = _sql_string(INTERIM_DIR / "crop_mask.parquet")
        location_path = _sql_string(INTERIM_DIR / "location.parquet")
        semantic_checks = {
            "meteorology_tmin_above_tmax": connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({meteorology_path}) WHERE tmin > tmax"
            ).fetchone()[0],
            "meteorology_tavg_outside_tmin_tmax": connection.execute(
                f"""SELECT COUNT(*) FROM read_parquet({meteorology_path})
                    WHERE tavg < tmin OR tavg > tmax"""
            ).fetchone()[0],
            "crop_calendar_sos_after_eos": connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({calendar_path}) WHERE sos > eos"
            ).fetchone()[0],
            "crop_calendar_crosses_calendar_year": connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({calendar_path}) WHERE sos > eos"
            ).fetchone()[0],
            "crop_calendar_invalid_day_of_year": connection.execute(
                f"""SELECT COUNT(*) FROM read_parquet({calendar_path})
                    WHERE sos < 1 OR sos > 366 OR eos < 1 OR eos > 366"""
            ).fetchone()[0],
            "yield_zero_yield_rows": connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({yield_path}) WHERE yield = 0"
            ).fetchone()[0],
            "yield_zero_harvest_area_rows": connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({yield_path}) WHERE harvest_area = 0"
            ).fetchone()[0],
            "fpar_values_outside_percentage_range": connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({fpar_path}) WHERE fpar < 0 OR fpar > 100"
            ).fetchone()[0],
            "crop_mask_percentage_outside_range": connection.execute(
                f"""SELECT COUNT(*) FROM read_parquet({crop_mask_path})
                    WHERE crop_area_percentage < 0 OR crop_area_percentage > 100"""
            ).fetchone()[0],
        }
        cwb_rows, cwb_within_tolerance, cwb_max_abs_difference = connection.execute(
            f"""SELECT COUNT(*),
                       SUM(CASE WHEN abs(cwb - (prec - et0)) <= 0.002 THEN 1 ELSE 0 END),
                       MAX(abs(cwb - (prec - et0)))
                FROM read_parquet({meteorology_path})"""
        ).fetchone()
        semantic_checks["meteorology_cwb_formula"] = {
            "formula": "cwb = prec - et0",
            "rows": cwb_rows,
            "rows_within_tolerance_0_002": cwb_within_tolerance,
            "max_absolute_difference": cwb_max_abs_difference,
            "validation_status": (
                "VALIDATED" if cwb_rows == cwb_within_tolerance else "REVIEW"
            ),
        }
        semantic_checks["soil_drainage_class_codes"] = connection.execute(
            f"SELECT DISTINCT drainage_class FROM read_parquet({_sql_string(INTERIM_DIR / 'soil.parquet')}) ORDER BY drainage_class"
        ).fetchnumpy()["drainage_class"].tolist()
        semantic_checks["soil_drainage_class_codes"] = list(
            map(int, semantic_checks["soil_drainage_class_codes"])
        )
        area_rows, area_within_tolerance, area_max_difference = connection.execute(
            f"""SELECT COUNT(*),
                       SUM(CASE WHEN abs(mask.crop_area_percentage
                           - (100.0 * mask.crop_area / location.region_area)) <= 0.1
                           THEN 1 ELSE 0 END),
                       MAX(abs(mask.crop_area_percentage
                           - (100.0 * mask.crop_area / location.region_area)))
                FROM read_parquet({crop_mask_path}) AS mask
                INNER JOIN read_parquet({location_path}) AS location
                  USING (crop_name, adm_id)
                WHERE location.region_area > 0"""
        ).fetchone()
        semantic_checks["crop_mask_percentage_vs_region_area"] = {
            "matched_source_ids": area_rows,
            "rows_within_tolerance_0_1_percentage_points": area_within_tolerance,
            "max_absolute_difference_percentage_points": area_max_difference,
            "validation_scope": "diagnostic on matching source IDs; not a conformed geography join",
            "validation_status": (
                "CONSISTENT_ON_OVERLAP"
                if area_rows == area_within_tolerance
                else "REVIEW"
            ),
        }

        location_relation = f"read_parquet({_sql_string(INTERIM_DIR / 'location.parquet')})"
        coverage_comparisons = {}
        for dataset_name in INPUT_FILES:
            source_relation = f"read_parquet({_sql_string(INTERIM_DIR / f'{dataset_name}.parquet')})"
            coverage_comparisons[dataset_name] = _fetch_one(
                connection,
                f"""
                WITH source_ids AS (
                    SELECT DISTINCT crop_name, adm_id FROM {source_relation}
                ), location_ids AS (
                    SELECT DISTINCT crop_name, adm_id FROM {location_relation}
                )
                SELECT COUNT(*) FILTER (WHERE source.adm_id IS NOT NULL) AS source_ids,
                       COUNT(*) FILTER (
                           WHERE source.adm_id IS NOT NULL AND location.adm_id IS NOT NULL
                       ) AS ids_also_in_location,
                       COUNT(*) FILTER (
                           WHERE source.adm_id IS NOT NULL AND location.adm_id IS NULL
                       ) AS ids_absent_from_location,
                       COUNT(*) FILTER (
                           WHERE source.adm_id IS NULL AND location.adm_id IS NOT NULL
                       ) AS location_ids_absent_from_source
                FROM source_ids AS source
                FULL OUTER JOIN location_ids AS location
                  ON source.crop_name = location.crop_name
                 AND source.adm_id = location.adm_id
                """,
            )

    findings = [
        {
            "severity": "warning",
            "status": "open",
            "issue": "Zero values have not been given field-specific semantics.",
            "action": "Preserve zeros; distinguish true measurements from no-crop/not-applicable states only after source confirmation.",
        },
        {
            "severity": "warning",
            "status": "open",
            "issue": "Some variable units or codebooks remain provisional or unresolved.",
            "action": "Use the variable-level metadata registry; retain unresolved fields without conversions or inferred category labels.",
        },
        {
            "severity": "warning",
            "status": "open",
            "issue": "Season labels in yield do not map to daily observations through a validated crop calendar.",
            "action": "Keep yield on dim_season and dated sources on dim_date; do not join on harvest_year/calendar year.",
        },
        {
            "severity": "warning",
            "status": "open",
            "issue": "The sources cover different subsets of administrative units; location attributes are not present for every ID.",
            "action": "Use the CY-Bench v1.10 conformed adm_id dimension and preserve null location attributes for IDs absent from location.csv.",
        },
    ]
    if semantic_checks["meteorology_tmin_above_tmax"]:
        findings.append(
            {
                "severity": "warning",
                "status": "open",
                "issue": "Some meteorology rows have tmin > tmax.",
                "count": semantic_checks["meteorology_tmin_above_tmax"],
                "action": "Review source definitions/calculation; retain rows and flag for EDA.",
            }
        )
    if semantic_checks["meteorology_tavg_outside_tmin_tmax"]:
        findings.append(
            {
                "severity": "warning",
                "status": "open",
                "issue": "Some meteorology rows have tavg outside [tmin, tmax].",
                "count": semantic_checks["meteorology_tavg_outside_tmin_tmax"],
                "action": "Review source definitions/calculation; do not silently recalculate tavg.",
            }
        )
    if semantic_checks["crop_calendar_sos_after_eos"]:
        findings.append(
            {
                "severity": "info",
                "status": "interpreted_from_documented_DOY_semantics",
                "issue": "The crop-calendar season crosses the calendar-year boundary when sos > eos.",
                "count": semantic_checks["crop_calendar_sos_after_eos"],
                "action": "A crosses_calendar_year flag is safe; do not synthesize actual calendar dates without a reference year.",
            }
        )
    if semantic_checks["meteorology_cwb_formula"]["validation_status"] == "VALIDATED":
        findings.append(
            {
                "severity": "info",
                "status": "validated",
                "issue": "cwb matches prec - et0 within floating-point/rounding tolerance.",
                **semantic_checks["meteorology_cwb_formula"],
            }
        )
    if semantic_checks["fpar_values_outside_percentage_range"]:
        findings.append(
            {
                "severity": "warning",
                "status": "open",
                "issue": "FPAR has values outside its documented percentage range [0, 100].",
                "count": semantic_checks["fpar_values_outside_percentage_range"],
            }
        )
    if semantic_checks["crop_mask_percentage_outside_range"]:
        findings.append(
            {
                "severity": "warning",
                "status": "open",
                "issue": "Crop-area percentages fall outside [0, 100].",
                "count": semantic_checks["crop_mask_percentage_outside_range"],
            }
        )
    if semantic_checks["crop_mask_percentage_vs_region_area"]["validation_status"] == "CONSISTENT_ON_OVERLAP":
        findings.append(
            {
                "severity": "info",
                "status": "consistent_on_overlap",
                "issue": "crop_area_percentage agrees with crop_area / region_area on IDs present in both files.",
                **semantic_checks["crop_mask_percentage_vs_region_area"],
            }
        )
    for dataset_name, temporal in (
        (name, audit["temporal_coverage"])
        for name, audit in audits.items()
        if audit["temporal_coverage"] is not None
    ):
        if temporal["max_interval_days"] and temporal["max_interval_days"] > 1:
            findings.append(
                {
                    "severity": "info" if dataset_name == "fpar" else "warning",
                    "status": "open",
                    "issue": f"{dataset_name} has observed intervals longer than one day.",
                    "max_interval_days": temporal["max_interval_days"],
                    "admins_with_gap_over_30_days": temporal[
                        "admins_with_gap_over_30_days"
                    ],
                    "admins_with_gap_over_180_days": temporal[
                        "admins_with_gap_over_180_days"
                    ],
                    "action": (
                        "Treat as observation cadence/gap information, not as null values; "
                        "confirm expected acquisition schedule before resampling."
                    ),
                }
            )

    report = {
        "status": "PASS_WITH_WARNINGS",
        "preparation_id": preparation_manifest["preparation_id"],
        "source_metadata_registry": str(SOURCE_METADATA_PATH),
        "source_metadata_sha256": hashlib.sha256(
            SOURCE_METADATA_PATH.read_bytes()
        ).hexdigest(),
        "dataset_bundle": source_metadata["dataset_bundle"],
        "version_history": source_metadata["version_history"],
        "management_scenario": source_metadata["management_scenario"],
        "raw_sources_modified": False,
        "transformations_applied": [
            "identifier whitespace/case normalization",
            "YYYYMMDD codes parsed to DATE in prepared copies",
        ],
        "imputation_performed": False,
        "unit_conversion_performed": False,
        "datasets": audits,
        "schema_reconciliation": source_metadata.get("schema_reconciliation", {}),
        "semantic_checks": semantic_checks,
        "admin_id_overlap_diagnostics_vs_location": coverage_comparisons,
        "findings": findings,
    }

    QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = QUALITY_REPORT_DIR / "data_quality_report.json"
    temporary_path = report_path.with_name(f".{report_path.stem}.{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_path.replace(report_path)
    return report
