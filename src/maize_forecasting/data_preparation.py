"""Preparacion reproducible de copias estandarizadas de los CSV originales."""

import json
import os
from pathlib import Path
from uuid import uuid4

import duckdb

from maize_forecasting.config import INPUT_FILES, INTERIM_DIR, MAIZE_DATA_DIR


def _sql_string(value: str | Path) -> str:
    """Escapa un literal SQL para rutas locales controladas por el proyecto."""
    return "'" + str(value).replace("'", "''") + "'"


def _standardized_query(dataset_name: str, source_path: Path) -> str:
    replacements = [
        "lower(trim(CAST(crop_name AS VARCHAR))) AS crop_name",
        "upper(trim(CAST(adm_id AS VARCHAR))) AS adm_id",
    ]
    if dataset_name == "yield":
        replacements.append("upper(trim(CAST(country_code AS VARCHAR))) AS country_code")
    if dataset_name in {"meteorology", "soil_moisture", "ndvi", "fpar"}:
        replacements.append(
            "strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE AS date"
        )

    replace_sql = ", ".join(replacements)
    return (
        f"SELECT * REPLACE ({replace_sql}) "
        f"FROM read_csv_auto({_sql_string(source_path)})"
    )


def prepare_standardized_sources(*, overwrite: bool = False) -> list[dict[str, object]]:
    """Crea copias Parquet estandarizadas, preservando intactos los CSV raw.

    Solo normaliza espacios/case de identificadores y parsea `date` a DATE.
    No imputa, elimina filas, convierte unidades ni intenta alinear campanas.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {name: INTERIM_DIR / f"{name}.parquet" for name in INPUT_FILES}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Ya existen copias preparadas; usa overwrite=True para regenerarlas: "
            + ", ".join(str(path) for path in existing)
        )

    staged: list[tuple[Path, Path]] = []
    manifest: list[dict[str, object]] = []
    preparation_id = uuid4().hex
    connection = duckdb.connect(database=":memory:")
    try:
        for dataset_name, filename in INPUT_FILES.items():
            source_path = MAIZE_DATA_DIR / filename
            if not source_path.is_file():
                raise FileNotFoundError(f"No se encontro el CSV fuente: {source_path}")

            output_path = outputs[dataset_name]
            temporary_path = output_path.with_name(
                f".{output_path.stem}.{uuid4().hex}.tmp.parquet"
            )
            query = _standardized_query(dataset_name, source_path)
            connection.execute(
                f"COPY ({query}) TO {_sql_string(temporary_path)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            staged.append((temporary_path, output_path))
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({_sql_string(temporary_path)})"
            ).fetchone()[0]
            manifest.append(
                {
                    "dataset": dataset_name,
                    "source_file": filename,
                    "source_bytes": source_path.stat().st_size,
                    "prepared_file": str(output_path),
                    "prepared_bytes": temporary_path.stat().st_size,
                    "rows": row_count,
                }
            )
    except Exception:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    for temporary_path, output_path in staged:
        os.replace(temporary_path, output_path)

    manifest_path = INTERIM_DIR / "preparation_manifest.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.stem}.{uuid4().hex}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(
            {
                "preparation_id": preparation_id,
                "raw_sources_modified": False,
                "transformations": [
                    "trim and lowercase crop_name",
                    "trim and uppercase adm_id",
                    "trim and uppercase country_code in yield",
                    "parse YYYYMMDD date codes as DATE",
                    "write Parquet with Zstandard compression",
                ],
                "not_performed": [
                    "imputation",
                    "unit conversion",
                    "row filtering or deduplication",
                    "cross-source joins",
                    "crop-season to calendar-date mapping",
                ],
                "datasets": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest
