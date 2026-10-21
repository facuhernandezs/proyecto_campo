"""Carga y perfilado inicial de los CSV originales de maiz."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from maize_forecasting.config import INPUT_FILES, MAIZE_DATA_DIR


def _dataset_path(dataset_name: str) -> Path:
    """Devuelve la ruta de un dataset configurado o informa nombres validos."""
    try:
        filename = INPUT_FILES[dataset_name]
    except KeyError as exc:
        available = ", ".join(INPUT_FILES)
        raise ValueError(
            f"Dataset desconocido: {dataset_name!r}. Disponibles: {available}."
        ) from exc

    path = MAIZE_DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el dataset {dataset_name!r}: {path}")
    return path


def load_dataset(
    dataset_name: str,
    *,
    nrows: int | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Carga un CSV individual, opcionalmente limitado a filas o columnas."""
    path = _dataset_path(dataset_name)
    return pd.read_csv(path, nrows=nrows, usecols=columns)


def profile_dataset(dataset_name: str, *, chunksize: int = 100_000) -> dict[str, Any]:
    """Perfila un CSV completo por bloques sin retenerlo entero en memoria."""
    if chunksize <= 0:
        raise ValueError("chunksize debe ser mayor que cero.")

    path = _dataset_path(dataset_name)
    row_count = 0
    missing_counts: dict[str, int] = {}
    dtypes: dict[str, str] = {}
    numeric_ranges: dict[str, dict[str, int | float | None]] = {}
    identifier_values: dict[str, set[Any]] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize):
        row_count += len(chunk)
        if not dtypes:
            dtypes = chunk.dtypes.astype(str).to_dict()
            missing_counts = {column: 0 for column in chunk.columns}

        for column in ("crop_name", "country_code", "adm_id"):
            if column in chunk:
                values = chunk[column].dropna().unique().tolist()
                identifier_values.setdefault(column, set()).update(values)

        chunk_missing = chunk.isna().sum()
        for column, count in chunk_missing.items():
            missing_counts[column] += int(count)

        for column in chunk.select_dtypes(include="number").columns:
            values = chunk[column].dropna()
            if values.empty:
                continue

            current_min = values.min().item()
            current_max = values.max().item()
            if column not in numeric_ranges:
                numeric_ranges[column] = {"min": current_min, "max": current_max}
            else:
                numeric_ranges[column]["min"] = min(
                    numeric_ranges[column]["min"], current_min
                )
                numeric_ranges[column]["max"] = max(
                    numeric_ranges[column]["max"], current_max
                )

    missing_percent = {
        column: (count / row_count * 100 if row_count else 0.0)
        for column, count in missing_counts.items()
    }

    return {
        "dataset": dataset_name,
        "path": str(path),
        "file_size_mib": path.stat().st_size / (1024**2),
        "rows": row_count,
        "columns": list(dtypes),
        "dtypes": dtypes,
        "distinct_counts": {
            column: len(values) for column, values in identifier_values.items()
        },
        "missing_values": missing_counts,
        "missing_percent": missing_percent,
        "numeric_ranges": numeric_ranges,
    }
