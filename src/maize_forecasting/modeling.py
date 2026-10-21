"""Comparacion temporal y ajuste de un regresor candidato para rendimiento."""

import json
import os
from pathlib import Path
from uuid import uuid4

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maize_forecasting.config import (
    FEATURE_REPORT_DIR,
    MODELING_REPORT_DIR,
    MODELS_DIR,
    STAR_SCHEMA_DB,
)
from maize_forecasting.feature_engineering import ROLLING_VALIDATION_FOLDS

TARGET_COLUMN = "target_yield"
YEAR_COLUMN = "season_start_year"
MODEL_INPUT_TABLE = "model_input_candidates_cutoff_50"
ARTIFACT_PATH = MODELS_DIR / "maize_yield_cutoff50.joblib"


def _build_models() -> dict[str, Pipeline | DummyRegressor]:
    imputer = lambda: SimpleImputer(strategy="median", add_indicator=True)
    return {
        "ridge_alpha10": Pipeline(
            [
                ("imputer", imputer()),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        max_features=0.8,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=250,
                        max_features=0.8,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=150,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting_huber": Pipeline(
            [
                ("imputer", imputer()),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="huber",
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _metrics(
    actual: np.ndarray, predicted: np.ndarray, years: np.ndarray
) -> dict[str, object]:
    per_season = {
        str(int(year)): float(
            mean_absolute_error(actual[years == year], predicted[years == year])
        )
        for year in np.unique(years)
    }
    return {
        "n": len(actual),
        "n_seasons": len(per_season),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "macro_season_mae": float(np.mean(list(per_season.values()))),
        "season_mae": per_season,
    }


def compare_models_rolling_origin(
    frame: pd.DataFrame, feature_names: list[str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare fixed regression pipelines using expanding-window season folds."""
    models: dict[str, Pipeline | DummyRegressor] = {
        "dummy_median": DummyRegressor(strategy="median"),
        **_build_models(),
    }
    all_results = []
    for model_name, model_template in models.items():
        oof_actual = []
        oof_predictions = []
        oof_years = []
        fold_results = []
        for train_start, train_end, validation_start, validation_end in (
            ROLLING_VALIDATION_FOLDS
        ):
            train_mask = frame[YEAR_COLUMN].between(train_start, train_end)
            validation_mask = frame[YEAR_COLUMN].between(
                validation_start, validation_end
            )
            train = frame.loc[train_mask]
            validation = frame.loc[validation_mask]
            if train.empty or validation.empty:
                raise ValueError(
                    f"Fold vacio para {model_name}: train {train_start}-{train_end}, "
                    f"validation {validation_start}-{validation_end}."
                )

            model = clone(model_template)
            model.fit(train[feature_names], train[TARGET_COLUMN])
            predictions = np.asarray(model.predict(validation[feature_names]), dtype=float)
            actual = validation[TARGET_COLUMN].to_numpy(dtype=float)
            years = validation[YEAR_COLUMN].to_numpy(dtype=int)
            fold_metrics = _metrics(actual, predictions, years)
            fold_results.append(
                {
                    "train_years": [train_start, train_end],
                    "validation_years": [validation_start, validation_end],
                    **fold_metrics,
                }
            )
            oof_actual.append(actual)
            oof_predictions.append(predictions)
            oof_years.append(years)

        aggregate = _metrics(
            np.concatenate(oof_actual),
            np.concatenate(oof_predictions),
            np.concatenate(oof_years),
        )
        all_results.append(
            {
                "model": model_name,
                "feature_names": feature_names,
                **aggregate,
                "folds": fold_results,
            }
        )

    complexity = {
        "ridge_alpha10": 1,
        "hist_gradient_boosting": 2,
        "gradient_boosting_huber": 3,
        "random_forest": 4,
        "extra_trees": 5,
        "dummy_median": 0,
    }
    all_results.sort(key=lambda result: (result["macro_season_mae"], complexity[result["model"]]))
    best_candidate = next(result for result in all_results if result["model"] != "dummy_median")
    dummy = next(result for result in all_results if result["model"] == "dummy_median")
    best = best_candidate if best_candidate["macro_season_mae"] < dummy["macro_season_mae"] else dummy
    return all_results, {
        "selected_model": best["model"],
        "candidate_best_model": best_candidate["model"],
        "dummy_baseline_macro_season_mae": dummy["macro_season_mae"],
        "best_candidate_macro_season_mae": best_candidate["macro_season_mae"],
        "candidate_beats_dummy": (
            best_candidate["macro_season_mae"] < dummy["macro_season_mae"]
        ),
        "selection_metric": "minimum rolling-origin OOF macro-season MAE",
        "tie_break": "prefer the simpler fixed estimator",
    }


def _atomic_json(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def run_modeling(*, overwrite: bool = False) -> dict[str, object]:
    """Select a regression candidate by rolling CV and fit on development years.

    The 2019-2022 holdout is reported as exploratory because it was previously
    inspected during feature engineering; it is not treated as a pristine test.
    """
    if not STAR_SCHEMA_DB.is_file():
        raise FileNotFoundError(f"No existe la base DuckDB: {STAR_SCHEMA_DB}")
    input_report_path = FEATURE_REPORT_DIR / "model_input_preparation_report.json"
    if not input_report_path.is_file():
        raise FileNotFoundError(
            "Prepara primero model_input_candidates_cutoff_50 antes de modelar."
        )
    if ARTIFACT_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {ARTIFACT_PATH}; usa overwrite=True para reemplazarlo."
        )

    input_report = json.loads(input_report_path.read_text(encoding="utf-8"))
    feature_names = input_report["selected_features_from_prior_validation"]
    input_table = input_report["table"]
    with duckdb.connect(str(STAR_SCHEMA_DB)) as connection:
        columns = {row[0] for row in connection.execute(f"DESCRIBE {input_table}").fetchall()}
        missing_features = set(feature_names) - columns
        if missing_features:
            raise ValueError(
                "Faltan features seleccionadas en la tabla de entrada: "
                + ", ".join(sorted(missing_features))
            )
        frame = connection.execute(
            f"""SELECT administrative_unit_key, season_key, season_start_year,
                       target_yield, {', '.join(f'"{name}"' for name in feature_names)}
                FROM {input_table}
                WHERE season_start_year BETWEEN 2003 AND 2022
                ORDER BY season_start_year, administrative_unit_key"""
        ).df()

    if frame.duplicated(["administrative_unit_key", "season_key"]).any():
        raise ValueError("La tabla de entrada tiene mas de una fila por region y campaña.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("La tabla de entrada contiene targets nulos.")

    comparison_results, selection = compare_models_rolling_origin(frame, feature_names)
    development = frame.loc[frame[YEAR_COLUMN].between(2003, 2018)].copy()
    exploratory_holdout = frame.loc[frame[YEAR_COLUMN].between(2019, 2022)].copy()
    candidates = {"dummy_median": DummyRegressor(strategy="median"), **_build_models()}
    selected_model = clone(candidates[selection["selected_model"]])
    selected_model.fit(development[feature_names], development[TARGET_COLUMN])
    holdout_predictions = np.asarray(
        selected_model.predict(exploratory_holdout[feature_names]), dtype=float
    )
    holdout_metrics = _metrics(
        exploratory_holdout[TARGET_COLUMN].to_numpy(dtype=float),
        holdout_predictions,
        exploratory_holdout[YEAR_COLUMN].to_numpy(dtype=int),
    )
    holdout_metrics["negative_predictions"] = int(np.sum(holdout_predictions < 0))
    holdout_baseline = DummyRegressor(strategy="median")
    holdout_baseline.fit(np.zeros((len(development), 1)), development[TARGET_COLUMN])
    baseline_holdout_predictions = holdout_baseline.predict(
        np.zeros((len(exploratory_holdout), 1))
    )
    holdout_baseline_metrics = _metrics(
        exploratory_holdout[TARGET_COLUMN].to_numpy(dtype=float),
        np.asarray(baseline_holdout_predictions, dtype=float),
        exploratory_holdout[YEAR_COLUMN].to_numpy(dtype=int),
    )
    holdout_region_scores = []
    scored_holdout = exploratory_holdout[
        ["administrative_unit_key", TARGET_COLUMN, YEAR_COLUMN]
    ].copy()
    scored_holdout["prediction"] = holdout_predictions
    for admin_key, region in scored_holdout.groupby("administrative_unit_key"):
        holdout_region_scores.append(
            {
                "administrative_unit_key": int(admin_key),
                "campaigns": len(region),
                "mae": float(mean_absolute_error(region[TARGET_COLUMN], region["prediction"])),
            }
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model = ARTIFACT_PATH.with_name(
        f".{ARTIFACT_PATH.stem}.{uuid4().hex}.joblib"
    )
    joblib.dump(
        {
            "pipeline": selected_model,
            "model_name": selection["selected_model"],
            "feature_names": feature_names,
            "target": TARGET_COLUMN,
            "target_unit": "t ha-1",
            "cutoff_pct": input_report["cutoff_pct"],
            "training_start_years": [2003, 2018],
            "feature_registry_sha256_at_evaluation": json.loads(
                (FEATURE_REPORT_DIR / "feature_engineering_report.json").read_text(
                    encoding="utf-8"
                )
            ).get("feature_registry_sha256"),
            "holdout_status": "exploratory_not_pristine",
        },
        temporary_model,
        compress=3,
    )
    os.replace(temporary_model, ARTIFACT_PATH)

    comparison_frame = pd.DataFrame(
        [
            {
                "model": result["model"],
                "split": "rolling_origin_oof",
                "n": result["n"],
                "n_seasons": result["n_seasons"],
                "macro_season_mae": result["macro_season_mae"],
                "mae": result["mae"],
                "rmse": result["rmse"],
                "r2": result["r2"],
                "season_mae_json": json.dumps(result["season_mae"]),
            }
            for result in comparison_results
        ]
        + [
            {
                "model": selection["selected_model"],
                "split": "exploratory_holdout",
                "n": holdout_metrics["n"],
                "n_seasons": holdout_metrics["n_seasons"],
                "macro_season_mae": holdout_metrics["macro_season_mae"],
                "mae": holdout_metrics["mae"],
                "rmse": holdout_metrics["rmse"],
                "r2": holdout_metrics["r2"],
                "season_mae_json": json.dumps(holdout_metrics["season_mae"]),
            },
            {
                "model": "dummy_median",
                "split": "exploratory_holdout",
                "n": holdout_baseline_metrics["n"],
                "n_seasons": holdout_baseline_metrics["n_seasons"],
                "macro_season_mae": holdout_baseline_metrics["macro_season_mae"],
                "mae": holdout_baseline_metrics["mae"],
                "rmse": holdout_baseline_metrics["rmse"],
                "r2": holdout_baseline_metrics["r2"],
                "season_mae_json": json.dumps(holdout_baseline_metrics["season_mae"]),
            }
        ]
    )
    with duckdb.connect(str(STAR_SCHEMA_DB)) as connection:
        connection.register("model_comparison_frame", comparison_frame)
        connection.execute(
            """CREATE OR REPLACE TABLE model_comparison_results AS
               SELECT * FROM model_comparison_frame"""
        )
        connection.unregister("model_comparison_frame")

    report = {
        "status": "MODEL_FITTED_EXPLORATORY",
        "input_table": input_table,
        "target": TARGET_COLUMN,
        "target_unit": "t ha-1",
        "cutoff_pct": input_report["cutoff_pct"],
        "feature_names": feature_names,
        "development_period": [2003, 2018],
        "rolling_origin_validation_folds": [
            {
                "train": [train_start, train_end],
                "validation": [validation_start, validation_end],
            }
            for train_start, train_end, validation_start, validation_end
            in ROLLING_VALIDATION_FOLDS
        ],
        "selection": selection,
        "rolling_origin_model_comparison": comparison_results,
        "exploratory_holdout_metrics": holdout_metrics,
        "exploratory_holdout_dummy_baseline": holdout_baseline_metrics,
        "exploratory_holdout_region_metrics": {
            "regions": len(holdout_region_scores),
            "median_mae": float(np.median([score["mae"] for score in holdout_region_scores])),
            "p90_mae": float(np.quantile([score["mae"] for score in holdout_region_scores], 0.9)),
            "per_region": holdout_region_scores,
            "status": "not_pristine; this period was previously inspected during feature engineering",
        },
        "model_artifact": str(ARTIFACT_PATH),
        "model_comparison_table": "model_comparison_results",
        "hyperparameter_tuning_performed": False,
        "spatial_holdout_performed": False,
        "final_deployment_model": False,
    }
    MODELING_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELING_REPORT_DIR / "model_comparison_report.json"
    _atomic_json(report_path, report)
    return {"report": report, "model": selected_model, "artifact": ARTIFACT_PATH}
