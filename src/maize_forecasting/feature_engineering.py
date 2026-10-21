"""Construccion y evaluacion temporal de variables candidatas para yield."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from maize_forecasting.config import (
    FEATURE_CUTOFFS,
    FEATURE_REGISTRY_PATH,
    FEATURE_REPORT_DIR,
    SOURCE_METADATA_PATH,
    STAR_SCHEMA_DB,
)
from maize_forecasting.data_quality import audit_data_quality
from maize_forecasting.star_schema import refresh_metadata_catalog

ROLLING_VALIDATION_FOLDS = (
    (2003, 2008, 2009, 2010),
    (2003, 2010, 2011, 2012),
    (2003, 2012, 2013, 2014),
    (2003, 2014, 2015, 2018),
)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _feature_groups(registry: dict[str, object]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for feature in registry["features"]:
        if feature.get("status") == "candidate":
            groups.setdefault(feature["group"], []).append(feature["name"])
    return groups


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _create_candidate_feature_table(
    connection: duckdb.DuckDBPyConnection,
    cutoff_fraction: float,
) -> dict[str, object]:
    cutoff_pct = round(cutoff_fraction * 100)
    table_name = f"candidate_features_cutoff_{cutoff_pct}"
    sql = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    WITH campaigns AS (
        SELECT yield.administrative_unit_key,
               yield.season_key,
               season.season_label,
               season.harvest_year,
               bridge.season_start_year,
               bridge.season_start_date,
               bridge.season_end_date,
               bridge.season_duration_days,
               bridge.season_start_date
                   + CAST(FLOOR(bridge.season_duration_days * {cutoff_fraction}) AS INTEGER)
                   AS cutoff_date,
               yield."yield" AS target_yield
        FROM fact_yield AS yield
        INNER JOIN dim_season AS season USING (season_key)
        INNER JOIN bridge_administrative_season AS bridge
          USING (administrative_unit_key, season_key)
        WHERE bridge.mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'
          AND bridge.season_start_year BETWEEN 2003 AND 2022
    ), weather AS (
        SELECT campaigns.administrative_unit_key,
               campaigns.season_key,
               AVG(fact.tavg) AS tavg_mean_to_cutoff,
               MAX(fact.tmax) AS tmax_max_to_cutoff,
               AVG(fact.tmax - fact.tmin) AS diurnal_range_mean_to_cutoff,
               SUM(fact.cwb) AS cwb_sum_to_cutoff,
               SUM(fact.et0) AS et0_sum_to_cutoff,
               SUM(fact.rad) AS rad_sum_to_cutoff,
               AVG(fact.tavg) FILTER (
                   WHERE calendar.calendar_date >= campaigns.cutoff_date - INTERVAL 29 DAY
               ) AS tavg_mean_last_30d,
               SUM(fact.cwb) FILTER (
                   WHERE calendar.calendar_date >= campaigns.cutoff_date - INTERVAL 29 DAY
               ) AS cwb_sum_last_30d,
               COUNT(*) AS weather_observation_count,
               MAX(calendar.calendar_date) AS latest_weather_date
        FROM campaigns
        INNER JOIN fact_meteorology AS fact
          USING (administrative_unit_key)
        INNER JOIN dim_date AS calendar USING (date_key)
        WHERE calendar.calendar_date BETWEEN campaigns.season_start_date
                                         AND campaigns.cutoff_date
        GROUP BY campaigns.administrative_unit_key, campaigns.season_key,
                 campaigns.cutoff_date
    ), soil_moisture AS (
        SELECT campaigns.administrative_unit_key,
               campaigns.season_key,
               ARG_MAX(fact.ssm, calendar.calendar_date) AS latest_ssm,
               ARG_MAX(fact.rsm, calendar.calendar_date) AS latest_rsm,
               AVG(fact.rsm) FILTER (
                   WHERE calendar.calendar_date >= campaigns.cutoff_date - INTERVAL 29 DAY
               ) AS rsm_mean_last_30d,
               DATE_DIFF('day', MAX(calendar.calendar_date), campaigns.cutoff_date)
                   AS soil_moisture_age_days,
               COUNT(*) AS soil_moisture_observation_count,
               MAX(calendar.calendar_date) AS latest_soil_moisture_date
        FROM campaigns
        INNER JOIN fact_soil_moisture AS fact
          USING (administrative_unit_key)
        INNER JOIN dim_date AS calendar USING (date_key)
        WHERE calendar.calendar_date BETWEEN campaigns.season_start_date
                                         AND campaigns.cutoff_date
        GROUP BY campaigns.administrative_unit_key, campaigns.season_key,
                 campaigns.cutoff_date
    ), ndvi AS (
        SELECT campaigns.administrative_unit_key,
               campaigns.season_key,
               ARG_MAX(ndvi_observation.ndvi, ndvi_observation.calendar_date)
                   AS latest_ndvi,
               CASE WHEN COUNT(ndvi_observation.ndvi) >= 2
                    THEN ARG_MAX(ndvi_observation.ndvi, ndvi_observation.calendar_date)
                       - ARG_MIN(ndvi_observation.ndvi, ndvi_observation.calendar_date)
               END AS ndvi_change_since_first_observation,
               DATE_DIFF('day', MAX(ndvi_observation.calendar_date), campaigns.cutoff_date)
                   AS ndvi_age_days,
               COUNT(ndvi_observation.ndvi) AS ndvi_observation_count,
               MAX(ndvi_observation.calendar_date) AS latest_ndvi_date
        FROM campaigns
        LEFT JOIN (
            SELECT fact.administrative_unit_key, fact.ndvi,
                   calendar.calendar_date
            FROM fact_ndvi AS fact
            INNER JOIN dim_date AS calendar USING (date_key)
        ) AS ndvi_observation
          ON ndvi_observation.administrative_unit_key =
             campaigns.administrative_unit_key
         AND ndvi_observation.calendar_date BETWEEN campaigns.season_start_date
                                                AND campaigns.cutoff_date
        GROUP BY campaigns.administrative_unit_key, campaigns.season_key,
                 campaigns.cutoff_date
    ), fpar AS (
        SELECT campaigns.administrative_unit_key,
               campaigns.season_key,
               ARG_MAX(observation.fpar, observation.calendar_date) AS latest_fpar,
               AVG(observation.fpar) AS fpar_mean_to_cutoff,
               DATE_DIFF('day', MAX(observation.calendar_date), campaigns.cutoff_date)
                   AS fpar_age_days,
               COUNT(observation.fpar) AS fpar_observation_count,
               MAX(observation.calendar_date) AS latest_fpar_date
        FROM campaigns
        LEFT JOIN (
            SELECT fact.administrative_unit_key, fact.fpar,
                   calendar.calendar_date
            FROM fact_fpar AS fact
            INNER JOIN dim_date AS calendar USING (date_key)
        ) AS observation
          ON observation.administrative_unit_key = campaigns.administrative_unit_key
         AND observation.calendar_date BETWEEN campaigns.season_start_date
                                           AND campaigns.cutoff_date
        GROUP BY campaigns.administrative_unit_key, campaigns.season_key,
                 campaigns.cutoff_date
    ), static_context AS (
        SELECT campaigns.administrative_unit_key,
               campaigns.season_key,
               mask.crop_area_percentage,
               soil.bulk_density,
               soil.awc AS available_water_capacity_cm_m,
               CAST(drainage.source_class_code AS VARCHAR) AS drainage_class_code,
               admin.latitude AS latitude_deg,
               admin.longitude AS longitude_deg
        FROM campaigns
        LEFT JOIN fact_crop_mask AS mask USING (administrative_unit_key)
        LEFT JOIN fact_soil AS soil USING (administrative_unit_key)
        LEFT JOIN dim_drainage_class AS drainage
          ON drainage.drainage_class_key = soil.drainage_class_key
        LEFT JOIN dim_administrative_unit AS admin USING (administrative_unit_key)
    )
    SELECT campaigns.administrative_unit_key,
           campaigns.season_key,
           campaigns.season_label,
           campaigns.season_start_year,
           campaigns.harvest_year,
           campaigns.season_start_date,
           campaigns.cutoff_date,
           campaigns.season_end_date,
           campaigns.season_duration_days,
           {cutoff_fraction}::DOUBLE AS cutoff_fraction,
           CASE WHEN {cutoff_fraction} < 1.0 / 3.0 THEN 'early'
                WHEN {cutoff_fraction} < 2.0 / 3.0 THEN 'mid'
                ELSE 'late' END AS prediction_phase,
            CASE WHEN campaigns.season_start_year BETWEEN 2003 AND 2018 THEN 'development'
                 WHEN campaigns.season_start_year BETWEEN 2019 AND 2022 THEN 'exploratory_holdout'
            END AS temporal_split,
           campaigns.target_yield,
           weather.tavg_mean_to_cutoff,
           weather.tmax_max_to_cutoff,
           weather.diurnal_range_mean_to_cutoff,
           weather.cwb_sum_to_cutoff,
           weather.et0_sum_to_cutoff,
           weather.rad_sum_to_cutoff,
           weather.tavg_mean_last_30d,
           weather.cwb_sum_last_30d,
           weather.weather_observation_count,
           weather.latest_weather_date,
           soil_moisture.latest_ssm,
           soil_moisture.latest_rsm,
           soil_moisture.rsm_mean_last_30d,
           soil_moisture.soil_moisture_age_days,
           soil_moisture.soil_moisture_observation_count,
           soil_moisture.latest_soil_moisture_date,
           ndvi.latest_ndvi,
           ndvi.ndvi_change_since_first_observation,
           ndvi.ndvi_age_days,
           ndvi.ndvi_observation_count,
           ndvi.latest_ndvi_date,
           fpar.latest_fpar,
           fpar.fpar_mean_to_cutoff,
           fpar.fpar_age_days,
           fpar.fpar_observation_count,
           fpar.latest_fpar_date,
           static_context.crop_area_percentage,
           static_context.bulk_density,
           static_context.available_water_capacity_cm_m,
           static_context.drainage_class_code,
           static_context.latitude_deg,
           static_context.longitude_deg
    FROM campaigns
    LEFT JOIN weather USING (administrative_unit_key, season_key)
    LEFT JOIN soil_moisture USING (administrative_unit_key, season_key)
    LEFT JOIN ndvi USING (administrative_unit_key, season_key)
    LEFT JOIN fpar USING (administrative_unit_key, season_key)
    LEFT JOIN static_context USING (administrative_unit_key, season_key)
    """
    connection.execute(sql)

    summary = connection.execute(
        f"""SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT (administrative_unit_key, season_key)) AS unique_campaigns,
                   COUNT(*) FILTER (WHERE target_yield IS NULL) AS null_targets,
                   MIN(season_start_year) AS first_start_year,
                   MAX(season_start_year) AS last_start_year,
                   COUNT(*) FILTER (WHERE weather_observation_count IS NULL)
                       AS no_weather_observations,
                   COUNT(*) FILTER (
                       WHERE soil_moisture_observation_count IS NULL
                          OR soil_moisture_observation_count = 0
                   )
                       AS campaigns_without_soil_moisture_values,
                   COUNT(*) FILTER (WHERE ndvi_observation_count = 0)
                       AS no_ndvi_observations,
                   COUNT(*) FILTER (WHERE fpar_observation_count = 0)
                        AS no_fpar_observations,
                   SUM(CASE WHEN latest_weather_date > cutoff_date
                         OR latest_soil_moisture_date > cutoff_date
                         OR latest_ndvi_date > cutoff_date
                         OR latest_fpar_date > cutoff_date
                         THEN 1 ELSE 0 END) AS future_feature_dates
            FROM {table_name}"""
    ).fetchone()
    columns = [item[0] for item in connection.execute(f"DESCRIBE {table_name}").fetchall()]
    return {
        "table": table_name,
        "cutoff_fraction": cutoff_fraction,
        "rows": summary[0],
        "unique_campaigns": summary[1],
        "null_targets": summary[2],
        "first_start_year": summary[3],
        "last_start_year": summary[4],
        "campaigns_without_weather_observations": summary[5],
        "campaigns_without_soil_moisture_values": summary[6],
        "campaigns_without_ndvi_observations": summary[7],
        "campaigns_without_fpar_observations": summary[8],
        "cutoff_violations": summary[9],
        "columns": columns,
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, years: np.ndarray) -> dict[str, object]:
    seasonal_mae = {
        str(int(year)): float(mean_absolute_error(y_true[years == year], y_pred[years == year]))
        for year in np.unique(years)
    }
    return {
        "n": len(y_true),
        "n_seasons": len(seasonal_mae),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "macro_season_mae": float(np.mean(list(seasonal_mae.values()))),
        "season_mae": seasonal_mae,
    }


def _one_standard_error_selection(
    results: list[dict[str, object]],
    *,
    score_key: str,
    complexity_key: Callable[[dict[str, object]], object],
) -> tuple[dict[str, object], dict[str, object], float, list[dict[str, object]]]:
    """Select the simplest/earliest option within one SE of the best score."""
    best = min(results, key=lambda result: result[score_key])
    season_scores = np.asarray(list(best["season_mae"].values()), dtype=float)
    standard_error = (
        float(np.std(season_scores, ddof=1) / np.sqrt(len(season_scores)))
        if len(season_scores) > 1
        else 0.0
    )
    threshold = float(best[score_key]) + standard_error
    eligible = [result for result in results if result[score_key] <= threshold]
    selected = min(
        eligible,
        key=lambda result: (complexity_key(result), result[score_key]),
    )
    return best, selected, threshold, eligible


def _fit_and_predict(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    features: list[str],
    *,
    dummy: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    target = "target_yield"
    active_features = [
        feature for feature in features if train[feature].notna().any()
    ]
    if dummy:
        model = DummyRegressor(strategy="median")
        model.fit(np.zeros((len(train), 1)), train[target])
        predictions = model.predict(np.zeros((len(evaluate), 1)))
    else:
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        model.fit(train[active_features], train[target])
        predictions = model.predict(evaluate[active_features])
    details = {
        "requested_features": features,
        "active_features": active_features,
        "dropped_all_null_features": sorted(set(features) - set(active_features)),
    }
    return np.asarray(predictions, dtype=float), details


def _fit_and_score(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    features: list[str],
    *,
    dummy: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    predictions, details = _fit_and_predict(train, evaluate, features, dummy=dummy)
    target = "target_yield"
    metrics = _metrics(
        evaluate[target].to_numpy(dtype=float),
        predictions,
        evaluate["season_start_year"].to_numpy(dtype=int),
    )
    return metrics, details


def _rolling_origin_score(
    frame: pd.DataFrame,
    features: list[str],
    *,
    dummy: bool = False,
) -> dict[str, object]:
    observed_targets = []
    out_of_fold_predictions = []
    observed_years = []
    fold_results = []
    feature_details = None

    for train_start, train_end, validation_start, validation_end in ROLLING_VALIDATION_FOLDS:
        train = frame.loc[
            frame["season_start_year"].between(train_start, train_end)
        ]
        validation = frame.loc[
            frame["season_start_year"].between(validation_start, validation_end)
        ]
        if train.empty or validation.empty:
            raise ValueError(
                "Un fold rolling-origin no tiene train o validation: "
                f"{train_start}-{train_end} / {validation_start}-{validation_end}."
            )

        predictions, feature_details = _fit_and_predict(
            train, validation, features, dummy=dummy
        )
        years = validation["season_start_year"].to_numpy(dtype=int)
        actual = validation["target_yield"].to_numpy(dtype=float)
        fold_metrics = _metrics(actual, predictions, years)
        fold_results.append(
            {
                "train_years": [train_start, train_end],
                "validation_years": [validation_start, validation_end],
                **fold_metrics,
            }
        )
        observed_targets.append(actual)
        out_of_fold_predictions.append(predictions)
        observed_years.append(years)

    aggregate_metrics = _metrics(
        np.concatenate(observed_targets),
        np.concatenate(out_of_fold_predictions),
        np.concatenate(observed_years),
    )
    aggregate_metrics["folds"] = fold_results
    aggregate_metrics["feature_details"] = feature_details
    return aggregate_metrics


def _evaluate_candidate_tables(
    connection: duckdb.DuckDBPyConnection,
    registry: dict[str, object],
) -> dict[str, object]:
    feature_groups = _feature_groups(registry)
    core_features = feature_groups["climate_core"]
    cutoffs = [round(fraction * 100) for fraction in FEATURE_CUTOFFS]
    cohort_counts = {}
    feature_coverage_by_cutoff = {}
    climate_core_results = []
    feature_bundle_results = []
    dummy_results = []
    feature_bundles = {
        "climate_core": feature_groups["climate_core"],
        "climate_plus_recent_weather": feature_groups["climate_core"]
        + feature_groups["recent_weather"],
        "climate_plus_soil_water": feature_groups["climate_core"]
        + feature_groups["soil_water"],
        "climate_plus_vegetation": feature_groups["climate_core"]
        + feature_groups["vegetation"],
        "climate_plus_static_context": feature_groups["climate_core"]
        + feature_groups["static_context"],
    }
    for group_name in ("soil_static", "spatial_context"):
        if feature_groups.get(group_name):
            feature_bundles[f"climate_plus_{group_name}"] = (
                feature_groups["climate_core"] + feature_groups[group_name]
            )
    feature_bundles["all_candidate_groups"] = [
        feature["name"]
        for feature in registry["features"]
        if feature.get("status") == "candidate"
    ]

    for cutoff_pct in cutoffs:
        table_name = f"candidate_features_cutoff_{cutoff_pct}"
        frame = connection.execute(f"SELECT * FROM {table_name}").df()
        cohort_counts[str(cutoff_pct)] = len(frame)
        feature_coverage_by_cutoff[str(cutoff_pct)] = {
            feature["name"]: {
                "non_null_rows": int(frame[feature["name"]].notna().sum()),
                "rows": len(frame),
                "non_null_fraction": float(frame[feature["name"]].notna().mean()),
            }
            for feature in registry["features"]
        }
        if frame[["administrative_unit_key", "season_key"]].duplicated().any():
            raise ValueError(f"{table_name} contiene campañas duplicadas.")
        climate_metrics = _rolling_origin_score(frame, core_features)
        climate_core_results.append(
            {
                "cutoff_pct": cutoff_pct,
                "feature_bundle": "climate_core",
                "split": "rolling_origin_oof",
                **climate_metrics,
            }
        )
        for bundle_name, features in feature_bundles.items():
            metrics = (
                climate_metrics
                if bundle_name == "climate_core"
                else _rolling_origin_score(frame, features)
            )
            feature_bundle_results.append(
                {
                    "cutoff_pct": cutoff_pct,
                    "feature_bundle": bundle_name,
                    "split": "rolling_origin_oof",
                    **metrics,
                }
            )
        dummy_metrics = _rolling_origin_score(frame, [], dummy=True)
        dummy_results.append(
            {
                "cutoff_pct": cutoff_pct,
                "feature_bundle": "dummy_median",
                "split": "rolling_origin_oof",
                **dummy_metrics,
            }
        )

    cutoff_summary = []
    for cutoff_pct in cutoffs:
        results = [
            result
            for result in feature_bundle_results
            if result["cutoff_pct"] == cutoff_pct
        ]
        climate_result = next(
            result for result in results if result["feature_bundle"] == "climate_core"
        )
        full_result = next(
            result
            for result in results
            if result["feature_bundle"] == "all_candidate_groups"
        )
        best_bundle = min(
            results,
            key=lambda result: (
                result["macro_season_mae"],
                len(result["feature_details"]["active_features"]),
            ),
        )
        cutoff_summary.append(
            {
                "cutoff_pct": cutoff_pct,
                "climate_core_macro_season_mae": climate_result[
                    "macro_season_mae"
                ],
                "all_candidates_macro_season_mae": full_result[
                    "macro_season_mae"
                ],
                "feature_bundle": best_bundle["feature_bundle"],
                "best_bundle": best_bundle["feature_bundle"],
                "best_bundle_macro_season_mae": best_bundle["macro_season_mae"],
                "best_bundle_standard_error": float(
                    np.std(list(best_bundle["season_mae"].values()), ddof=1)
                    / np.sqrt(best_bundle["n_seasons"])
                ),
                "split": "rolling_origin_oof",
                "n": best_bundle["n"],
                "n_seasons": best_bundle["n_seasons"],
                "mae": best_bundle["mae"],
                "rmse": best_bundle["rmse"],
                "r2": best_bundle["r2"],
                "macro_season_mae": best_bundle["macro_season_mae"],
                "season_mae": best_bundle["season_mae"],
                "campaigns": best_bundle["n"],
            }
        )

    raw_best_cutoff, selected_cutoff_summary, cutoff_one_se_threshold, eligible_cutoffs = (
        _one_standard_error_selection(
            cutoff_summary,
            score_key="best_bundle_macro_season_mae",
            complexity_key=lambda result: result["cutoff_pct"],
        )
    )
    selected_cutoff = int(selected_cutoff_summary["cutoff_pct"])
    cutoff_selection = {
        "metric": "macro-season MAE on rolling-origin OOF validation",
        "raw_best_cutoff_pct": raw_best_cutoff["cutoff_pct"],
        "raw_best_macro_season_mae": raw_best_cutoff[
            "best_bundle_macro_season_mae"
        ],
        "one_standard_error_threshold": cutoff_one_se_threshold,
        "eligible_cutoffs": [result["cutoff_pct"] for result in eligible_cutoffs],
        "selected_cutoff_pct": selected_cutoff,
        "rule": "choose the earliest forecast cutoff within one standard error of the raw best",
    }
    bundles_at_cutoff = [
        result
        for result in feature_bundle_results
        if result["cutoff_pct"] == selected_cutoff
    ]
    raw_best_bundle, selected_bundle_result, bundle_one_se_threshold, eligible_bundles = (
        _one_standard_error_selection(
            bundles_at_cutoff,
            score_key="macro_season_mae",
            complexity_key=lambda result: len(
                result["feature_details"]["active_features"]
            ),
        )
    )
    selected_bundle = selected_bundle_result["feature_bundle"]
    selected_features = selected_bundle_result["feature_details"]["active_features"]
    bundle_selection = {
        "selected_cutoff_pct": selected_cutoff,
        "raw_best_bundle": raw_best_bundle["feature_bundle"],
        "raw_best_macro_season_mae": raw_best_bundle["macro_season_mae"],
        "one_standard_error_threshold": bundle_one_se_threshold,
        "eligible_bundles": [result["feature_bundle"] for result in eligible_bundles],
        "selected_bundle": selected_bundle,
        "rule": "choose the fewest-feature bundle within one standard error of the raw best",
    }

    selected_table = f"candidate_features_cutoff_{selected_cutoff}"
    selected_frame = connection.execute(f"SELECT * FROM {selected_table}").df()
    development = selected_frame.loc[
        selected_frame["season_start_year"].between(2003, 2018)
    ].copy()
    validation_years = selected_frame.loc[
        selected_frame["season_start_year"].between(2009, 2018)
    ].copy()
    exploratory_holdout = selected_frame.loc[
        selected_frame["season_start_year"].between(2019, 2022)
    ].copy()
    target_profile = {
        "rows": len(selected_frame),
        "zero_yield_rows": int((selected_frame["target_yield"] == 0).sum()),
        "min_yield": float(selected_frame["target_yield"].min()),
        "median_yield": float(selected_frame["target_yield"].median()),
        "mean_yield": float(selected_frame["target_yield"].mean()),
        "max_yield": float(selected_frame["target_yield"].max()),
        "unit": "t ha-1",
    }

    feature_ablation = []
    for feature in selected_features:
        reduced_features = [item for item in selected_features if item != feature]
        reduced_metrics = _rolling_origin_score(selected_frame, reduced_features)
        feature_ablation.append(
            {
                "feature": feature,
                "macro_season_mae_with_feature": selected_bundle_result[
                    "macro_season_mae"
                ],
                "macro_season_mae_without_feature": reduced_metrics[
                    "macro_season_mae"
                ],
                "delta_mae_when_removed": (
                    reduced_metrics["macro_season_mae"]
                    - selected_bundle_result["macro_season_mae"]
                ),
                "interpretation": "positive delta means validation error increased when removed",
            }
        )

    validation_features = validation_years[selected_features]
    correlation_matrix = validation_features.corr(method="pearson", min_periods=30)
    highly_correlated_pairs = []
    for left_index, left_feature in enumerate(selected_features):
        for right_feature in selected_features[left_index + 1 :]:
            correlation = correlation_matrix.loc[left_feature, right_feature]
            if pd.notna(correlation) and abs(correlation) >= 0.9:
                highly_correlated_pairs.append(
                    {
                        "feature_a": left_feature,
                        "feature_b": right_feature,
                        "pearson_r": float(correlation),
                        "action": "review redundancy; correlation alone does not trigger removal",
                    }
                )

    pruned_features = [
        result["feature"]
        for result in feature_ablation
        if result["delta_mae_when_removed"] >= 0
    ]
    pruned_validation_metrics = _rolling_origin_score(selected_frame, pruned_features)
    selected_feature_bundle = f"{selected_bundle}_ablation_pruned"

    test_metrics, test_details = _fit_and_score(
        development, exploratory_holdout, pruned_features
    )
    test_baseline_metrics, _ = _fit_and_score(
        development, exploratory_holdout, [], dummy=True
    )
    per_region_test = []
    test_model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    test_model.fit(development[pruned_features], development["target_yield"])
    test_predictions = test_model.predict(exploratory_holdout[pruned_features])
    negative_test_predictions = int(np.sum(test_predictions < 0))
    test_scored = exploratory_holdout[
        ["administrative_unit_key", "target_yield"]
    ].copy()
    test_scored["prediction"] = test_predictions
    for administrative_unit_key, region_frame in test_scored.groupby(
        "administrative_unit_key"
    ):
        per_region_test.append(
            {
                "administrative_unit_key": int(administrative_unit_key),
                "campaigns": len(region_frame),
                "mae": float(
                    mean_absolute_error(
                        region_frame["target_yield"], region_frame["prediction"]
                    )
                ),
            }
        )

    unmapped_target_rows = connection.execute(
        """SELECT CASE WHEN season_start_year BETWEEN 2003 AND 2018 THEN 'development'
                        WHEN season_start_year BETWEEN 2019 AND 2022 THEN 'exploratory_holdout'
                   END AS split,
                   COUNT(*) AS rows
            FROM bridge_administrative_season
            WHERE mapping_status = 'MISSING_CROP_CALENDAR'
              AND season_start_year BETWEEN 2003 AND 2022
            GROUP BY split"""
    ).fetchall()
    unmapped_by_split = {split: rows for split, rows in unmapped_target_rows}

    return {
        "target_cohort": {
            "start_years": [2003, 2022],
            "mapped_campaigns_by_cutoff": cohort_counts,
            "yield_campaigns_without_calendar_by_split": unmapped_by_split,
            "feature_coverage_by_cutoff": feature_coverage_by_cutoff,
            "target_distribution": target_profile,
            "temporal_split_counts": {
                "development_train_rows": len(development),
                "rolling_validation_oof_rows": pruned_validation_metrics["n"],
                "exploratory_holdout_rows": len(exploratory_holdout),
            },
            "split_definition": {
                "development": [2003, 2018],
                "rolling_origin_folds": [
                    {
                        "train": [train_start, train_end],
                        "validation": [validation_start, validation_end],
                    }
                    for train_start, train_end, validation_start, validation_end
                    in ROLLING_VALIDATION_FOLDS
                ],
                "exploratory_holdout": [2019, 2022],
                "grouping": "all administrative units from a crop-season label stay in the same temporal fold",
            },
        },
        "model": {
            "baseline": "DummyRegressor(strategy='median')",
            "feature_model": "HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=42)",
            "missing_values": "HistGradientBoostingRegressor native NaN handling; no imputation fitted outside training data",
            "primary_metric": "macro-season MAE",
            "secondary_metrics": ["pooled MAE", "RMSE", "R2"],
        },
        "cutoff_validation_results": cutoff_summary,
        "climate_core_cutoff_validation_results": climate_core_results,
        "dummy_validation_results": dummy_results,
        "selected_cutoff_pct": selected_cutoff,
        "cutoff_selection": cutoff_selection,
        "feature_bundle_validation_results": feature_bundle_results,
        "best_group_bundle": selected_bundle,
        "feature_bundle_selection": bundle_selection,
        "selected_feature_bundle": selected_feature_bundle,
        "selected_features": pruned_features,
        "selected_bundle_rolling_validation_metrics": pruned_validation_metrics,
        "feature_ablation_validation": feature_ablation,
        "feature_ablation_pruning_rule": "Keep a feature only when removing it does not reduce validation macro-season MAE; test metrics are not used for pruning.",
        "highly_correlated_feature_pairs_validation": highly_correlated_pairs,
        "correlation_flag_threshold": 0.9,
        "exploratory_holdout_metrics_selected_candidate": test_metrics,
        "exploratory_holdout_model_details": test_details,
        "exploratory_holdout_metrics_dummy_baseline": test_baseline_metrics,
        "exploratory_holdout_negative_predictions": negative_test_predictions,
        "holdout_region_stability": {
            "regions_evaluated": len(per_region_test),
            "median_region_mae": float(
                np.median([result["mae"] for result in per_region_test])
            ),
            "p90_region_mae": float(
                np.quantile([result["mae"] for result in per_region_test], 0.9)
            ),
            "per_region": per_region_test,
            "scope": "regions represented in development; unseen-region transfer is not tested",
        },
        "selection_policy": {
            "cutoff": "one-standard-error rule on the best bundle for each rolling-origin cutoff; choose earliest eligible checkpoint",
            "feature_bundle": "one-standard-error rule over pre-registered bundles at selected cutoff; choose fewest features",
            "feature_ablation": "prune candidates only using leave-one-feature-out rolling-origin validation",
            "exploratory_holdout": "2019-2022 is not used for selection; it was previously accessed for the initial all-candidate bundle, so it is not a pristine final test",
        },
    }


def build_candidate_feature_tables(
    *, database_path: Path = STAR_SCHEMA_DB, overwrite: bool = False
) -> list[dict[str, object]]:
    """Materialize one same-grain candidate feature table per cutoff fraction."""
    if not database_path.is_file():
        raise FileNotFoundError(f"No existe la base DuckDB: {database_path}")
    with duckdb.connect(str(database_path)) as connection:
        existing_tables = {
            row[0] for row in connection.execute("SHOW TABLES").fetchall()
        }
        candidate_tables = {
            f"candidate_features_cutoff_{round(fraction * 100)}"
            for fraction in FEATURE_CUTOFFS
        }
        if existing_tables & candidate_tables and not overwrite:
            raise FileExistsError(
                "Ya existen tablas candidatas en DuckDB; usa overwrite=True para regenerarlas."
            )
        return [
            _create_candidate_feature_table(connection, fraction)
            for fraction in FEATURE_CUTOFFS
        ]


def prepare_model_input_candidates(
    *,
    cutoff_pct: int | None = None,
    selected_features: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Prepare a model-input candidate table without fitting or saving a model."""
    if not STAR_SCHEMA_DB.is_file():
        raise FileNotFoundError(f"No existe la base DuckDB: {STAR_SCHEMA_DB}")
    if not FEATURE_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Falta el feature registry: {FEATURE_REGISTRY_PATH}")

    registry = json.loads(FEATURE_REGISTRY_PATH.read_text(encoding="utf-8"))
    feature_report_path = FEATURE_REPORT_DIR / "feature_engineering_report.json"
    previous_report = (
        json.loads(feature_report_path.read_text(encoding="utf-8"))
        if feature_report_path.is_file()
        else None
    )
    feature_decisions = {
        item["name"]: item.get("evaluation_decision")
        for item in (previous_report or {}).get("feature_decisions", [])
    }
    if cutoff_pct is None:
        if previous_report is None:
            raise FileNotFoundError(
                "Indica cutoff_pct/selected_features o genera primero el reporte de Feature Engineering."
            )
        evaluation = previous_report["evaluation"]
        cutoff_pct = int(evaluation["selected_cutoff_pct"])
        selected_features = selected_features or evaluation["selected_features"]
    feature_decisions = {
        item["name"]: item.get("evaluation_decision")
        for item in (previous_report or {}).get("feature_decisions", [])
    }
    if cutoff_pct not in {round(value * 100) for value in FEATURE_CUTOFFS}:
        raise ValueError("cutoff_pct debe ser uno de los checkpoints 25, 50 o 75.")
    if selected_features is None:
        raise ValueError("selected_features no puede quedar vacio.")

    candidate_table = f"candidate_features_cutoff_{cutoff_pct}"
    model_table = f"model_input_candidates_cutoff_{cutoff_pct}"
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in selected_features):
        raise ValueError("El feature registry contiene nombres de columna no validos.")
    quality_report = audit_data_quality()
    metadata_sync = refresh_metadata_catalog()
    with duckdb.connect(str(STAR_SCHEMA_DB)) as connection:
        table_names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if candidate_table not in table_names:
            raise FileNotFoundError(
                f"Falta {candidate_table}; ejecuta build_candidate_feature_tables primero."
            )
        if model_table in table_names and not overwrite:
            raise FileExistsError(
                f"Ya existe {model_table}; usa overwrite=True para regenerarla."
            )

        available_columns = {
            row[0] for row in connection.execute(f"DESCRIBE {candidate_table}").fetchall()
        }
        invalid_features = set(selected_features) - available_columns
        if invalid_features:
            raise ValueError(
                "Features seleccionadas ausentes de la tabla candidata: "
                + ", ".join(sorted(invalid_features))
            )

        feature_projection = ",\n                   ".join(
            f'bundle."{name}"' for name in selected_features
        )
        extra_candidate_specs = {
            "available_water_capacity_cm_m": "soil.awc",
            "latitude_deg": "admin.latitude",
            "longitude_deg": "admin.longitude",
            "drainage_class_code": "CAST(drainage.source_class_code AS VARCHAR)",
        }
        extra_candidate_columns = [
            column
            for column in extra_candidate_specs
            if column not in selected_features
        ]
        extra_projection = ",\n                   ".join(
            f"{extra_candidate_specs[column]} AS {column}"
            for column in extra_candidate_columns
        )
        model_input_projection = ",\n                   ".join(
            projection
            for projection in (feature_projection, extra_projection)
            if projection
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {model_table} AS
            SELECT crop.crop_name,
                   admin.country_code,
                   admin.adm_id,
                   bundle.administrative_unit_key,
                   bundle.season_key,
                   bundle.season_label,
                   bundle.season_start_year,
                   bundle.harvest_year,
                   bundle.season_start_date,
                   bundle.cutoff_date AS prediction_date,
                   bundle.season_end_date,
                   bundle.cutoff_fraction,
                   bundle.prediction_phase,
                    bundle.temporal_split,
                    bundle.target_yield,
                    {model_input_projection},
                    admin.location_attributes_available
            FROM {candidate_table} AS bundle
            INNER JOIN dim_crop AS crop
              ON crop.crop_name = {_sql_string(registry['project_scope']['crop'])}
            INNER JOIN dim_administrative_unit AS admin
              ON admin.administrative_unit_key = bundle.administrative_unit_key
            LEFT JOIN fact_soil AS soil
              ON soil.administrative_unit_key = bundle.administrative_unit_key
             AND soil.crop_key = crop.crop_key
            LEFT JOIN dim_drainage_class AS drainage
              ON drainage.drainage_class_key = soil.drainage_class_key
            """
        )

        row_count, unique_grain, null_targets, zero_yields = connection.execute(
            f"""SELECT COUNT(*),
                       COUNT(DISTINCT (administrative_unit_key, season_key)),
                       COUNT(*) FILTER (WHERE target_yield IS NULL),
                       COUNT(*) FILTER (WHERE target_yield = 0)
                FROM {model_table}"""
        ).fetchone()
        if row_count != unique_grain or null_targets:
            raise ValueError(
                f"La tabla candidata no preserva el grano/target: "
                f"rows={row_count}, unique={unique_grain}, null_targets={null_targets}."
            )

        quality = connection.execute(
            f"""SELECT COUNT(*) FILTER (WHERE available_water_capacity_cm_m IS NULL),
                       COUNT(*) FILTER (WHERE drainage_class_code IS NULL),
                       COUNT(*) FILTER (WHERE latitude_deg IS NULL OR longitude_deg IS NULL),
                        MIN(target_yield), MAX(target_yield)
                FROM {model_table}"""
        ).fetchone()
        unmapped_campaigns_sensor_period = connection.execute(
            """SELECT COUNT(*) FROM bridge_administrative_season
               WHERE mapping_status = 'MISSING_CROP_CALENDAR'
                 AND season_start_year BETWEEN 2003 AND 2022"""
        ).fetchone()[0]
        column_metadata = [
            (
                "crop_name",
                "scope_key",
                "CY-Bench bundle",
                "crop",
                None,
                "documented",
                "Maize only; not a predictor.",
            ),
            (
                "country_code",
                "scope_key",
                "CY-Bench bundle",
                "country_code",
                None,
                "documented",
                "AR only; not a predictor.",
            ),
            (
                "adm_id",
                "administrative_identifier",
                "CY-Bench bundle",
                "adm_id",
                None,
                "conformed_by_CY-Bench_v1.10",
                "Join through dim_administrative_unit; not a numeric predictor.",
            ),
            (
                "administrative_unit_key",
                "surrogate_key",
                "dim_administrative_unit",
                "administrative_unit_key",
                None,
                "validated",
                "Warehouse key; never a predictor.",
            ),
            (
                "season_key",
                "surrogate_key",
                "dim_season",
                "planting_year",
                None,
                "validated",
                "Crop-season key; never a predictor.",
            ),
            (
                "season_label",
                "campaign_identifier",
                "yield",
                "planting_year",
                None,
                "source_label",
                "Source campaign label; use for split/group metadata, not as numeric feature.",
            ),
            (
                "season_start_year",
                "split_metadata",
                "derived from season_label",
                "planting_year",
                "year",
                "validated_label_anchor",
                "Used for chronological folds; excluded from predictors.",
            ),
            (
                "harvest_year",
                "campaign_metadata",
                "yield",
                "harvest_year",
                "year",
                "documented",
                "Campaign metadata; excluded from predictors to avoid trend memorization.",
            ),
            (
                "season_start_date",
                "calendar_metadata",
                "crop_calendar",
                "sos",
                "date",
                "static_calendar_proxy",
                "Projected from static DOY; not an observed annual planting date.",
            ),
            (
                "target_yield",
                "target",
                "yield",
                "yield",
                "t ha-1",
                "documented",
                "label; never a predictor",
            ),
            (
                "prediction_date",
                "forecast_cutoff",
                "crop_calendar",
                "sos/eos",
                "date",
                "static_calendar_proxy",
                "midpoint of the projected local growing-season window",
            ),
            (
                "season_end_date",
                "calendar_metadata",
                "crop_calendar",
                "eos",
                "date",
                "static_calendar_proxy",
                "Growing-season EOS; not the harvest date.",
            ),
            (
                "cutoff_fraction",
                "forecast_metadata",
                "derived",
                "season_duration_days",
                "fraction",
                "fixed_0.5",
                "Checkpoint selected by one-standard-error rule.",
            ),
            (
                "prediction_phase",
                "forecast_metadata",
                "derived",
                "cutoff_fraction",
                None,
                "fixed_midseason",
                "Forecast is made at the middle-season checkpoint.",
            ),
            (
                "temporal_split",
                "split_metadata",
                "derived",
                "season_start_year",
                None,
                "chronological",
                "Development/exploratory-holdout label; excluded from predictors.",
            ),
            (
                "location_attributes_available",
                "data_quality_flag",
                "location",
                "adm_id",
                "boolean",
                "documented",
                "Indicates centroid/region-area coverage, not a model predictor.",
            ),
        ]
        source_metadata_by_name = {
            feature["name"]: feature for feature in registry["features"]
        }
        for feature_name in selected_features:
            feature = source_metadata_by_name[feature_name]
            source_variable = feature["source_variables"][0].split(".")
            column_metadata.append(
                (
                    feature_name,
                    "selected_feature_candidate",
                    source_variable[0],
                    ", ".join(feature["source_variables"]),
                    feature["unit"],
                    "selected_in_prior_validation_not_finalized",
                    feature["hypothesis"],
                )
            )

        feature_metadata_by_name = {
            feature["name"]: feature for feature in registry["features"]
        }
        for candidate_name in extra_candidate_columns:
            feature = feature_metadata_by_name[candidate_name]
            decision = feature_decisions.get(candidate_name)
            if decision == "pruned_by_validation_ablation":
                role = "evaluated_candidate_not_selected"
                status = decision
            elif decision == "selected_candidate_not_finalized":
                role = "selected_feature_candidate"
                status = decision
            else:
                role = "candidate_not_evaluated"
                status = feature.get("status", "not_evaluated")
            column_metadata.append(
                (
                    candidate_name,
                    role,
                    feature["source_variables"][0].split(".")[0],
                    ", ".join(feature["source_variables"]),
                    feature.get("unit"),
                    status,
                    feature["hypothesis"],
                )
            )

        column_metadata.extend(
            [
                (
                    "administrative_unit_key",
                    "surrogate_key",
                    "dim_administrative_unit",
                    "administrative_unit_key",
                    None,
                    "validated",
                    "Warehouse key; not a predictor.",
                ),
                (
                    "season_key",
                    "surrogate_key",
                    "dim_season",
                    "planting_year",
                    None,
                    "validated",
                    "Campaign key; not a predictor.",
                ),
                (
                    "temporal_split",
                    "split_metadata",
                    "derived",
                    "season_start_year",
                    None,
                    "chronological",
                    "Development/holdout marker; not a predictor.",
                ),
            ]
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE model_input_column_metadata (
                column_name VARCHAR,
                column_role VARCHAR,
                source_dataset VARCHAR,
                source_column VARCHAR,
                unit VARCHAR,
                metadata_status VARCHAR,
                note VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO model_input_column_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            column_metadata,
        )

    feature_report = previous_report or {}
    if feature_report:
        current_registry_sha = hashlib.sha256(
            FEATURE_REGISTRY_PATH.read_bytes()
        ).hexdigest()
        current_data_dictionary_sha = hashlib.sha256(
            SOURCE_METADATA_PATH.read_bytes()
        ).hexdigest()
        awc_evaluation_status = feature_decisions.get(
            "available_water_capacity_cm_m"
        )
        awc_evaluation_note = (
            "AWC in cm m-1 was included in the latest validation comparison."
            if awc_evaluation_status
            in ("selected_candidate_not_finalized", "pruned_by_validation_ablation")
            else "AWC in cm m-1 is documented but was not included in the saved feature validation."
        )
        feature_report["current_feature_registry_sha256"] = current_registry_sha
        feature_report["current_data_dictionary_sha256"] = current_data_dictionary_sha
        feature_report["post_evaluation_metadata_updates"] = [
            awc_evaluation_note,
            "Raw latitude/longitude and drainage code are included as unscored candidates because CRS/codebook validation is pending.",
        ]
        temporary_feature_report = feature_report_path.with_name(
            f".{feature_report_path.stem}.{uuid4().hex}.tmp"
        )
        temporary_feature_report.write_text(
            json.dumps(feature_report, indent=2), encoding="utf-8"
        )
        temporary_feature_report.replace(feature_report_path)

    additional_not_evaluated = [
        name
        for name in extra_candidate_columns
        if feature_decisions.get(name) in (None, "pending_metadata_or_encoding")
    ]
    evaluated_not_selected = [
        name
        for name in extra_candidate_columns
        if feature_decisions.get(name) == "pruned_by_validation_ablation"
    ]
    report = {
        "status": "PREPARED_CANDIDATE_NOT_TRAINED",
        "dataset_bundle": registry["project_scope"],
        "database": str(STAR_SCHEMA_DB),
        "table": model_table,
        "row_grain": "one administrative unit x crop season x selected 50% forecast checkpoint",
        "cutoff_pct": cutoff_pct,
        "rows": row_count,
        "unique_admin_season_keys": unique_grain,
        "target_column": "target_yield",
        "target_unit": "t ha-1",
        "zero_yield_rows_preserved": zero_yields,
        "unmapped_campaigns_excluded": unmapped_campaigns_sensor_period,
        "selected_features_from_prior_validation": selected_features,
        "additional_candidate_columns_not_evaluated": additional_not_evaluated,
        "evaluated_candidate_columns_not_selected": evaluated_not_selected,
        "additional_candidate_null_rows": {
            "available_water_capacity_cm_m": quality[0],
            "drainage_class_code": quality[1],
            "centroid_coordinates": quality[2],
        },
        "production_area_features_excluded": ["production", "harvest_area", "planted_area"],
        "precipitation_and_vpd_excluded": "prec remains provisional; VPD unit remains unresolved",
        "scenario_code": registry["project_scope"].get("management_scenario_code", "mm0; assumed"),
        "date_availability_assumption": "observation_date <= prediction_date; source publication timestamps are absent",
        "feature_evaluation_performed_now": False,
        "model_training_performed": False,
        "metadata_sync": metadata_sync,
        "quality_report_status": quality_report["status"],
        "target_yield_profile": {
            "min": quality[3],
            "max": quality[4],
            "zero_rows": zero_yields,
        },
        "column_metadata_table": "model_input_column_metadata",
    }

    FEATURE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preparation_report_path = FEATURE_REPORT_DIR / "model_input_preparation_report.json"
    temporary_path = preparation_report_path.with_name(
        f".{preparation_report_path.stem}.{uuid4().hex}.tmp"
    )
    temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_path.replace(preparation_report_path)
    return report


def run_feature_engineering(*, overwrite: bool = False) -> dict[str, object]:
    """Build cutoff candidate tables, evaluate them, and document decisions."""
    if not STAR_SCHEMA_DB.is_file():
        raise FileNotFoundError(f"No existe la base DuckDB: {STAR_SCHEMA_DB}")
    if not FEATURE_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Falta el feature registry: {FEATURE_REGISTRY_PATH}")
    registry = json.loads(FEATURE_REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_sha256 = hashlib.sha256(FEATURE_REGISTRY_PATH.read_bytes()).hexdigest()

    candidate_summaries = build_candidate_feature_tables(overwrite=overwrite)
    with duckdb.connect(str(STAR_SCHEMA_DB)) as connection:
        evaluation = _evaluate_candidate_tables(connection, registry)
        evaluation_rows = []
        for key, results in (
            ("cutoff_validation", evaluation["cutoff_validation_results"]),
            ("dummy_validation", evaluation["dummy_validation_results"]),
            ("feature_bundle_validation", evaluation["feature_bundle_validation_results"]),
        ):
            for result in results:
                evaluation_rows.append(
                    {
                        "evaluation_type": key,
                        "cutoff_pct": result["cutoff_pct"],
                        "feature_bundle": result["feature_bundle"],
                        "split": result["split"],
                        "n": result["n"],
                        "n_seasons": result["n_seasons"],
                        "mae": result["mae"],
                        "rmse": result["rmse"],
                        "r2": result["r2"],
                        "macro_season_mae": result["macro_season_mae"],
                        "season_mae_json": json.dumps(result["season_mae"]),
                    }
                )
        evaluation_rows.extend(
            [
                {
                    "evaluation_type": "exploratory_holdout_selected_candidate",
                    "cutoff_pct": evaluation["selected_cutoff_pct"],
                    "feature_bundle": evaluation["selected_feature_bundle"],
                    "split": "exploratory_holdout",
                    "n": evaluation["exploratory_holdout_metrics_selected_candidate"]["n"],
                    "n_seasons": evaluation["exploratory_holdout_metrics_selected_candidate"][
                        "n_seasons"
                    ],
                    "mae": evaluation["exploratory_holdout_metrics_selected_candidate"]["mae"],
                    "rmse": evaluation["exploratory_holdout_metrics_selected_candidate"]["rmse"],
                    "r2": evaluation["exploratory_holdout_metrics_selected_candidate"]["r2"],
                    "macro_season_mae": evaluation["exploratory_holdout_metrics_selected_candidate"][
                        "macro_season_mae"
                    ],
                    "season_mae_json": json.dumps(
                        evaluation["exploratory_holdout_metrics_selected_candidate"]["season_mae"]
                    ),
                },
                {
                    "evaluation_type": "exploratory_holdout_dummy_baseline",
                    "cutoff_pct": evaluation["selected_cutoff_pct"],
                    "feature_bundle": "dummy_median",
                    "split": "exploratory_holdout",
                    "n": evaluation["exploratory_holdout_metrics_dummy_baseline"]["n"],
                    "n_seasons": evaluation["exploratory_holdout_metrics_dummy_baseline"]["n_seasons"],
                    "mae": evaluation["exploratory_holdout_metrics_dummy_baseline"]["mae"],
                    "rmse": evaluation["exploratory_holdout_metrics_dummy_baseline"]["rmse"],
                    "r2": evaluation["exploratory_holdout_metrics_dummy_baseline"]["r2"],
                    "macro_season_mae": evaluation["exploratory_holdout_metrics_dummy_baseline"][
                        "macro_season_mae"
                    ],
                    "season_mae_json": json.dumps(
                        evaluation["exploratory_holdout_metrics_dummy_baseline"]["season_mae"]
                    ),
                },
            ]
        )
        evaluation_frame = pd.DataFrame(evaluation_rows)
        connection.register("feature_evaluation_frame", evaluation_frame)
        connection.execute(
            """CREATE OR REPLACE TABLE feature_evaluation_results AS
               SELECT * FROM feature_evaluation_frame"""
        )
        connection.unregister("feature_evaluation_frame")

    selected_feature_names = set(evaluation["selected_features"])
    ablation_by_feature = {
        result["feature"]: result
        for result in evaluation["feature_ablation_validation"]
    }
    correlated_partners: dict[str, list[str]] = {}
    for pair in evaluation["highly_correlated_feature_pairs_validation"]:
        correlated_partners.setdefault(pair["feature_a"], []).append(pair["feature_b"])
        correlated_partners.setdefault(pair["feature_b"], []).append(pair["feature_a"])
    decisions = []
    for feature in registry["features"]:
        ablation = ablation_by_feature.get(feature["name"])
        decisions.append(
            {
                **feature,
                "evaluation_decision": (
                    "pending_metadata_or_encoding"
                    if feature.get("status") != "candidate"
                    else "selected_candidate_not_finalized"
                    if feature["name"] in selected_feature_names
                    else "pruned_by_validation_ablation"
                    if ablation is not None
                    else "not_in_selected_bundle"
                ),
                "validation_ablation": ablation,
                "high_correlation_partners": correlated_partners.get(
                    feature["name"], []
                ),
            }
        )
    report = {
        "status": "EVALUATED_NOT_FINALIZED",
        "feature_registry": str(FEATURE_REGISTRY_PATH),
        "feature_registry_sha256": registry_sha256,
        "data_dictionary_sha256": hashlib.sha256(
            (FEATURE_REGISTRY_PATH.parent / "data_dictionary.json").read_bytes()
        ).hexdigest(),
        "dataset_scope": registry["project_scope"],
        "cutoff_candidate_tables": candidate_summaries,
        "evaluation": evaluation,
        "feature_decisions": decisions,
        "decisions_and_limitations": [
            "Use the 2003-2022 sensor era so the target cohort is temporally comparable across available sources.",
            "Keep the same mapped target campaigns at all three cutoff fractions; source gaps remain as NaN and are reported.",
            "Do not use admin_id, planting/harvest labels, future observations, or target-derived historical statistics as predictors.",
            "Use cwb rather than raw precipitation in the primary candidates because precipitation units remain provisional.",
            "Exclude vpd, awc, drainage codes, and coordinates until their units/codebook/CRS are resolved.",
            "Do not remove zero or high target yields; CY-Bench v1.8 filters invalid values and no further evidence justifies filtering.",
            "Do not use MAPE because the yield target contains zeros.",
            "Rolling-origin OOF validation is used for cutoff, group, and feature-ablation decisions.",
            "The 2019-2022 holdout was previously accessed for the initial all-candidate bundle; its current metrics are exploratory, not a pristine final test.",
            "Feature-level ablation and validation correlation are diagnostics; they do not automatically establish causality or final acceptance.",
            "Per-region holdout errors assess stability only in regions represented in development; transfer to unseen regions is not evaluated.",
            "This evaluation does not tune estimator hyperparameters and does not constitute final model selection.",
        ],
        "feature_aggregation_performed": True,
        "final_training_mart_created": False,
    }

    FEATURE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = FEATURE_REPORT_DIR / "feature_engineering_report.json"
    temporary_path = report_path.with_name(f".{report_path.stem}.{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_path.replace(report_path)
    return report
