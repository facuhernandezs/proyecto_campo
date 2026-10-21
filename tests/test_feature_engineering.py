from maize_forecasting.feature_engineering import (
    _feature_groups,
    _one_standard_error_selection,
)


def test_feature_groups_exclude_pending_context_until_encoded_or_validated():
    registry = {
        "features": [
            {"name": "weather", "group": "climate_core", "status": "candidate"},
            {"name": "awc", "group": "soil_static", "status": "candidate"},
            {
                "name": "latitude",
                "group": "spatial_context",
                "status": "candidate_pending_spatial_validation",
            },
            {
                "name": "drainage_code",
                "group": "soil_categorical",
                "status": "pending_one_hot_encoding_and_codebook",
            },
        ]
    }

    groups = _feature_groups(registry)

    assert groups == {"climate_core": ["weather"], "soil_static": ["awc"]}


def test_cutoff_one_standard_error_rule_prefers_earlier_eligible_checkpoint():
    results = [
        {
            "cutoff_pct": 25,
            "score": 1.2,
            "season_mae": {"a": 1.1, "b": 1.3},
        },
        {
            "cutoff_pct": 50,
            "score": 1.08,
            "season_mae": {"a": 1.0, "b": 1.16},
        },
        {
            "cutoff_pct": 75,
            "score": 1.0,
            "season_mae": {"a": 0.9, "b": 1.1},
        },
    ]

    best, selected, threshold, eligible = _one_standard_error_selection(
        results,
        score_key="score",
        complexity_key=lambda result: result["cutoff_pct"],
    )

    assert best["cutoff_pct"] == 75
    assert threshold > 1.0
    assert [result["cutoff_pct"] for result in eligible] == [50, 75]
    assert selected["cutoff_pct"] == 50


def test_bundle_one_standard_error_rule_prefers_fewer_features():
    results = [
        {
            "bundle": "larger",
            "feature_count": 8,
            "score": 1.0,
            "season_mae": {"a": 0.9, "b": 1.1},
        },
        {
            "bundle": "smaller",
            "feature_count": 3,
            "score": 1.08,
            "season_mae": {"a": 1.0, "b": 1.16},
        },
    ]

    best, selected, threshold, eligible = _one_standard_error_selection(
        results,
        score_key="score",
        complexity_key=lambda result: result["feature_count"],
    )

    assert best["bundle"] == "larger"
    assert threshold > 1.08
    assert len(eligible) == 2
    assert selected["bundle"] == "smaller"
