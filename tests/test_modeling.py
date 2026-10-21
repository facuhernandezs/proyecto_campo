import pandas as pd

from maize_forecasting.modeling import compare_models_rolling_origin


def test_regression_model_comparison_uses_forward_season_folds():
    rows = []
    for season_year in range(2003, 2023):
        for region in range(3):
            x = float(season_year - 2000 + region)
            rows.append(
                {
                    "season_start_year": season_year,
                    "administrative_unit_key": region,
                    "season_key": season_year - 2002,
                    "feature_x": x,
                    "feature_z": float(region),
                    "target_yield": 1.5 * x + 0.2 * region,
                }
            )

    results, selection = compare_models_rolling_origin(
        pd.DataFrame(rows), ["feature_x", "feature_z"]
    )

    assert "dummy_median" in {result["model"] for result in results}
    assert selection["selected_model"] in {result["model"] for result in results}
    for result in results:
        assert result["n"] == 30
        assert result["n_seasons"] == 10
        assert len(result["folds"]) == 4
        assert all(
            fold["train_years"][1] < fold["validation_years"][0]
            for fold in result["folds"]
        )
