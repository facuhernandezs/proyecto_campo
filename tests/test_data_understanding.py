import pandas as pd
import pytest

from maize_forecasting import data_understanding


def test_load_dataset_reads_only_requested_rows_and_columns(tmp_path, monkeypatch):
    source = pd.DataFrame({"adm_id": ["AR1", "AR2"], "yield": [2.5, 3.0]})
    source.to_csv(tmp_path / "yield.csv", index=False)
    monkeypatch.setattr(data_understanding, "MAIZE_DATA_DIR", tmp_path)
    monkeypatch.setattr(data_understanding, "INPUT_FILES", {"yield": "yield.csv"})

    loaded = data_understanding.load_dataset("yield", nrows=1, columns=["yield"])

    assert loaded.to_dict("records") == [{"yield": 2.5}]


def test_profile_dataset_aggregates_chunks_and_missing_values(tmp_path, monkeypatch):
    source = pd.DataFrame(
        {"adm_id": ["AR1", "AR2", "AR1"], "date": [20010101, 20010102, 20010103], "value": [1.0, None, 3.0]}
    )
    source.to_csv(tmp_path / "observations.csv", index=False)
    monkeypatch.setattr(data_understanding, "MAIZE_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        data_understanding, "INPUT_FILES", {"observations": "observations.csv"}
    )

    profile = data_understanding.profile_dataset("observations", chunksize=2)

    assert profile["rows"] == 3
    assert profile["distinct_counts"]["adm_id"] == 2
    assert profile["missing_values"]["value"] == 1
    assert profile["missing_percent"]["value"] == pytest.approx(100 / 3)
    assert profile["numeric_ranges"]["date"] == {"min": 20010101, "max": 20010103}
    assert profile["numeric_ranges"]["value"] == {"min": 1.0, "max": 3.0}


def test_load_dataset_rejects_unknown_names():
    with pytest.raises(ValueError, match="Dataset desconocido"):
        data_understanding.load_dataset("unknown")
