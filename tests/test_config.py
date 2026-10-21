from maize_forecasting.config import INPUT_FILES, MAIZE_DATA_DIR, PROJECT_ROOT


def test_project_paths_and_core_inputs_exist():
    assert PROJECT_ROOT.is_dir()
    assert MAIZE_DATA_DIR.is_dir()
    assert (MAIZE_DATA_DIR / INPUT_FILES["yield"]).is_file()
    assert (MAIZE_DATA_DIR / INPUT_FILES["meteorology"]).is_file()
