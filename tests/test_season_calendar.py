from datetime import date

from maize_forecasting.season_calendar import derive_campaign_calendar


def test_campaign_midpoint_uses_wrapping_calendar_and_leap_year():
    calendar = derive_campaign_calendar("2020/21", 2021, 300.0, 120.0)

    assert calendar["mapping_status"] == "MAPPED_STATIC_CALENDAR_PROXY"
    assert calendar["season_start_date"] == date(2020, 10, 26)
    assert calendar["season_end_date"] == date(2021, 4, 30)
    assert calendar["season_duration_days"] == 186
    assert calendar["midseason_cutoff_date"] == date(2021, 1, 27)
    assert calendar["cutoff_progress"] == 0.5
    assert calendar["crosses_calendar_year"] is True


def test_growing_season_end_can_precede_harvest_year():
    calendar = derive_campaign_calendar("2020/21", 2021, 222.0, 301.0)

    assert calendar["mapping_status"] == "MAPPED_STATIC_CALENDAR_PROXY"
    assert calendar["season_end_year"] == 2020
    assert calendar["calendar_end_year_matches_harvest_year"] is False
    assert calendar["season_end_date"] == date(2020, 10, 27)


def test_campaign_without_calendar_is_retained_but_unmapped():
    calendar = derive_campaign_calendar("2020/21", 2021, None, None)

    assert calendar["mapping_status"] == "MISSING_CROP_CALENDAR"
    assert calendar["season_start_date"] is None


def test_campaign_label_must_agree_with_harvest_year():
    calendar = derive_campaign_calendar("2020/21", 2022, 300.0, 120.0)

    assert calendar["mapping_status"] == "HARVEST_YEAR_LABEL_MISMATCH"
    assert calendar["season_start_date"] is None


def test_two_digit_season_suffix_handles_century_rollover():
    calendar = derive_campaign_calendar("1999/00", 2000, 300.0, 120.0)

    assert calendar["mapping_status"] == "MAPPED_STATIC_CALENDAR_PROXY"
    assert calendar["season_start_year"] == 1999
    assert calendar["season_end_year"] == 2000
