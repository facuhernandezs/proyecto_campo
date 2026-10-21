"""Calendario aproximado por unidad administrativa y campana de rendimiento."""

import math
import re
from datetime import date, timedelta
from typing import Any

import duckdb

SEASON_LABEL_PATTERN = re.compile(r"^(\d{4})/(\d{2})$")
TEMPORAL_FACTS = {
    "meteorology": "fact_meteorology",
    "soil_moisture": "fact_soil_moisture",
    "ndvi": "fact_ndvi",
    "fpar": "fact_fpar",
}


def _date_from_day_of_year(year: int, day_of_year: float) -> date:
    """Convierte DOY fraccional al dia calendario mas cercano."""
    rounded_day = math.floor(day_of_year + 0.5)
    first_day = date(year, 1, 1)
    days_in_year = (date(year + 1, 1, 1) - first_day).days
    if not 1 <= rounded_day <= days_in_year:
        raise ValueError(f"DOY {day_of_year} fuera de rango para {year}.")
    return first_day + timedelta(days=rounded_day - 1)


def derive_campaign_calendar(
    season_label: str,
    harvest_year: int,
    sos: float | None,
    eos: float | None,
) -> dict[str, Any]:
    """Deriva limites proxy y cutoff de mitad de temporada para una campana."""
    result: dict[str, Any] = {
        "season_start_year": None,
        "season_end_year": None,
        "season_start_date": None,
        "midseason_cutoff_date": None,
        "season_end_date": None,
        "season_duration_days": None,
        "cutoff_progress": None,
        "crosses_calendar_year": None,
        "calendar_end_year_matches_harvest_year": None,
        "mapping_status": None,
        "mapping_note": None,
    }

    match = SEASON_LABEL_PATTERN.fullmatch(season_label)
    if match is None:
        result["mapping_status"] = "INVALID_SEASON_LABEL"
        result["mapping_note"] = "Expected a YYYY/YY crop-season label."
        return result

    start_year = int(match.group(1))
    result["season_start_year"] = start_year
    expected_harvest_year = start_year + 1
    label_end_suffix = int(match.group(2))
    if (
        label_end_suffix != expected_harvest_year % 100
        or expected_harvest_year != harvest_year
    ):
        result["mapping_status"] = "HARVEST_YEAR_LABEL_MISMATCH"
        result["mapping_note"] = (
            f"Season label implies harvest year {expected_harvest_year}; "
            f"yield records state {harvest_year}."
        )
        return result

    if sos is None or eos is None:
        result["mapping_status"] = "MISSING_CROP_CALENDAR"
        result["mapping_note"] = "No static sos/eos calendar is available for this admin_id."
        return result

    crosses_calendar_year = sos > eos
    end_year = start_year + int(crosses_calendar_year)
    result["season_end_year"] = end_year
    result["crosses_calendar_year"] = crosses_calendar_year
    result["calendar_end_year_matches_harvest_year"] = end_year == harvest_year

    try:
        start_date = _date_from_day_of_year(start_year, sos)
        end_date = _date_from_day_of_year(end_year, eos)
    except ValueError as exc:
        result["mapping_status"] = "INVALID_DAY_OF_YEAR"
        result["mapping_note"] = str(exc)
        return result

    duration_days = (end_date - start_date).days
    if duration_days <= 0:
        result["mapping_status"] = "NON_POSITIVE_SEASON_DURATION"
        result["mapping_note"] = "Rounded season end is not after rounded season start."
        return result

    midpoint_offset = duration_days // 2
    result.update(
        {
            "season_start_date": start_date,
            "midseason_cutoff_date": start_date + timedelta(days=midpoint_offset),
            "season_end_date": end_date,
            "season_duration_days": duration_days,
            "cutoff_progress": midpoint_offset / duration_days,
            "mapping_status": "MAPPED_STATIC_CALENDAR_PROXY",
            "mapping_note": (
                "Dates are projected from static DOY onto the first year in "
                "planting_year; eos is growing-season end, not harvest date."
            ),
        }
    )
    return result


def build_season_alignment(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Build the season bridge and stage views, retaining unmapped yield rows."""
    campaigns = connection.execute(
        """
        SELECT yield.administrative_unit_key,
               yield.season_key,
               season.season_label,
               season.harvest_year,
               season.planting_date_source_label,
               season.harvest_date_source_label,
               calendar.sos,
               calendar.eos
        FROM fact_yield AS yield
        INNER JOIN dim_season AS season USING (season_key)
        LEFT JOIN fact_crop_calendar AS calendar USING (administrative_unit_key)
        ORDER BY yield.administrative_unit_key, season.season_key
        """
    ).fetchall()

    rows = []
    for (
        administrative_unit_key,
        season_key,
        season_label,
        harvest_year,
        planting_date_label,
        harvest_date_label,
        sos,
        eos,
    ) in campaigns:
        derived = derive_campaign_calendar(
            season_label,
            harvest_year,
            float(sos) if sos is not None else None,
            float(eos) if eos is not None else None,
        )
        rows.append(
            (
                administrative_unit_key,
                season_key,
                season_label,
                harvest_year,
                planting_date_label,
                harvest_date_label,
                sos,
                eos,
                *(
                    derived[column]
                    for column in (
                        "season_start_year",
                        "season_end_year",
                        "season_start_date",
                        "midseason_cutoff_date",
                        "season_end_date",
                        "season_duration_days",
                        "cutoff_progress",
                        "crosses_calendar_year",
                        "calendar_end_year_matches_harvest_year",
                        "mapping_status",
                        "mapping_note",
                    )
                ),
            )
        )

    connection.execute(
        """
        CREATE TABLE bridge_administrative_season (
            administrative_unit_key BIGINT,
            season_key INTEGER,
            season_label VARCHAR,
            harvest_year INTEGER,
            planting_date_source_label VARCHAR,
            harvest_date_source_label VARCHAR,
            sos_day_of_year DOUBLE,
            eos_day_of_year DOUBLE,
            season_start_year INTEGER,
            season_end_year INTEGER,
            season_start_date DATE,
            midseason_cutoff_date DATE,
            season_end_date DATE,
            season_duration_days INTEGER,
            cutoff_progress DOUBLE,
            crosses_calendar_year BOOLEAN,
            calendar_end_year_matches_harvest_year BOOLEAN,
            mapping_status VARCHAR,
            mapping_note VARCHAR,
            PRIMARY KEY (administrative_unit_key, season_key)
        )
        """
    )
    placeholders = ", ".join("?" for _ in range(19))
    connection.executemany(
        f"INSERT INTO bridge_administrative_season VALUES ({placeholders})", rows
    )

    mapped_rows = connection.execute(
        """SELECT COUNT(*) FROM bridge_administrative_season
           WHERE mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'"""
    ).fetchone()[0]
    missing_calendar_rows = connection.execute(
        """SELECT COUNT(*) FROM bridge_administrative_season
           WHERE mapping_status = 'MISSING_CROP_CALENDAR'"""
    ).fetchone()[0]
    invalid_rows = connection.execute(
        """SELECT COUNT(*) FROM bridge_administrative_season
           WHERE mapping_status NOT IN (
               'MAPPED_STATIC_CALENDAR_PROXY', 'MISSING_CROP_CALENDAR'
           )"""
    ).fetchone()[0]
    mapped_yield_admin_units = connection.execute(
        """SELECT COUNT(DISTINCT administrative_unit_key)
           FROM bridge_administrative_season
           WHERE mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'"""
    ).fetchone()[0]

    overlaps = connection.execute(
        """
        WITH ordered AS (
            SELECT administrative_unit_key, season_key, season_end_date,
                   LEAD(season_start_date) OVER (
                       PARTITION BY administrative_unit_key ORDER BY season_start_date
                   ) AS next_season_start_date
            FROM bridge_administrative_season
            WHERE mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'
        )
        SELECT COUNT(*) FROM ordered
        WHERE next_season_start_date IS NOT NULL
          AND season_end_date >= next_season_start_date
        """
    ).fetchone()[0]
    if overlaps:
        raise ValueError(
            f"Hay {overlaps} intervalos administrativos de campana solapados; "
            "no se crean vistas de etapa ambiguas."
        )

    yield_view = """
        CREATE VIEW vw_yield_midseason_calendar AS
        SELECT fact.*,
               bridge.season_start_date,
               bridge.midseason_cutoff_date,
               bridge.season_end_date,
               bridge.season_duration_days,
               bridge.crosses_calendar_year,
               bridge.mapping_status AS season_calendar_status,
               CASE WHEN bridge.mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'
                    THEN 'mid' END AS prediction_phase
        FROM fact_yield AS fact
        LEFT JOIN bridge_administrative_season AS bridge
          USING (administrative_unit_key, season_key)
    """
    connection.execute(yield_view)
    yield_view_rows = connection.execute(
        "SELECT COUNT(*) FROM vw_yield_midseason_calendar"
    ).fetchone()[0]
    source_yield_rows = connection.execute("SELECT COUNT(*) FROM fact_yield").fetchone()[0]
    if yield_view_rows != source_yield_rows:
        raise ValueError("La vista de rendimiento/cutoff altero la cardinalidad objetivo.")

    observation_checks = {}
    for dataset_name, fact_table in TEMPORAL_FACTS.items():
        view_name = f"vw_{dataset_name}_season_stage"
        connection.execute(
            f"""
            CREATE VIEW {view_name} AS
            WITH joined AS (
                SELECT fact.*,
                       calendar.calendar_date AS observation_date,
                       bridge.season_key AS campaign_season_key,
                       bridge.season_label AS campaign_label,
                       bridge.season_start_date,
                       bridge.midseason_cutoff_date,
                       bridge.season_end_date,
                       bridge.season_duration_days,
                       CASE WHEN bridge.season_key IS NOT NULL
                            THEN 'IN_CAMPAIGN'
                            WHEN EXISTS (
                                SELECT 1 FROM bridge_administrative_season AS missing
                                WHERE missing.administrative_unit_key =
                                      fact.administrative_unit_key
                                  AND missing.mapping_status = 'MISSING_CROP_CALENDAR'
                            ) THEN 'CROP_CALENDAR_UNAVAILABLE'
                            WHEN EXISTS (
                                SELECT 1 FROM bridge_administrative_season AS available
                                WHERE available.administrative_unit_key =
                                      fact.administrative_unit_key
                                  AND available.mapping_status =
                                      'MAPPED_STATIC_CALENDAR_PROXY'
                            ) THEN 'OUTSIDE_CAMPAIGN_WINDOW'
                            ELSE 'NO_YIELD_CAMPAIGN'
                       END AS observation_mapping_status,
                       CASE WHEN bridge.season_key IS NOT NULL
                            THEN DATE_DIFF('day', bridge.season_start_date,
                                           calendar.calendar_date)
                       END AS days_since_sos
                FROM {fact_table} AS fact
                INNER JOIN dim_date AS calendar USING (date_key)
                LEFT JOIN bridge_administrative_season AS bridge
                  ON bridge.administrative_unit_key = fact.administrative_unit_key
                 AND bridge.mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'
                 AND calendar.calendar_date BETWEEN bridge.season_start_date
                                                AND bridge.season_end_date
            ), progress AS (
                SELECT *,
                       CASE WHEN campaign_season_key IS NOT NULL
                            THEN days_since_sos + 1 END AS season_day,
                       CASE WHEN campaign_season_key IS NOT NULL
                            THEN days_since_sos::DOUBLE / season_duration_days
                       END AS season_progress
                FROM joined
            )
            SELECT *,
                   CASE WHEN season_progress IS NULL THEN NULL
                        WHEN season_progress < 1.0 / 3.0 THEN 'early'
                        WHEN season_progress < 2.0 / 3.0 THEN 'mid'
                        ELSE 'late' END AS season_phase,
                   CASE WHEN campaign_season_key IS NOT NULL
                        THEN observation_date <= midseason_cutoff_date
                   END AS observation_on_or_before_cutoff
            FROM progress
            """
        )

        input_rows = connection.execute(
            f"SELECT COUNT(*) FROM {fact_table}"
        ).fetchone()[0]
        view_rows, mapped_observations, before_cutoff, after_cutoff = connection.execute(
            f"""SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE campaign_season_key IS NOT NULL),
                       COUNT(*) FILTER (WHERE observation_on_or_before_cutoff),
                       COUNT(*) FILTER (WHERE observation_on_or_before_cutoff = FALSE)
                FROM {view_name}"""
        ).fetchone()
        multiple_mappings = connection.execute(
            f"""SELECT COUNT(*) FROM (
                    SELECT administrative_unit_key, date_key, COUNT(*) AS matches
                    FROM {view_name}
                    WHERE campaign_season_key IS NOT NULL
                    GROUP BY administrative_unit_key, date_key
                    HAVING COUNT(*) > 1
                )"""
        ).fetchone()[0]
        if input_rows != view_rows or multiple_mappings:
            raise ValueError(
                f"La vista {view_name} no preserva las filas fuente "
                f"(input={input_rows}, view={view_rows}, duplicados={multiple_mappings})."
            )
        mapping_status_counts = dict(
            connection.execute(
                f"""SELECT observation_mapping_status, COUNT(*)
                    FROM {view_name}
                    GROUP BY observation_mapping_status"""
            ).fetchall()
        )
        phase_counts = dict(
            connection.execute(
                f"""SELECT season_phase, COUNT(*) FROM {view_name}
                    WHERE season_phase IS NOT NULL GROUP BY season_phase"""
            ).fetchall()
        )
        admin_units_by_mapping_status = dict(
            connection.execute(
                f"""SELECT observation_mapping_status,
                           COUNT(DISTINCT administrative_unit_key)
                    FROM {view_name}
                    GROUP BY observation_mapping_status"""
            ).fetchall()
        )
        campaign_windows_with_observations = connection.execute(
            f"""SELECT COUNT(DISTINCT (administrative_unit_key, campaign_season_key))
                FROM {view_name}
                WHERE campaign_season_key IS NOT NULL"""
        ).fetchone()[0]
        admin_units_with_observations = admin_units_by_mapping_status.get(
            "IN_CAMPAIGN", 0
        )
        observation_checks[dataset_name] = {
            "input_rows": input_rows,
            "view_rows": view_rows,
            "observations_inside_campaign": mapped_observations,
            "observations_on_or_before_cutoff": before_cutoff,
            "observations_after_cutoff": after_cutoff,
            "multiply_mapped_observation_keys": multiple_mappings,
            "yield_admin_units_with_calendar": mapped_yield_admin_units,
            "admin_units_with_observations_inside_campaign": admin_units_with_observations,
            "admin_units_without_observations_inside_campaign": (
                mapped_yield_admin_units - admin_units_with_observations
            ),
            "yield_campaign_windows_with_observations": campaign_windows_with_observations,
            "observation_mapping_status_counts": mapping_status_counts,
            "admin_units_by_mapping_status": admin_units_by_mapping_status,
            "season_phase_counts": phase_counts,
            "date_is_availability_proxy": True,
        }

    duration_stats = connection.execute(
        """SELECT MIN(season_duration_days), MEDIAN(season_duration_days),
                  MAX(season_duration_days),
                  COUNT(*) FILTER (WHERE crosses_calendar_year),
                  COUNT(*) FILTER (WHERE NOT crosses_calendar_year),
                  COUNT(*) FILTER (
                      WHERE NOT calendar_end_year_matches_harvest_year
                  )
           FROM bridge_administrative_season
           WHERE mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'"""
    ).fetchone()
    calendar_end_harvest_mismatch_rows = connection.execute(
        """SELECT COUNT(*) FROM bridge_administrative_season
           WHERE mapping_status = 'MAPPED_STATIC_CALENDAR_PROXY'
             AND NOT calendar_end_year_matches_harvest_year"""
    ).fetchone()[0]
    static_calendar_profile = connection.execute(
        """WITH calendars AS (
               SELECT sos, eos, sos > eos AS crosses_calendar_year,
                      CASE WHEN sos > eos THEN 365 - sos + eos ELSE eos - sos END
                          AS season_duration_doy
               FROM fact_crop_calendar
           )
           SELECT COUNT(*) AS admin_units,
                  COUNT(*) FILTER (WHERE crosses_calendar_year) AS crossing_year_units,
                  COUNT(*) FILTER (WHERE NOT crosses_calendar_year) AS same_year_units,
                  MIN(season_duration_doy) AS min_duration_doy,
                  MEDIAN(season_duration_doy) AS median_duration_doy,
                  MAX(season_duration_doy) AS max_duration_doy
           FROM calendars"""
    ).fetchone()
    missing_calendar_admins = connection.execute(
        """SELECT COUNT(DISTINCT administrative_unit_key)
           FROM bridge_administrative_season
           WHERE mapping_status = 'MISSING_CROP_CALENDAR'"""
    ).fetchone()[0]

    findings = [
        {
            "severity": "warning",
            "status": "unmapped",
            "issue": "Some yield admin-season rows have no crop_calendar entry.",
            "count": missing_calendar_rows,
            "admin_units": missing_calendar_admins,
            "action": "Keep those target rows; their midseason cutoff and season phase remain null.",
        },
        {
            "severity": "info",
            "status": "documented",
            "issue": "Non-wrapping calendars can end in the start year while yield harvest is reported in the following year.",
            "count": calendar_end_harvest_mismatch_rows,
            "action": "Keep separate SOS/EOS growing-season dates from yield harvest dates.",
        },
    ]
    for dataset_name, checks in observation_checks.items():
        admins_without_yield = checks["admin_units_by_mapping_status"].get(
            "NO_YIELD_CAMPAIGN", 0
        )
        if admins_without_yield:
            findings.append(
                {
                    "severity": "info",
                    "status": "no_target_match",
                    "issue": f"{dataset_name} contains admin units without yield campaigns.",
                    "admin_units": admins_without_yield,
                    "observations": checks["observation_mapping_status_counts"][
                        "NO_YIELD_CAMPAIGN"
                    ],
                    "action": "Retain source observations; they cannot form supervised yield rows without target data.",
                }
            )
        admins_without_inseason_data = checks[
            "admin_units_without_observations_inside_campaign"
        ]
        if admins_without_inseason_data:
            findings.append(
                {
                    "severity": "warning",
                    "status": "no_observation_in_window",
                    "issue": f"{dataset_name} has no in-campaign observations for some yield admin units with calendars.",
                    "admin_units": admins_without_inseason_data,
                    "action": "Distinguish source temporal coverage from within-season observation gaps before creating training rows; do not impute here.",
                }
            )

    report = {
        "status": "PASS_WITH_WARNINGS",
        "prediction_definition": {
            "target": "final yield",
            "training_observation_grain": "administrative unit x crop-season label",
            "forecast_checkpoint": "midpoint of local static sos/eos calendar",
            "forecast_horizon": "end of crop season",
        },
        "decisions": [
            "Parse the start year from planting_year labels of the form YYYY/YY.",
            "Map SOS to the first year in planting_year; map EOS to that year unless eos < sos, then use the following year.",
            "Round fractional DOY to the nearest calendar day (floor(doy + 0.5)); retain source SOS/EOS unchanged.",
            "Set cutoff to start_date plus floor(season_duration_days / 2).",
            "Use early/mid/late thirds as generic season phases, not named phenological stages.",
            "Retain planting_date/harvest_date source labels, but do not parse them as exact dates because their day-level semantics are undocumented.",
            "Treat source date as an availability proxy because no publication timestamp is supplied.",
            "Do not replace missing crop calendars with a national or neighboring-region calendar.",
        ],
        "assumptions": [
            "The first year in planting_year is the calendar year of SOS; local labels all imply harvest_year = start_year + 1.",
            "Static SOS/EOS are a typical seasonal window and approximate each historical campaign.",
            "Harvest date is not equated with EOS; EOS is the end of growing season.",
            "Observation date is used as the information cutoff proxy; publication latency is not represented in the source files.",
        ],
        "checks": {
            "yield_rows": len(campaigns),
            "mapped_yield_rows": mapped_rows,
            "missing_crop_calendar_yield_rows": missing_calendar_rows,
            "yield_admin_units_missing_crop_calendar": missing_calendar_admins,
            "invalid_calendar_or_season_label_rows": invalid_rows,
            "overlapping_campaign_intervals": overlaps,
            "static_crop_calendar_profile": {
                "admin_units": static_calendar_profile[0],
                "crosses_calendar_year": static_calendar_profile[1],
                "same_calendar_year": static_calendar_profile[2],
                "min_duration_doy": static_calendar_profile[3],
                "median_duration_doy": static_calendar_profile[4],
                "max_duration_doy": static_calendar_profile[5],
            },
            "yield_midseason_view_rows": yield_view_rows,
            "season_duration_days": {
                "min": duration_stats[0],
                "median": duration_stats[1],
                "max": duration_stats[2],
            },
            "crosses_calendar_year_admin_seasons": duration_stats[3],
            "same_calendar_year_admin_seasons": duration_stats[4],
            "calendar_end_year_differs_from_harvest_year_rows": calendar_end_harvest_mismatch_rows,
            "date_label_harvest_year_mismatches": connection.execute(
                """SELECT COUNT(*) FROM bridge_administrative_season
                   WHERE mapping_status = 'HARVEST_YEAR_LABEL_MISMATCH'"""
            ).fetchone()[0],
        },
        "temporal_observation_views": observation_checks,
        "findings": findings,
        "feature_aggregation_performed": False,
        "training_mart_created": False,
    }
    return report
