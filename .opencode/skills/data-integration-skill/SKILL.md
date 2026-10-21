---
name: data-integration-skill
description: Guide an AI agent in integrating heterogeneous agricultural datasets into a consistent, validated analytical structure suitable for downstream feature engineering and machine learning.
---



The skill focuses on:

- understanding the structure and meaning of each dataset
- identifying the grain of each table
- establishing canonical keys
- aligning spatial and temporal information
- aligning agricultural seasons and crop-development stages
- normalizing units and semantics
- reconciling different temporal frequencies
- detecting duplicates, gaps, inconsistencies, and invalid joins
- preventing temporal leakage during integration
- constructing canonical intermediate datasets
- producing a final ML training dataset at a clearly defined modeling grain

The agent must treat data integration as a **semantic alignment problem**, not merely a dataframe merging problem.

---

# Core Principle

> **Never join datasets until their grain, keys, temporal semantics, spatial semantics, units, and availability are understood.**

A technically valid join can still be scientifically incorrect.

The goal is not to produce one large table containing every variable.

The goal is to produce a dataset in which:

> **Every row represents one clearly defined modeling observation, and every variable belongs to that observation without introducing ambiguity or future information.**

---

# 1. Understand the Prediction Problem First

Before integrating data, the agent MUST identify:

```yaml
target:
prediction_unit:
spatial_unit:
temporal_unit:
prediction_date:
forecast_horizon:
crop:
crop_cycle:
available_information_cutoff:
```

Examples:

```yaml
target: yield
prediction_unit: administrative region × crop season
spatial_unit: adm_id
temporal_unit: daily
prediction_date: 2024-01-15
forecast_horizon: end_of_season
crop: maize
```

The integration strategy depends on the prediction problem.

A model predicting final yield before harvest requires a different training table from a model explaining yield after the entire season has already occurred.

---

# 2. Identify the Grain of Every Dataset

For every source dataset, determine:

```yaml
dataset:
grain:
primary_key:
spatial_key:
temporal_key:
crop_key:
frequency:
target_or_predictor:
```

The agent must explicitly describe what one row represents.

Example:

```yaml
dataset: meteo_maize_AR
grain: adm_id × date
frequency: daily

dataset: ndvi_maize_AR
grain: adm_id × date
frequency: satellite_observation

dataset: yield_maize_AR
grain: adm_id × crop_season
frequency: seasonal
```

The agent MUST NOT assume that similarly named columns imply the same grain.

---

# 3. Establish Canonical Keys

Identify the keys used to connect datasets.

Potential CY-Bench keys include:

```text
crop_name
adm_id
date
season
```

However, the agent must verify the actual semantics of each key before using it.

For each key:

```yaml
key:
meaning:
data_type:
uniqueness:
nullable:
source_datasets:
validation:
```

Example:

```yaml
key: adm_id
meaning: administrative geographic identifier
data_type: integer
uniqueness: not globally unique without geographic context
nullable: false
```

Do not create artificial join keys unless the semantic relationship is understood.

---

# 4. Spatial Alignment

All spatial datasets must be mapped to a common spatial definition.

The agent must verify:

- `adm_id` meaning
- administrative level
- geographic boundaries
- country
- region
- coordinate system when coordinates are present
- whether different datasets refer to exactly the same spatial unit

Example:

```text
weather
adm_id = 42

soil
adm_id = 42

yield
adm_id = 42
```

does NOT automatically prove that all three datasets refer to the same spatial concept.

The agent should verify metadata and source definitions.

---

# 5. Temporal Alignment

Temporal alignment is one of the highest-priority integration tasks.

The agent must distinguish between:

```text
calendar date
crop season
planting date
harvest date
days after planting
growth stage
prediction date
```

A raw date should not automatically be treated as the modeling time axis.

For agricultural modeling, the agent should consider deriving:

```text
year
month
day_of_year
season
days_since_planting
days_until_harvest
growth_stage
```

when the required information is available.

---

# 6. Agricultural Season Alignment

The agent must determine how observations belong to a crop cycle.

For example:

```text
calendar dates
      ↓
crop calendar
      ↓
crop season
      ↓
days since planting
      ↓
growth stage
```

The agent must avoid assuming that:

```text
season = calendar year
```

unless the dataset explicitly supports that interpretation.

If a crop season crosses calendar years, the agent must preserve the crop-cycle identity independently of calendar year.

Example:

```text
season_id = maize_AR_2024_2025
```

is conceptually different from:

```text
year = 2024
```

---

# 7. Frequency Reconciliation

Different datasets may have different observation frequencies.

For example:

```text
Meteorology      daily
NDVI              irregular / periodic
FPAR              periodic
Soil moisture     periodic
Yield             seasonal
```

The agent must never solve this simply by blindly joining on `date`.

Instead, it must define the required modeling grain first.

Example:

```text
weather daily
     ↓
30-day aggregation
     ↓
prediction_date
     ↓
training observation
```

Possible temporal transformations include:

```text
daily → weekly
daily → monthly
daily → crop-stage
daily → rolling window
daily → season
```

Every aggregation must specify:

```yaml
source_frequency:
target_frequency:
aggregation:
window:
reason:
information_cutoff:
```

Example:

```yaml
source_frequency: daily
target_frequency: prediction_date
aggregation: rolling_mean
window: 30_days
reason: represent recent thermal conditions
information_cutoff: prediction_date
```

---

# 8. Temporal Availability and Leakage

The agent MUST determine when every variable becomes available.

For a prediction at:

```text
prediction_date = T
```

a feature may only use information available at or before `T`.

Invalid:

```text
yield
future NDVI
future precipitation
post-harvest measurements
```

Valid:

```text
precipitation up to T
NDVI available up to T
soil moisture available up to T
crop calendar information known by T
```

The integration layer must preserve temporal provenance so downstream feature engineering can enforce leakage prevention.

---

# 9. Unit Normalization

Before joining or aggregating datasets, identify measurement units.

For every numerical variable:

```yaml
variable:
source_unit:
canonical_unit:
conversion:
validation:
```

Examples of variables requiring explicit unit validation:

```text
temperature
precipitation
radiation
evapotranspiration
VPD
soil moisture
yield
```

Never silently convert units.

Never assume units from variable names alone.

If units cannot be established confidently, mark the variable:

```text
UNIT_UNVERIFIED
```

and prevent it from entering the canonical analytical dataset until resolved.

---

# 10. Data Type and Semantic Standardization

Standardize:

- column names
- data types
- date formats
- categorical values
- missing-value representations
- units
- boolean representations
- geographic identifiers

Example:

```text
"2024-01-01"
2024-01-01
01/01/2024
```

must be normalized to a single canonical date representation.

Likewise:

```text
NA
null
"N/A"
-999
```

must not automatically be treated as equivalent without understanding their source semantics.

---

# 11. Missingness and Temporal Gaps

The agent must distinguish between:

```text
missing observation
zero measurement
not applicable
sensor/satellite gap
outside observation period
```

For every dataset, evaluate:

```yaml
missing_rate:
missing_variables:
missing_periods:
longest_gap:
spatial_gaps:
systematic_missingness:
```

Do not automatically impute during the integration layer.

Integration should preserve the original information state.

Imputation belongs in a clearly defined downstream preprocessing or feature-engineering step unless the integration itself requires a justified transformation.

---

# 12. Duplicate Detection

Before joining datasets, check whether the proposed key is actually unique.

Example:

```text
adm_id + date
```

should be tested for duplicates before being treated as a primary key.

If duplicates exist, determine why:

```text
multiple observations
multiple measurements
different products
data duplication
different spatial units
```

Do not use:

```python
drop_duplicates()
```

as a default solution.

Duplicates are a data-quality problem until their meaning is understood.

---

# 13. Join Validation

Every join must have an explicit relationship.

Allowed relationship types should be documented:

```text
1:1
1:N
N:1
N:N
```

The agent should reject unexplained many-to-many joins.

For each join:

```yaml
left_dataset:
right_dataset:
left_key:
right_key:
relationship:
expected_match_rate:
actual_match_rate:
unmatched_left:
unmatched_right:
validation_status:
```

Example:

```yaml
left_dataset: weather_daily
right_dataset: crop_calendar
left_key: adm_id
right_key: adm_id
relationship: N:1
validation_status: VALIDATED
```

---

# 14. Never Create a "Mega Table" by Default

The agent should NOT automatically merge all source CSVs into one table.

Instead, preserve datasets according to their natural grains.

Recommended architecture:

```text
RAW
 │
 ├── weather
 ├── vegetation
 ├── soil
 ├── crop_calendar
 ├── crop_mask
 └── yield
       │
       ▼
STANDARDIZED
       │
       ▼
CANONICAL INTERMEDIATE TABLES
       │
       ▼
FEATURE ENGINEERING
       │
       ▼
ML TRAINING MART
```

The objective is not minimizing the number of tables.

The objective is preserving semantic correctness.

---

# 15. Canonical Intermediate Tables

Create stable intermediate representations for major concepts.

Potential CY-Bench structure:

```text
dim_crop
dim_location
dim_date
dim_season

fact_weather_daily
fact_vegetation_daily
fact_soil_daily
fact_crop_calendar
fact_yield
```

The exact tables must depend on the actual source schemas.

Do not force every source into the same grain.

---

# 16. Star Schema Guidance

A dimensional model may be useful for analytical organization.

Potential dimensions:

```text
dim_crop
dim_location
dim_date
dim_season
```

Potential facts:

```text
fact_weather
fact_vegetation
fact_soil
fact_yield
```

However:

> The star schema is an analytical representation, not necessarily the final machine-learning dataset.

The agent must distinguish:

```text
analytical model
```

from:

```text
modeling dataset
```

---

# 17. Define the Modeling Grain

Before creating the final training dataset, explicitly declare:

```yaml
training_grain:
one_row_represents:
target:
prediction_date:
feature_cutoff:
```

For example:

```yaml
training_grain: adm_id × season × prediction_date
one_row_represents: one yield forecast for one region at one point in the crop cycle
target: yield
feature_cutoff: prediction_date
```

Alternatively, for end-of-season modeling:

```yaml
training_grain: adm_id × season
one_row_represents: one completed crop season in one region
target: yield
```

The agent must never construct the training table before this decision is explicit.

---

# 18. Build the ML Training Mart

The final training dataset should be a materialized analytical object built from validated intermediate tables.

Conceptually:

```text
adm_id
season
prediction_date

weather_features
vegetation_features
soil_features
crop_features

yield
```

Example:

```text
adm_id
season
prediction_date
prec_30d
tavg_30d
vpd_14d
rad_30d
ndvi_latest
ndvi_change
soil_moisture_latest
yield
```

The exact features belong to Feature Engineering, not Data Integration.

Data Integration is responsible for ensuring that those source signals can be correctly aligned to the modeling observation.

---

# 19. Data Provenance

Every canonical variable should retain enough information to trace it back to its source.

Example:

```yaml
variable: tavg
source_dataset: meteo_maize_AR
source_column: tavg
transformation: none
unit: verified
temporal_transformation: none
spatial_transformation: none
```

For derived values:

```yaml
variable: prec_30d
source_dataset: meteo_maize_AR
source_column: prec
transformation: rolling_sum
window: 30_days
cutoff: prediction_date
```

This enables reproducibility and debugging.

---

# 20. Integration Validation

Before the data is passed to Feature Engineering, the agent must generate an integration validation report.

Minimum checks:

```text
[ ] Dataset grains identified
[ ] Primary/canonical keys identified
[ ] Spatial alignment validated
[ ] Temporal alignment validated
[ ] Crop-season alignment validated
[ ] Units verified
[ ] Data types standardized
[ ] Duplicate checks passed
[ ] Missingness analyzed
[ ] Temporal gaps analyzed
[ ] Join cardinalities validated
[ ] Many-to-many joins investigated
[ ] Leakage risks checked
[ ] Training grain defined
[ ] Target alignment validated
[ ] Provenance available
```

---

# 21. Integration Report

The agent should produce a structured report:

```yaml
integration_status: PASS

datasets:
  - name:
    grain:
    spatial_key:
    temporal_key:
    frequency:
    unit_status:
    missingness_status:
    duplicate_status:

joins:
  - left:
    right:
    keys:
    relationship:
    match_rate:
    status:

training_grain:
target:
prediction_date:
information_cutoff:

leakage_risks:
  - description:
    severity:
    mitigation:

data_quality_issues:
  - description:
    severity:
    status:

unresolved_questions:
  - question:
```

Possible final status values:

```text
PASS
PASS_WITH_WARNINGS
BLOCKED
```

`BLOCKED` should be used when unresolved integration problems could invalidate downstream modeling.

---

# 22. Separation of Responsibilities

The agent must maintain a strict separation between:

### Data Integration

> "How do these datasets correctly represent the same agricultural observation?"

### Feature Engineering

> "What information can we derive from those aligned observations?"

### Feature Evaluation

> "Does that information improve the model reliably?"

### Modeling

> "Which predictive model best captures the relationship?"

Do not move unresolved integration problems into Feature Engineering.

For example:

```text
"Let's calculate a 30-day precipitation feature."
```

is Feature Engineering.

But:

```text
"Which 30 days belong to this crop season?"
```

is Data Integration.

---

# 23. Integration Failure Rules

The agent MUST reject or block an integration when:

- dataset grain is unknown
- join keys are ambiguous
- spatial units are inconsistent
- temporal semantics are unclear
- units are unknown for a critical variable
- unexplained many-to-many joins occur
- target alignment is ambiguous
- future information enters the observation
- crop seasons cannot be reliably identified
- duplicate records change the intended grain
- missingness is incorrectly interpreted as zero
- aggregation is performed without a defined reason
- the modeling grain has not been defined

The agent should explain **why** integration failed rather than silently modifying the data.

---

# 24. Decision Framework

When encountering a new dataset, ask:

```text
1. What does one row represent?
2. What is its grain?
3. What identifies the observation?
4. What spatial unit does it represent?
5. What temporal unit does it represent?
6. What crop/cycle does it represent?
7. What are its units?
8. When does the information become available?
9. Which existing dataset can it be related to?
10. At what grain should the relationship occur?
11. Does the join preserve information?
12. Does the join introduce duplication?
13. Does the join introduce future information?
14. Should the data be aggregated?
15. What should the canonical representation be?
```

Only after these questions are answered should the agent construct the integration.

---

# 25. Guiding Principle

The agent should continuously ask:

> **"What does one row mean, and can I prove that every variable attached to this row refers to the same agricultural observation?"**

If the answer is unclear, the agent should stop integration and investigate the data rather than guessing.

The final objective is not:

```text
all CSVs → one dataframe
```

It is:

```text
heterogeneous sources
        ↓
semantic alignment
        ↓
validated canonical data
        ↓
well-defined modeling grain
        ↓
leakage-safe training dataset
```

Only then should Feature Engineering begin.