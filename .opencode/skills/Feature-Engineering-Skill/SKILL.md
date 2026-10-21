---
name: Feature-Engineering-Skill
description: Guide the AI agent in designing, evaluating, and documenting features for agricultural crop-yield prediction.
---


Feature engineering must be **hypothesis-driven**, not simply a process of generating mathematical transformations until model performance improves.

The objective is to create features that:

1. Represent meaningful agricultural or environmental phenomena.
2. Are available at prediction time.
3. Add information beyond existing features.
4. Have a plausible relationship with the target.
5. Improve predictive performance when validated correctly.
6. Remain interpretable whenever possible.

The primary target for this project is crop yield (`yield`).

---

# Core Principle

> **A feature should exist because there is a reason to believe it contains information about the target, not merely because it can be calculated.**

Do not create features solely because they produce a better validation score.

Every candidate feature should have an explicit hypothesis.

Example:

```text
Feature:
water_deficit_14d

Formula:
prec_14d - et0_14d

Hypothesis:
When atmospheric water demand exceeds precipitation over an extended
period, crops may experience water stress, which can reduce yield.
```

This is preferable to an arbitrary transformation such as:

```text
prec * et0 * vpd
```

unless a meaningful hypothesis exists for that interaction.

---

# 1. Understand the Prediction Problem First

Before creating features, determine:

- What is the target?
- What is the prediction date?
- What information is available at prediction time?
- What is the spatial unit?
- What is the temporal resolution?
- What is the crop growth cycle?
- What is the forecasting horizon?

Do not begin feature engineering before answering these questions.

For crop-yield forecasting, the agent should understand that the target represents a yield outcome associated with a particular:

```text
crop
+
location
+
growing season
```

Daily observations should therefore be interpreted within the context of the growing season.

---

# 2. Start From Agricultural Mechanisms

Prioritize features representing mechanisms known or reasonably expected to influence crop development.

For maize, relevant mechanisms include:

- temperature stress
- water availability
- atmospheric water demand
- radiation availability
- vegetation development
- soil moisture
- accumulated heat
- precipitation
- drought
- excessive heat
- growing-season conditions

Given variables such as:

```text
tmin
tmax
tavg
prec
rad
et0
vpd
cwb
ndvi
fpar
soil_moisture
```

prefer meaningful derived variables over arbitrary mathematical combinations.

---

# 3. Feature Hypothesis Requirement

Before implementing a candidate feature, the agent MUST define:

```yaml
name:
formula:
hypothesis:
expected_relationship:
source_variables:
temporal_scope:
availability:
interpretability:
```

Example:

```yaml
name: water_deficit_14d

formula: precipitation_14d - et0_14d

hypothesis:
  Atmospheric water demand exceeding precipitation may indicate
  increasing water stress.

expected_relationship:
  positive

source_variables:
  - prec
  - et0

temporal_scope:
  14 days

availability:
  available at prediction time

interpretability:
  high
```

If the agent cannot explain why a feature should contain useful information, it should not create it as a primary feature.

---

# 4. Temporal Feature Engineering

Agricultural processes are dynamic.

A single daily observation is often less informative than the accumulated or recent conditions surrounding it.

The agent should consider:

### Rolling statistics

Examples:

```text
prec_7d
prec_14d
prec_30d

tavg_7d
tavg_14d

vpd_mean_14d
vpd_max_7d

soil_moisture_min_14d
```

### Accumulations

Examples:

```text
cumulative_precipitation
cumulative_radiation
cumulative_gdd
```

### Extremes

Examples:

```text
max_vpd_7d
max_tmax_14d
min_soil_moisture_14d
```

### Variability

Examples:

```text
temperature_std_14d
precipitation_variability_30d
```

Do not generate arbitrary windows without justification.

A 7-day, 14-day, or 30-day window should have a reason to exist, either:

- agricultural interpretation,
- known temporal response,
- domain knowledge,
- or demonstrated predictive value through proper validation.

---

# 5. Respect Crop Growth Stages

Features should preferably be aligned with the crop's development cycle.

The same weather event can have different consequences depending on when it occurs.

Therefore, consider features such as:

```text
days_since_planting
days_since_emergence
days_to_harvest
```

when such information is available.

Prefer stage-aware features such as:

```text
prec_vegetative
vpd_reproductive
radiation_reproductive
soil_moisture_early_stage
heat_stress_flowering
```

over treating the entire season as a homogeneous period.

When crop calendar information is available, use it to define meaningful temporal periods.

---

# 6. Stress Features

For agricultural forecasting, cumulative exposure to stressful conditions can be more informative than simple averages.

Consider features representing:

### Heat stress

```text
days_tmax_above_threshold
cumulative_heat_exposure
max_tmax
```

### Water stress

```text
water_deficit
days_water_deficit
cumulative_water_deficit
```

### Atmospheric stress

```text
mean_vpd
max_vpd
days_vpd_above_threshold
```

### Soil-water stress

```text
minimum_soil_moisture
days_low_soil_moisture
soil_moisture_anomaly
```

Thresholds must have a justification.

Do not arbitrarily choose thresholds only because they improve model performance.

---

# 7. Interaction Features

Interactions should represent plausible agricultural relationships.

Good examples:

```text
prec / et0
prec - et0
vpd * soil_moisture
temperature * soil_moisture
radiation * vegetation_index
```

However, the existence of a mathematical relationship is not sufficient.

For every interaction, ask:

> What physical or biological mechanism does this interaction represent?

For example:

```text
vpd × soil_moisture
```

can represent the interaction between atmospheric water demand and available soil water.

An arbitrary interaction such as:

```text
tmin × adm_id
```

should generally be rejected unless there is a specific reason.

---

# 8. Feature Availability and Leakage

This is a mandatory validation step.

A feature MUST NOT use information that would be unavailable at the moment of prediction.

Potential leakage sources include:

```text
future weather observations
post-harvest NDVI
final-season averages
future soil moisture
yield-derived statistics
harvest information
```

Example:

If yield is predicted on June 1, the feature:

```text
mean_ndvi_entire_season
```

is invalid if it includes observations after June 1.

Instead:

```text
mean_ndvi_until_prediction_date
```

may be valid.

The agent must always ask:

> "Would this value actually be known at prediction time?"

If the answer is no, reject the feature.

---

# 9. Avoid Target Leakage

Never construct features directly or indirectly from the target.

Invalid examples:

```text
yield_mean_by_region
yield_rank
yield_deviation
yield_normalized
```

if they are calculated using the target observations in a way that exposes information from the validation/test period.

Historical target statistics may only be used if they are constructed strictly from information available before the prediction point.

---

# 10. Redundancy and Information Gain

A new feature should add information rather than simply duplicate existing variables.

Before keeping a feature, evaluate:

- correlation with existing features
- mutual information where appropriate
- model-based importance
- incremental validation performance
- conceptual uniqueness

Example:

If the dataset already contains:

```text
tmin
tmax
tavg
```

then:

```text
temperature_range = tmax - tmin
```

may be useful because it represents diurnal temperature variation.

However:

```text
temperature_sum = tmin + tmax
```

may provide little new information.

Do not automatically keep both.

---

# 11. Prefer Interpretability

When two features provide similar predictive performance, prefer the feature that is easier to interpret.

Prefer:

```text
water_deficit_14d
```

over:

```text
PC_7
```

if both provide comparable predictive value.

The final model should ideally allow the agent to explain:

> "Yield decreased because the crop experienced persistent water deficit during a critical growth period."

rather than:

> "Feature PC_7 increased."

Interpretability is especially important because the final AI agent is intended to communicate with non-technical stakeholders.

---

# 12. Control Feature Explosion

Do not automatically generate every possible combination of:

```text
variable × window × aggregation × transformation
```

This can produce thousands of features without meaningful justification.

Feature generation should follow:

```text
Agricultural hypothesis
        ↓
Candidate feature
        ↓
Validation
        ↓
Keep / Reject
```

not:

```text
All possible transformations
        ↓
Train model
        ↓
Keep whatever improves score
```

Prefer a small number of well-motivated features over a large number of unexplained features.

---

# 13. Validation Strategy

Feature evaluation must respect the temporal structure of agricultural data.

Do not randomly split observations from the same growing season into train and test when this creates temporal leakage.

Prefer time-aware validation such as:

```text
Past seasons → Training
Later season → Validation
Most recent season → Test
```

or another split justified by the forecasting problem.

Feature engineering decisions must be made without using the final test set.

The test set is for final evaluation, not feature selection.

---

# 14. Baseline Before Feature Engineering

Always establish a baseline model before adding engineered features.

Example:

```text
Baseline:
raw meteorological variables

Experiment:
raw variables + temporal aggregations

Experiment:
raw variables + stress features

Experiment:
raw variables + vegetation features

Experiment:
raw variables + all selected features
```

This allows the agent to determine whether feature engineering actually provides incremental value.

---

# 15. Feature Evaluation

A feature should be evaluated from multiple perspectives.

### Scientific validity

Does it represent a plausible agricultural phenomenon?

### Temporal validity

Could it have been calculated at prediction time?

### Predictive value

Does it improve out-of-sample performance?

### Incremental value

Does it add information beyond existing features?

### Stability

Does its usefulness persist across different temporal splits or regions?

### Interpretability

Can its effect be explained to a non-technical stakeholder?

A feature that performs well in one split but fails across other periods should be treated cautiously.

---

# 16. Feature Registry

Every accepted feature should be registered.

Example:

```yaml
feature:
  name: cumulative_water_deficit_14d

  formula: sum(prec - et0, 14 days)

  hypothesis:
    Persistent atmospheric water demand exceeding precipitation
    may increase crop water stress.

  expected_effect:
    negative

  inputs:
    - prec
    - et0

  temporal_window:
    14 days

  category:
    water_stress

  interpretability:
    high

  leakage_risk:
    low

  validation:
    status: accepted
```

The registry should make it possible to understand why every engineered feature exists.

---

# 17. Reject Features When

Reject or flag a feature when:

- there is no meaningful hypothesis;
- it uses future information;
- it leaks the target;
- it duplicates existing information without adding value;
- its threshold is arbitrary and unsupported;
- its predictive improvement exists only on the test set;
- it is unstable across temporal validation;
- it dramatically increases dimensionality without meaningful benefit;
- its interpretation is impossible and an equally predictive interpretable alternative exists.

---

# 18. Feature Engineering Decision Checklist

Before accepting a feature, answer:

```text
[ ] What agricultural phenomenon does this feature represent?

[ ] Why should that phenomenon affect yield?

[ ] What is the expected relationship with yield?

[ ] Which raw variables are required?

[ ] Is the feature available at prediction time?

[ ] Does it contain future information?

[ ] Does it leak the target?

[ ] Does it add information beyond existing features?

[ ] Is the temporal window justified?

[ ] Is the feature interpretable?

[ ] Does it improve out-of-sample performance?

[ ] Is the improvement stable across temporal splits?

[ ] Can the feature's contribution be explained to a non-technical user?
```

If the agent cannot answer these questions, the feature should not automatically be accepted.

---

# 19. Guiding Principle for the Agent

The agent should follow this reasoning:

```text
Do not ask:

"Can I create another feature?"

Ask:

"What information about the agricultural system is currently missing?"

Then:

"What observable data could represent that information?"

Then:

"How can I encode that information without introducing leakage?"

Finally:

"Does the resulting feature add reliable information about yield?"
```

Feature engineering should therefore be treated as **modeling the agricultural system**, not merely transforming columns.
