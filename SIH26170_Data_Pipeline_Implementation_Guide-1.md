# SIH26170 — FINAL Generalized Data Pipeline Implementation Guide

> **Team rule:** This document is the single implementation source of truth. Read Section 3 first. It explains the system hierarchy before the detailed sections. Every important technical term is explained in simple language before implementation rules.

# Contents

3. Overall Structural Hierarchy — Understand This First
4. End-to-End Pipeline
5. Pipeline Responsibilities
6. Dataset Compatibility Contract
7. Dataset Intake
8. Automatic Dataset Profiling
9. Compatibility Assessment
10. Context Identification
11. Schema and Semantic Mapping
12. Unit Mapping and Conversion
13. Required vs Optional Information
14. Data-Quality Validation
15. Identity Integrity
16. Time Handling
17. Test and Stress Context
18. Confounder Handling
19. Reference Population and Grouping Engine
20. Canonical Internal Representation
21. Separate Model Inputs from Metadata
22. Feature / Behaviour Engineering
23. Trajectory Pattern Coverage
24. Isolation Forest Branch
25. GPR Forecast Branch
26. Model-Specific Branch Preparation
27. Evaluation Strategy
28. Current Data Rule for This Project
29. Authentic-Dataset Methodology
30. Limits, Thresholds and Rules
31. Risk Fusion
32. Decision Layer
33. Explanation Layer
34. GPR Explanation
35. Physical-Cause Context
36. Progressive / Streaming State
37. API Contract
38. Audit and Traceability
39. Database-Oriented Storage Model
40. Dashboard Contract
41. Recommended Repository Structure
42. Implementation Order
43. Team Workstream Mapping
44. Testing Strategy
45. No-Leakage Rule
46. Data Drift and Deployment Monitoring
47. Security and Access Boundaries
48. Unknowns That Must Remain Explicit
49. Future QA Question–Answer Layer
50. Definition of Done
51. Final Implementation Principle
52. Quick Glossary for Team Members
53. Source Mapping Used for This Guide
54. One-Sentence Architecture Summary

---


**Problem Statement:** SIH26170 — AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Scope:** Burn-in screening data only  
**Primary model branches:** Isolation Forest for current/trajectory anomaly detection + Gaussian Process Regression (GPR) for drift forecasting and predictive uncertainty  
**Design principle:** Configuration-driven, time-aware, context-aware, dataset-independent at the pipeline level, and human-supervised at the decision layer.

---

## Purpose

This guide defines the complete implementation of the data pipeline used by the SIH26170 solution.

The pipeline converts heterogeneous, authentic burn-in/screening datasets into one controlled internal representation and then feeds two model branches:

1. **Anomaly branch:** determine whether a component's observed behaviour is unusual relative to the appropriate comparison population and its own history.
2. **Forecast branch:** use available history to estimate future behaviour and quantify predictive uncertainty.
3. **Decision branch:** combine anomaly evidence, forecast evidence, engineering rules, data quality, and context into a conservative recommendation.
4. **Explanation branch:** produce simple-English, evidence-backed reasons that a QA/engineering user can inspect.
5. **Audit branch:** preserve the complete decision chain so a result can be reconstructed later.

The design does **not** assume a fixed source file format, fixed column names, fixed units, fixed parameter count, or one universal specification limit.

> **Important:** The pipeline is generalized, but it is not “any file in, guaranteed result out.” A new dataset must satisfy the compatibility requirements defined in Section 6 and must have enough documented context and history for the requested model branch.

---

## Source Basis and Design Constraints

This guide is based on the project's final problem decomposition, Part 3 data design, Part 7 confounder design, Part 10 deployment design, the project's research guide, and the agreed final solution architecture.

Key source-derived constraints:

- The project is scoped to **burn-in**, not all environmental stress screening.
- The real ISRO schema, exact parameter list, exact units, ingestion mechanism, engineering limits, and production QA policy are currently unknown; they must not be invented.
- Burn-in data may arrive progressively at checkpoints rather than as one completed table.
- The system must preserve component, part, lot, time, units, test conditions, and source context when available.
- Data quality must be checked before model execution.
- Component-level and lot-level evidence are different questions and should remain distinguishable.
- Real defects are expected to be rare, so supervised learning from many confirmed defect labels is not assumed.
- Isolation Forest is an unsupervised anomaly-detection candidate and is the selected anomaly branch for the current solution.
- GPR is used for forecasting with predictive uncertainty; uncertainty must be represented explicitly rather than invented as a generic confidence score.
- Thresholds, anomaly cut-offs, engineering rules, and dispositions are configurable/approved decision inputs, not facts automatically learned from a proxy dataset.
- AI provides recommendations; final operational disposition remains with the responsible QA/engineering process.
- Every decision must carry a defensible reason and be traceable to its data, configuration, model, rules, and human action.
- The project uses **authentic real datasets** for methodology development and evaluation. Synthetic datasets are not part of the current methodology.

---

# 3. Overall Structural Hierarchy — Understand This First

```text
AUTHENTIC BURN-IN / SCREENING DATA
│
├─ 1. UNDERSTAND THE DATA
│    ├─ Source/format inspection
│    ├─ Automatic profiling
│    └─ Compatibility check
│
├─ 2. UNDERSTAND WHAT EACH FIELD MEANS
│    ├─ Component / part / lot identity
│    ├─ Measurement / parameter meaning
│    ├─ Time / checkpoint meaning
│    ├─ Test / stress context
│    └─ Schema + semantic + unit mapping
│
├─ 3. MAKE DATA TRUSTWORTHY
│    ├─ Required / optional checks
│    ├─ Data-quality validation
│    ├─ Identity validation
│    ├─ Time validation
│    └─ Confounder checks
│
├─ 4. CREATE ONE COMMON INTERNAL FORMAT
│    └─ Canonical Internal Representation
│
├─ 5. DECIDE WHO IS COMPARABLE
│    └─ Reference Population / Grouping Engine
│         ├─ policy
│         ├─ selector
│         ├─ validator
│         ├─ safe fallback
│         └─ time-aware / leakage-safe selection
│
├─ 6. REPRESENT COMPONENT BEHAVIOUR
│    ├─ current values
│    ├─ own-history features
│    ├─ peer-relative features
│    ├─ lot-relative features
│    └─ trajectory / drift features
│
├─ 7. AI BRANCHES
│    ├─ Isolation Forest → unusual behaviour now
│    └─ GPR → future behaviour + predictive uncertainty
│
├─ 8. EVIDENCE + EXPLANATION
│    ├─ anomaly evidence
│    ├─ forecast evidence
│    ├─ limit/rule evidence
│    ├─ reference-population evidence
│    └─ confounder/data-quality evidence
│
├─ 9. OPERATIONAL RECOMMENDATION
│    └─ PASS / RETEST / REVIEW / REJECT
│
├─ 10. HUMAN QA / ENGINEERING
│    └─ final authorized disposition
│
└─ 11. AUDIT + API + DASHBOARD
```

### What this means in one sentence
The system first understands and validates the data, then determines the correct comparison population, then computes behaviour features, then runs the AI models, then turns their outputs into evidence and a controlled recommendation, and finally records the human QA action.

### Responsibility boundary
The **model** is not responsible for knowing raw column meanings, choosing engineering limits, deciding who is comparable, or making the final QA decision. Those are surrounding system responsibilities.

---

# 4. End-to-End Pipeline

The final generalized pipeline is:

```text
AUTHENTIC DATASET INTAKE
        │
        ▼
SOURCE / FORMAT INSPECTION
        │
        ▼
DATASET PROFILING
        │
        ▼
COMPATIBILITY CHECK
        │
        ├── incompatible ──► STOP + DATASET REJECTION REPORT
        │
        ▼
CONTEXT IDENTIFICATION
(part/component/lot/test/run/time/source metadata)
        │
        ▼
SCHEMA + SEMANTIC MAPPING
(fields → internal concepts)
        │
        ▼
UNIT MAPPING + CONVERSION
        │
        ▼
REQUIRED / OPTIONAL FIELD CHECK
        │
        ▼
DATA-QUALITY VALIDATION
(missing / duplicate / ID / timestamp / unit / saturation / noise /
 communication issues / ordering / clock mismatch / physical validity)
        │
        ├── critical identity/context failure ──► QUARANTINE / STOP FOR AFFECTED DATA
        │
        ▼
PREPROCESSING
        │
        ▼
PART / COMPONENT / LOT / REFERENCE-POPULATION GROUPING
        │
        ▼
TIME NORMALIZATION + CHECKPOINT ALIGNMENT
        │
        ▼
FEATURE / BEHAVIOUR REPRESENTATION
        │
        ├───────────────────────────┐
        ▼                           ▼
ISOLATION FOREST BRANCH         GPR FORECAST BRANCH
current + history anomaly      future trend + uncertainty
        │                           │
        └──────────────┬────────────┘
                       ▼
             MODEL + RULE EVALUATION
                       │
                       ▼
              EVIDENCE GENERATION
                       │
                       ▼
              EXPLANATION LAYER
                       │
                       ▼
               RISK / DECISION
         PASS / RETEST / REVIEW / REJECT
                       │
                       ▼
                 HUMAN QA
                       │
                       ▼
             FINAL HUMAN ACTION
                       │
                       ▼
                 AUDIT RECORD
                       │
                       ▼
                 API / DASHBOARD
```

### 3.1 Core architectural rule

The **common data layer is built once**. The model-specific branches are separate.

```text
Dataset
  ↓
Profiling
  ↓
Mapping + Units + Context
  ↓
Validation
  ↓
Canonical Internal Representation
  ↓
Feature / Behaviour Representation
  ├───────────────► IF branch
  └───────────────► GPR branch
```

This prevents duplicate data-cleaning logic and keeps the pipeline portable across datasets.

---

# 5. Pipeline Responsibilities

Each stage has one clear responsibility.

| Stage | Main purpose | Human-configured? | Software-automated? |
|---|---|---:|---:|
| Dataset intake | Register and safely ingest source data | Yes | Yes |
| Profiling | Inspect structure, fields, types, ranges, cardinality | No | Yes |
| Compatibility check | Decide whether dataset supports intended task | Yes for approval rules | Yes |
| Context identification | Establish semantic meaning of identifiers and conditions | Yes | Partly |
| Schema mapping | Map source fields to canonical concepts | Yes | Yes after config exists |
| Unit mapping | Normalize units and conversions | Yes when ambiguous | Yes when configured |
| Required/optional check | Enforce minimum contract | Yes | Yes |
| Data-quality validation | Detect invalid/unreliable records | Yes for thresholds/rules | Yes |
| Preprocessing | Clean/transform valid measurements | Yes for policy | Yes |
| Grouping | Build component/lot/reference populations | Yes | Yes |
| Time handling | Order and align observations | Yes for time policy | Yes |
| Feature engineering | Create behaviour representations | Yes | Yes |
| IF branch | Current/trajectory anomaly score | Model config | Yes |
| GPR branch | Forecast + predictive uncertainty | Model config | Yes |
| Evaluation | Measure model performance | Yes for acceptance criteria | Yes |
| Evidence generation | Gather numerical/graphical support | Yes for evidence policy | Yes |
| Decision | Apply approved rules | Yes | Yes |
| QA action | Final human disposition | Yes | No |
| Audit | Persist decision chain | Yes for retention/access policy | Yes |

**Principle:** Software should automate repeatable computation; humans must define semantics, engineering context, and approval rules that cannot safely be inferred.

---

# 6. Dataset Compatibility Contract

A dataset is eligible for the common pipeline only when the required information is available or can be explicitly mapped.

## 5.1 Minimum compatibility requirements

The dataset should provide:

1. **Multiple components/devices or equivalent repeated entities**, so population-relative analysis is meaningful.
2. **Repeated measurements over time** for the intended forecasting use.
3. A measurable numeric target or parameter that can represent behaviour.
4. A trustworthy link between each measurement and its component/device identity.
5. A trustworthy time field, checkpoint, interval, or other ordered temporal representation.
6. Documented or resolvable units.
7. Enough history to support the intended forecast horizon.
8. Sufficient population diversity to create a meaningful normal/reference population.
9. Enough metadata to separate incompatible populations when required.
10. A reproducible train/validation/test strategy appropriate to the dataset.

## 5.2 Dataset may be rejected or partially supported when

- component identity is missing or ambiguous;
- temporal ordering cannot be reconstructed;
- units cannot be resolved safely;
- required measurement meaning is unknown;
- all observations come from one component with no reference population;
- there are too few time points for the requested forecast;
- critical context needed to interpret the signal is absent;
- the dataset is incompatible with the requested branch.

## 5.3 Compatibility is branch-specific

A dataset may be:

- **IF-compatible but not GPR-compatible**;
- **GPR-compatible but not suitable for population-relative anomaly detection**;
- **compatible with both**.

Do not force one dataset to serve both branches merely for architectural convenience.

---

# 7. Dataset Intake

## 6.1 Supported source concept

The intake layer should be able to receive sources such as:

- CSV;
- Excel;
- MATLAB/MAT;
- JSON;
- database export;
- equipment/file export;
- future API/stream input.

The real ISRO interface is currently unknown, so the production adapter must remain replaceable.

## 6.2 Ingestion modes

The same controlled ingestion contract should support:

```text
Manual source
(operator-entered file)
        │
        ├─────────────┐
        │             │
        ▼             ▼
source-aware validation
        │
        └──────► common mapping path

Automatic source
(equipment file / export / API)
        │
        ▼
source-aware validation
        │
        └──────► common mapping path
```

The system must not claim that ISRO currently uses either manual or automatic entry unless verified.

## 6.3 Intake metadata

Create a dataset registration record:

```yaml
dataset_id: unique-dataset-id
source_type: csv | excel | mat | json | api | other
source_name: human-readable-name
source_version: source-version-or-hash
received_at: ISO-8601 timestamp
provided_by: source-owner-if-known
schema_version: mapping-schema-version
mapping_version: mapping-config-version
unit_policy_version: unit-config-version
intended_tasks:
  - anomaly
  - forecasting
  - both
status: registered | profiling | approved | rejected
notes: free-text
```

Do not silently overwrite earlier dataset registrations.

---

# 8. Automatic Dataset Profiling

Profiling happens before semantic transformation.

## 7.1 What the profiler inspects

The profiler should report:

- file/source type;
- dimensions;
- table/sheet/array structure;
- column/field names;
- detected data types;
- missingness;
- unique counts;
- candidate identifier fields;
- candidate time fields;
- candidate numeric measurement fields;
- text/category fields;
- observed numeric ranges;
- duplicate records;
- timestamp ordering;
- repeated-entity counts;
- repeated-time counts;
- candidate grouping fields;
- candidate unit fields;
- suspicious constant columns;
- extreme values;
- saturation-like patterns;
- incompatible mixed data types.

## 7.2 What profiling must NOT do

Profiling does not automatically prove that:

- a field is a component ID;
- a field is a lot ID;
- a numeric field is leakage current;
- an observed range is an engineering limit;
- a prefix in an ID represents a semantic part type.

Those meanings require approved mapping/configuration.

## 7.3 Profiling output example

```yaml
profile:
  records: ...
  fields:
    candidate_id_fields: [...]
    candidate_time_fields: [...]
    candidate_numeric_fields: [...]
    candidate_context_fields: [...]
  missingness:
    by_field: {...}
  duplicates:
    exact_rows: ...
    candidate_measurement_duplicates: ...
  temporal:
    ordered: true|false
    unique_timestamps: ...
    irregular_intervals: true|false
  entities:
    unique_components: ...
    unique_lots: ...
  warnings:
    - ...
```

---

# 9. Compatibility Assessment

Compatibility is a controlled gate between profiling and model use.

## 8.1 Decision outputs

The checker returns one of:

- `COMPATIBLE`
- `PARTIALLY_COMPATIBLE`
- `INCOMPATIBLE`

It also returns a reason list.

Example:

```json
{
  "status": "PARTIALLY_COMPATIBLE",
  "anomaly_branch": "compatible",
  "forecast_branch": "incompatible",
  "reasons": [
    "repeated component measurements exist",
    "time ordering exists",
    "forecast horizon cannot be validated because later checkpoints are unavailable"
  ]
}
```

## 8.2 Never hide incompatibility

If required information is missing, the pipeline must say so.

Do not fabricate:

- component IDs;
- lot IDs;
- units;
- timestamps;
- limits;
- defect labels;
- physical causes.

A model should receive `unknown`/`missing` metadata or be stopped, not be fed invented semantics.

---

# 10. Context Identification

Context answers:

> “What is this measurement, which entity does it belong to, under what population and test conditions was it produced, and what other information is needed to interpret it?”

## 9.1 Canonical context concepts

The canonical model should be able to represent, where available:

- `component_id`
- `part_number`
- `part_type`
- `lot_id`
- `batch_id`
- `test_run_id`
- `measurement_id`
- `parameter_id`
- `parameter_name`
- `timestamp`
- `checkpoint`
- `unit`
- `test_condition`
- `temperature_context`
- `stress_context`
- `instrument_id`
- `calibration_reference`
- `source_id`
- `data_quality_status`

Not every field is mandatory for every dataset.

## 9.2 Context hierarchy

Conceptually:

```text
Part Type
   └── Part Number / Family
        └── Lot / Batch
             └── Component / Device
                  └── Test Run
                       └── Measurement
                            └── Parameter × Time
```

This hierarchy supports component-level and lot-level evidence without confusing one with the other.

---

# 11. Schema and Semantic Mapping

## 10.1 Why mapping exists

Different datasets may use different:

- column names;
- file formats;
- table layouts;
- parameter counts;
- identifier structures;
- unit conventions;
- timestamp representations.

Therefore the core ML code must not depend on source-specific names.

## 10.2 Mapping configuration

Use a versioned mapping file.

```yaml
mapping_version: "1.0"

fields:

  - source_field: "<source field name>"
    internal_concept: component_id
    data_type: string
    required: true
    nullable: false

  - source_field: "<source field name>"
    internal_concept: timestamp
    data_type: datetime
    required: true
    nullable: false

  - source_field: "<source field name>"
    internal_concept: checkpoint
    data_type: numeric_or_string
    required: false
    nullable: true

  - source_field: "<source field name>"
    internal_concept: parameter_value
    parameter_key: "<parameter identifier>"
    data_type: numeric
    required: true
    nullable: false

  - source_field: "<source field name>"
    internal_concept: lot_id
    data_type: string
    required: false
    nullable: true
```

## 10.3 Mapping rules

A mapping entry should define, where relevant:

- source field;
- canonical concept;
- parameter key;
- data type;
- unit;
- conversion;
- required/optional;
- validation rule;
- description;
- source notes.

## 10.4 Long vs wide datasets

The adapters must convert either structure into the same canonical representation.

### Wide source

```text
component | time | parameter_A | parameter_B | parameter_C
```

### Long source

```text
component | time | parameter_name | value | unit
```

### Canonical internal form

```text
component_id
lot_id
part_context
test_context
parameter_key
time
checkpoint
value
unit_canonical
quality_status
source_reference
```

The rest of the pipeline operates on the canonical representation rather than on source layout.

---

# 12. Unit Mapping and Conversion

Units are part of data meaning.

## 11.1 Unit contract

For each numeric parameter:

```yaml
parameter_key: "<canonical parameter>"
source_unit: "<source unit>"
canonical_unit: "<approved canonical unit>"
conversion:
  type: identity | scale | affine | custom_approved
  rule: "<explicit conversion rule>"
validation:
  plausible_min: null
  plausible_max: null
```

Do not invent a canonical unit when the source documentation does not establish one.

## 11.2 Unit validation

Reject or quarantine records when:

- unit is missing for a parameter that requires it and cannot be resolved;
- mixed units appear in the same series without explicit conversion;
- a conversion is ambiguous;
- the converted values violate the approved physical/data rule.

A failed unit check must never be silently “fixed” by guessing.

---

# 13. Required vs Optional Information

## 12.1 Three information states

Every canonical field should be represented as:

1. **Present and trusted**
2. **Missing**
3. **Present but uncertain/unverified**

Do not reduce all three states to a single null value when the distinction affects safety.

## 12.2 Required information

A field is required only when the selected downstream task needs it.

Example:

- component identity is required for component-level history;
- a comparison population is required for peer-relative analysis;
- sufficient time information is required for forecasting;
- a lot ID is valuable but may be optional if the dataset genuinely lacks lot structure.

## 12.3 Ambiguous information

When semantic meaning is ambiguous:

```text
Source field → AMBIGUOUS
                 │
                 ├── can be resolved by approved mapping → map
                 │
                 └── cannot be resolved → keep unknown / stop affected task
```

Never infer a critical field purely from a naming pattern without an approved rule.

---

# 14. Data-Quality Validation

Validation is a mandatory gate before model execution.

## 13.1 Validation cases

The validator must be able to detect and classify:

1. Missing reading.
2. Partial missingness.
3. Measurement/sensor error.
4. Duplicate reading.
5. Wrong timestamp.
6. Wrong component identity.
7. Wrong unit.
8. Measurement saturation.
9. Sensor noise/quantization.
10. Communication/equipment failure.
11. Out-of-order records.
12. Clock mismatch between instruments.

These cases are part of the PS decomposition and must remain visible in the implementation.

## 13.2 Validation severity

Use at least:

```text
INFO
WARNING
QUARANTINE
BLOCK
```

Example rules:

| Failure | Default handling |
|---|---|
| Optional field missing | WARNING |
| One non-critical parameter missing | QUARANTINE that parameter/record |
| Duplicate exact reading | mark duplicate; retain provenance |
| Wrong timestamp suspicion | QUARANTINE affected record |
| Wrong component ID suspicion | BLOCK affected identity chain |
| Unit ambiguity | BLOCK affected parameter |
| Saturation | WARNING or QUARANTINE depending on semantics |
| Noise/quantization | usually transform/robust feature handling |
| Corrupted/incomplete block | QUARANTINE |
| Out-of-order record | reorder only if time semantics remain trustworthy |
| Clock mismatch | correct only with verified reference; otherwise flag |

## 13.3 Do not delete silently

Every removed/quarantined record should preserve:

- original row/record reference;
- reason;
- validation rule ID;
- timestamp;
- pipeline run ID.

---

# 15. Identity Integrity

Wrong component identity is particularly dangerous because the record can look numerically valid while contaminating an entire trajectory.

## 14.1 Identity checks

Use available evidence such as:

- unique-record constraints;
- source IDs;
- run IDs;
- continuity;
- known test-run relationships;
- duplicate identity/time combinations;
- approved equipment metadata;
- cross-file consistency.

## 14.2 Identity rule

If identity cannot be trusted:

> Do not place the observation into a component trajectory used by the model.

Move it to quarantine or an unresolved pool.

---

# 16. Time Handling

Time is a first-class field.

## 15.1 Preserve both forms when available

Keep:

- original timestamp;
- normalized timestamp;
- relative elapsed time;
- checkpoint label/value, when present.

Example:

```text
original_time
normalized_time
elapsed_time
checkpoint
```

## 15.2 Sparse checkpoints

The pipeline must support sparse and unevenly spaced measurements.

Do not assume:

```text
0h → 24h → 48h → 72h → ...
```

Instead:

```text
t0 → t1 → t2 → t3
```

where actual elapsed time is retained.

## 15.3 Progressive arrival

The system must operate on partial trajectories.

Conceptual state:

```text
State after first checkpoint
        ↓ update
State after second checkpoint
        ↓ update
State after third checkpoint
        ↓ update
State after later checkpoint
```

Each state must use **only data that had actually arrived by that point**.

No future observations may leak into an earlier decision.

## 15.4 Out-of-order arrival

Data can physically arrive out of order while still having valid measurement time.

The system should:

1. preserve arrival timestamp;
2. sort by measurement time for trajectory construction;
3. verify that sorting does not hide a clock/data-integrity problem;
4. keep original order as audit metadata.

---

# 17. Test and Stress Context

Observed behaviour can be affected by the environment in which the component was tested.

Capture, where available:

- temperature;
- voltage;
- current/bias;
- frequency;
- duty cycle;
- dynamic/static state;
- chamber identifier;
- instrument identity;
- fixture/socket identity;
- calibration/reference status;
- test/run metadata.

Not every dataset contains all of these fields.

## 16.1 Context-aware interpretation

The model must avoid automatically labeling a component as defective when a pattern can reasonably be explained by:

- normal lot variation;
- chamber variation;
- instrument/calibration drift;
- fixture/contact problem;
- common-mode disturbance;
- normal early settling;
- low signal-to-noise conditions.

---

# 18. Confounder Handling

A measured drift is not automatically a component defect.

The pipeline therefore maintains a **confounder evidence channel**.

```text
Measurement behaviour
       │
       ▼
Is the change component-specific?
       │
   ┌───┴────┐
   │        │
  yes       no / uncertain
   │        │
   ▼        ▼
model      inspect common-mode /
evidence   test-condition evidence
```

## 17.1 Common-mode detection

If many comparable components show the same unusual change during the same test window, record:

```text
common_mode_signal = true
```

This should reduce confidence in a component-specific interpretation and can route the case to `REVIEW` or `RETEST` depending on data-quality evidence.

## 17.2 Early settling

A change during the first interval may be a normal stabilization effect.

Therefore feature engineering should allow:

- early-change feature;
- post-settling trajectory feature;
- sustained drift feature.

Do not automatically treat the first interval as a defect.

---

# 19. Reference Population and Grouping Engine

## 18.1 Why this is a separate layer
The model cannot safely infer from a component ID that another component is a valid peer. The system must explicitly determine the **reference population** before calculating peer-relative behaviour.

```text
Canonical context
      ↓
Reference Population Policy
      ↓
Population Selector
      ↓
Population Validator
      ↓
Comparable population
      ↓
Peer-relative features
      ↓
Isolation Forest / other model
```

## 18.2 Simple meaning of every term
**Reference population:** the components treated as valid peers for one analysis.

**Policy:** configurable rules describing what makes two components comparable.

**Selector:** code that finds the components that satisfy the policy.

**Validator:** code that checks whether the selected population is usable.

**Fallback:** the next approved population level when the preferred population cannot be used.

**Time filtering:** only use peer information allowed at the analysis time, especially for streaming and honest evaluation.

**Target exclusion:** do not let the component being scored influence its own peer baseline.

## 18.3 What can define comparability
Depending on the target environment, approved grouping keys may include:

```text
Part Type
Part Number
Lot / Batch
Test Run / Procedure
Compatible test conditions
Relevant instrument / setup context
Dataset / process context
```

The exact keys are **configuration**, not hard-coded universal truth.

## 18.4 Example hierarchy

```text
Level 1: same part type + part number + compatible context + same lot
             ↓ if unusable
Level 2: same part type + part number + compatible context across comparable lots
             ↓ if unusable
Level 3: another explicitly approved compatible population
             ↓
No valid population → controlled non-model outcome / REVIEW
```

These levels are examples. Do not present them as actual ISRO policy until validated.

## 18.5 Component-level vs lot-level analysis

```text
COMPONENT LEVEL
One component vs its comparable peers

LOT LEVEL
One lot vs comparable lots
```

Keep both evidence types separate. A whole lot can shift together while individual components remain similar to one another.

## 18.6 Different device families
Grouping prevents invalid comparisons such as a digital IC being treated as a peer of a power device when the configuration says they are incompatible.

This **does not automatically require one Isolation Forest per device family**. Use one common pipeline. Test whether the same model/configuration works across validated populations; register separate compatible model profiles only where evaluation shows they are necessary.

## 18.7 D1 and D2
D1 and D2 are different semiconductor datasets/process structures, not automatically different device-family labels. They have different measurement counts and different process-step structures. Therefore:

```text
D1 ─┐
    ├─ same generalized pipeline
D2 ─┘
```

but they should not be blindly pooled into one reference population merely because both are semiconductor datasets. Their dataset/process context belongs in configuration.

## 18.8 Population validation
Before a population is used, check:

```text
Enough distinct comparable components?
Compatible parameters available?
Compatible test context?
Usable history?
Acceptable data quality?
Target excluded?
No future information leakage?
```

The minimum population size is a configurable engineering setting; the guide intentionally does not invent a universal number.

## 18.9 Population identity and audit
Store:

```text
population_id
grouping_policy_version
selection_rule_id
reference size
fallback level used
analysis time
validation status
reference members or reproducible query reference
```

## 18.10 Pseudocode

```python
def select_reference(component, observations, policy, as_of_time):
    candidates = filter_context_compatible(observations, component, policy)
    candidates = filter_not_target(candidates, component.id)
    candidates = filter_time_available(candidates, as_of_time, policy)

    for level in policy.levels:
        group = apply_grouping(candidates, level.keys)
        if population_is_valid(group, level):
            return ReferencePopulation(
                population_id=build_population_id(level, component, policy),
                rule_id=level.name,
                members=group
            )

    return ReferenceUnavailable(reason="No approved comparable population")
```

## 18.11 Suggested code modules

```text
src/grouping/
├── population_policy.py
├── population_selector.py
├── population_validator.py
├── context_compatibility.py
├── population_features.py
└── population_registry.py
```

## 18.12 Final responsibility split

```text
Mapping             → what does this source field mean?
Context resolver    → what is this component's context?
Population policy   → what makes components comparable?
Population selector → which peers satisfy the policy?
Population validator→ is the peer group usable?
Features            → how different is this component from peers?
Isolation Forest    → does the behaviour look unusual?
GPR                 → where is it heading?
Evidence            → what facts support the result?
Decision            → what recommendation follows?
Human QA            → what final action is authorized?
```

# 20. Canonical Internal Representation

The common internal schema is the contract between data engineering and model branches.

## 19.1 Recommended structure

```text
Observation
├── identity
│   ├── component_id
│   ├── part_number
│   ├── part_type
│   ├── lot_id
│   ├── batch_id
│   └── test_run_id
├── measurement
│   ├── parameter_key
│   ├── value
│   ├── canonical_unit
│   ├── timestamp
│   ├── elapsed_time
│   └── checkpoint
├── test_context
│   ├── stress variables
│   ├── chamber context
│   ├── instrument context
│   └── fixture/contact context
├── quality
│   ├── quality_status
│   ├── validation_flags
│   └── provenance
└── configuration
    ├── mapping_version
    ├── unit_policy_version
    └── validation_version
```

## 19.2 Important rule

The canonical representation contains both:

- **values used for computation**, and
- **metadata needed to interpret/explain/audit those values**.

Not all canonical fields are model features.

---

# 21. Separate Model Inputs from Metadata

This separation is mandatory.

## 20.1 Model input categories

### A. Numerical/behaviour features

May include:

- current observed value;
- normalized value;
- time-relative value;
- change from baseline;
- slope/drift;
- curvature;
- variability;
- peer-relative deviation;
- multi-parameter relationships.

Only features that are actually supported by the dataset and validated preprocessing should be used.

### B. Context/grouping metadata

Used to choose the right population/model/config:

- part type;
- part number;
- lot;
- compatible test context.

These may determine which model/config is used without being passed as arbitrary numeric features.

### C. Explanation evidence

Examples:

- current value;
- relevant engineering limit;
- peer baseline;
- trend/drift;
- forecast;
- uncertainty;
- data-quality flags.

These are retained for explanation even if not direct model features.

### D. Evaluation metadata

Examples:

- known label, if available;
- future observation used as ground truth;
- reference group;
- split assignment.

Evaluation labels must never leak into an unsupervised model feature vector.

### E. Audit metadata

Examples:

- run ID;
- dataset ID/version;
- mapping version;
- model version;
- threshold version;
- reviewer;
- final action.

Audit fields are not model features.

---

# 22. Feature / Behaviour Engineering

Feature engineering transforms valid canonical observations into representations that describe behaviour rather than raw source formatting.

## 21.1 Core behaviour concepts

Where supported, build features for:

- level;
- change;
- rate of change;
- acceleration/deceleration;
- relative-to-baseline deviation;
- peer-relative deviation;
- within-component trajectory;
- cross-parameter relationship;
- stability/variability.

## 21.2 Baseline choices

Possible references include:

- component's first trusted measurement;
- population baseline;
- lot baseline;
- approved reference curve;
- engineering reference.

The selected reference must be stored in configuration and in the evidence output.

## 21.3 Do not over-engineer sparse data

For a short sequence, avoid feature designs that require a long history.

The feature builder should adapt to available history and report which features could not be calculated.

Example:

```json
{
  "features": {
    "level": 1.0,
    "drift": 0.08
  },
  "feature_status": {
    "curvature": "unavailable_insufficient_points"
  }
}
```

---

# 23. Trajectory Pattern Coverage

The pipeline should be able to represent and evaluate at least these behaviour classes from the PS decomposition:

1. Flat/stable.
2. Gross out-of-limit.
3. Slow steady drift.
4. Accelerating drift.
5. Late step change.
6. Early jump then recovery/flattening.
7. Non-monotonic/oscillatory behaviour.
8. Spike and recovery.
9. Sudden drop.
10. Single-parameter drift.
11. Multi-parameter correlated drift.
12. In-spec but peer-relative anomaly.
13. Borderline/ambiguous case.
14. Defect slower than the available observation horizon.
15. Same final value but different path.
16. Component versus peers.
17. Entire lot shifted.

These are behavioural test cases for the algorithm and evaluation plan, not claims that every incoming dataset contains every pattern.

---

# 24. Isolation Forest Branch

## 23.1 Objective

The anomaly branch asks:

> **“Is the currently observed behaviour unusual compared with the appropriate population and available history?”**

Isolation Forest is an unsupervised branch because reliable confirmed-defect labels are expected to be scarce.

## 23.2 Input contract

The IF branch receives:

```text
component context
+ approved reference population
+ validated behaviour features
+ current/available history
```

It must not receive:

- future data unavailable at decision time;
- evaluation labels;
- human final decisions;
- audit-only identifiers as arbitrary features.

## 23.3 Model configuration

Example:

```yaml
model:
  type: isolation_forest
  version: "<version>"
  random_state: 42
  contamination: "<approved/configured value or strategy>"
  n_estimators: "<validated value>"

population:
  grouping_rule: "<approved group definition>"
feature_set:
  version: "<feature config version>"
```

### Contamination rule

Isolation Forest's `contamination` is not a magically learned truth label. It is a configuration/engineering decision that should be justified using available defect knowledge, sensitivity analysis, and operating requirements.

Do not describe it as “the model discovered the defect rate.”

## 23.4 Output

Store at least:

```json
{
  "anomaly_score": 0.0,
  "anomaly_status": "normal|watch|high",
  "model_version": "...",
  "population_id": "...",
  "feature_version": "...",
  "decision_threshold_version": "..."
}
```

The exact score direction/range depends on the implementation and must be documented in code/model metadata rather than assumed.

---

# 25. GPR Forecast Branch

## 24.1 Objective

The forecasting branch asks:

> **“Given the trusted measurements available so far, what future value is predicted, and how uncertain is that prediction?”**

This is separate from current anomaly detection.

## 24.2 Input contract

The GPR branch receives:

```text
validated time
+ target parameter
+ valid history up to decision time
+ approved grouping/reference policy
```

## 24.3 Predictive uncertainty

A Gaussian Process provides a predictive distribution rather than only a point estimate.

Conceptually:

```text
forecast output
├── predictive mean
└── predictive variance / standard deviation
```

The software may display a prediction interval or other uncertainty representation derived from this output.

Do not label this as a generic “confidence score” unless the exact metric and statistical meaning are defined.

## 24.4 Output

```json
{
  "forecast_mean": 0.0,
  "forecast_std": 0.0,
  "forecast_horizon": "...",
  "interval": {
    "lower": 0.0,
    "upper": 0.0
  },
  "model_version": "...",
  "training_scope": "...",
  "validation_mae": 0.0
}
```

If the data are insufficient for a reliable GPR fit, return:

```text
forecast_status = unavailable_insufficient_history
```

rather than fabricating a forecast.

---

# 26. Model-Specific Branch Preparation

The common representation branches only after validation.

```text
Canonical observations
        │
        ▼
Feature / behaviour builder
        │
        ├──────────────► IF feature matrix
        │
        └──────────────► GPR time-value series
```

## 25.1 IF preparation

- select approved behaviour features;
- align to reference population;
- scale/transform only according to model design;
- remove/quarantine invalid rows;
- preserve feature provenance.

## 25.2 GPR preparation

- order observations by trusted time;
- retain actual elapsed time;
- select target parameter;
- remove invalid measurements;
- enforce a no-future-information rule;
- verify sufficient observations;
- train/validate using time-aware splitting.

---

# 27. Evaluation Strategy

Evaluation must mirror actual use.

## 26.1 Anomaly branch metrics

Where trustworthy labels or evaluation truth are available, measure:

- recall/sensitivity;
- false-negative count;
- precision;
- false-positive rate;
- stability across lots/populations;
- alert rate;
- behaviour-pattern coverage.

Accuracy alone is misleading when defective cases are rare.

## 26.2 Forecast branch metrics

Measure:

- MAE;
- error on degrading trajectories;
- early-warning usefulness;
- forecast coverage/interval behaviour where appropriate;
- performance by time horizon;
- performance by population.

## 26.3 Time-aware evaluation

Do not train on future observations and evaluate on the past.

For checkpoint-based data:

```text
training / reference history
        ↓
decision-time state
        ↓
future holdout
```

Example concept:

```text
Use observations up to t_k
        →
predict t_(k+1)
        →
compare with the actually observed t_(k+1)
```

## 26.4 Label leakage rule

Known labels and future outcomes may be used for **evaluation**, but must not be injected into the production input features in a way that would not exist at decision time.

---

# 28. Current Data Rule for This Project

For the current implementation, use **authentic real datasets from reliable sources** for model development and evaluation. Documentation may contain illustrative examples, but examples are not evidence and must not be represented as target-domain data. When real target-domain data becomes available, use it for controlled calibration/validation.

---

# 29. Authentic-Dataset Methodology

The current methodology uses authentic real datasets from reliable public sources for development/evaluation.

## 27.1 Required dataset record

For every external dataset used, store:

- dataset name;
- organization/source;
- public URL or provenance record;
- dataset version/date when available;
- file checksum;
- license/usage notes;
- parameter descriptions;
- units;
- known test conditions;
- entity identifiers;
- known labels/outcomes;
- transformations performed;
- limitations relative to target deployment.

## 27.2 Transfer to the target domain

External datasets establish methodology and demonstrate the pipeline.

They do **not** prove that a model is already calibrated for ISRO production data.

When approved target-domain data become available:

```text
Target-domain data
       ↓
schema mapping
       ↓
compatibility validation
       ↓
population assessment
       ↓
threshold / configuration calibration
       ↓
model validation
       ↓
engineering approval
```

---

# 30. Limits, Thresholds and Rules

Keep engineering decision values outside model source code.

Separate:

1. **Model configuration**
2. **Data-quality thresholds**
3. **Engineering specification limits**
4. **Decision thresholds**
5. **Alert/review thresholds**

Example:

```yaml
decision_rules:
  version: "..."
  specification:
    source: "<approved source/reference>"
    upper_limit: null
    lower_limit: null

  anomaly:
    review_threshold: null
    high_risk_threshold: null

  forecast:
    review_condition: "<approved rule>"

  data_quality:
    max_missing_fraction: null
```

`null` means “not yet known/configured,” not zero.

---

# 31. Risk Fusion

Anomaly evidence and forecast evidence answer different questions.

They must be evaluated independently before combination.

Conceptual form:

```text
risk evidence =
    anomaly evidence
  + forecast evidence
  + engineering-rule evidence
  + data-quality/confounder evidence
```

A numeric fusion score may be used only when the weights/thresholds are validated and documented.

Do not invent arbitrary weights merely to make a formula look complete.

## 29.1 Example evidence state

```json
{
  "anomaly": {
    "status": "high",
    "score": 0.0
  },
  "forecast": {
    "status": "available",
    "mean": 0.0,
    "std": 0.0
  },
  "specification": {
    "status": "within_limit"
  },
  "data_quality": {
    "status": "trusted"
  },
  "confounder": {
    "status": "no_strong_common_mode_signal"
  }
}
```

---

# 32. Decision Layer

The final solution uses four operational dispositions:

```text
PASS
RETEST
REVIEW
REJECT
```

These are not equivalent to raw model classes.

## 30.1 PASS

Use when:

- data quality is adequate;
- evidence is consistent with acceptable behaviour;
- no approved rule triggers escalation.

## 30.2 RETEST

Use when:

- measurements are not trustworthy enough;
- identity/timestamp/unit/data-integrity problems affect the result;
- the correct action is to repeat the measurement rather than infer a component defect.

## 30.3 REVIEW

Use when:

- evidence is conflicting or ambiguous;
- anomaly is meaningful but not sufficient for an automatic hard disposition;
- forecast risk is uncertain;
- confounders may explain the behaviour;
- component/lot evidence requires engineering judgement.

## 30.4 REJECT

Use only when an approved engineering/decision rule justifies rejection, such as a clearly unacceptable specification condition or sufficiently strong validated evidence.

The AI system should not claim independent authority to approve or reject flight hardware.

---

# 33. Explanation Layer

Every REVIEW/RETEST/REJECT recommendation must have a reason.

## 31.1 Minimum explanation payload

Include, where available:

1. current value;
2. relevant engineering specification limit;
3. lot/peer reference;
4. trend/drift evidence;
5. forecast;
6. predictive uncertainty;
7. data-quality evidence;
8. confounder evidence;
9. final rule that triggered the recommendation.

## 31.2 Plain-English template

```text
Reason:
The component is currently within the configured specification,
but its recent trajectory is more abnormal than the approved peer
population. The forecast indicates continued movement toward the
configured risk region. No strong measurement-quality failure was
detected. QA review is recommended.
```

This is a template. The generated reason must use actual available evidence.

## 31.3 Explainability rule

Do not show:

```text
HIGH RISK
```

without showing why.

---

# 34. GPR Explanation

GPR explanation should primarily use:

- observed trajectory;
- predicted mean;
- uncertainty interval;
- estimated drift;
- proximity to engineering limit;
- validation error;
- amount of available history.

Example:

```text
Forecast reason:
Using the measurements available through the current checkpoint,
the predicted future value continues the observed upward trend.
The prediction interval is wide because available history is limited,
so the result is advisory and routed to REVIEW.
```

---

# 35. Physical-Cause Context

The system may use known semiconductor degradation mechanisms as **interpretive hypotheses**, not automatic root-cause diagnoses.

Candidate physical mechanisms in the project analysis include:

- gate-oxide wear / TDDB;
- electromigration;
- threshold-voltage shift;
- weak wire bonds or package voids;
- contamination;
- micro-cracks.

Implementation rule:

```text
Observed behaviour
    ↓
statistical/anomaly evidence
    ↓
possible mechanism category (optional contextual hint)
    ↓
engineer verification
```

Never state:

> “The AI diagnosed TDDB”

unless a separately validated physical-diagnosis model exists.

---

# 36. Progressive / Streaming State

The deployment layer should update the analysis state whenever a new trusted checkpoint arrives.

## 34.1 State object

```json
{
  "component_id": "...",
  "received_observations": [...],
  "latest_checkpoint": "...",
  "data_quality_status": "...",
  "anomaly_state": {...},
  "forecast_state": {...},
  "decision_state": {...},
  "versions": {
    "mapping": "...",
    "features": "...",
    "anomaly_model": "...",
    "forecast_model": "...",
    "decision_rules": "..."
  }
}
```

## 34.2 Update rule

```text
new checkpoint
      ↓
validate only the new/affected records
      ↓
merge into trusted history
      ↓
recompute applicable features
      ↓
run IF branch
      ↓
run GPR branch when enough history exists
      ↓
re-evaluate decision
      ↓
record state transition
```

A later result must not overwrite the history of earlier states.

---

# 37. API Contract

The backend can be implemented as a FastAPI service.

## 35.1 Suggested endpoints

```text
POST /datasets/register
POST /datasets/profile
POST /datasets/validate
POST /datasets/map
POST /analyze
POST /checkpoints/ingest
GET  /components/{component_id}/state
GET  /components/{component_id}/history
GET  /runs/{run_id}
GET  /audit/{run_id}
GET  /configs/{config_version}
GET  /models/{model_version}
```

The exact endpoint naming can change, but the contract should preserve the separation of:

- ingestion;
- processing;
- analysis;
- decision;
- audit.

## 35.2 `POST /analyze` conceptual input

```json
{
  "dataset_id": "...",
  "mapping_version": "...",
  "config_version": "...",
  "component_scope": "...",
  "analysis_time": "...",
  "data_reference": "..."
}
```

## 35.3 Conceptual response

```json
{
  "run_id": "...",
  "status": "completed|partial|blocked",
  "data_quality": {...},
  "anomaly": {...},
  "forecast": {...},
  "evidence": {...},
  "recommendation": {
    "status": "PASS|RETEST|REVIEW|REJECT",
    "reason": "..."
  },
  "audit_reference": "..."
}
```

---

# 38. Audit and Traceability

Every production/prototype analysis run must be reconstructable.

## 36.1 Minimum audit fields

```text
run_id
component_id
dataset_id
source_reference
mapping_version
unit_policy_version
validation_version
feature_version
anomaly_model_version
forecast_model_version
threshold/rule_version
input observation references
data-quality status
anomaly result
forecast result
evidence
AI recommendation
reviewer
QA action
final decision
timestamps
```

## 36.2 AI recommendation vs final human disposition

Store separately:

```text
AI recommendation
        ≠
Human QA action
        ≠
Final recorded disposition (when policy requires)
```

This distinction is critical for auditability.

---

# 39. Database-Oriented Storage Model

A relational prototype can use SQLite/PostgreSQL.

Recommended logical entities:

```text
datasets
dataset_fields
mapping_versions
unit_rules
raw_records
canonical_observations
data_quality_events
components
lots
test_runs
feature_vectors
anomaly_results
forecast_results
decision_evidence
decisions
audit_events
model_registry
threshold_registry
review_actions
```

## 37.1 Provenance chain

```text
raw record
   ↓
canonical record
   ↓
feature record
   ↓
model result
   ↓
evidence
   ↓
AI recommendation
   ↓
human action
```

Each link should be addressable by a stable identifier.

---

# 40. Dashboard Contract

The UI is a visualization of pipeline outputs, not the place where hidden computation occurs.

Recommended screens:

### Component summary

Show:

- component ID;
- part/lot context;
- latest checkpoint;
- current trusted measurements;
- data quality state.

### Trajectory view

Show:

- selected component trajectory;
- peer/lot reference band when available;
- relevant limit;
- checkpoint markers;
- forecast region and uncertainty.

### Anomaly panel

Show:

- anomaly status;
- anomaly score;
- comparison population.

### Forecast panel

Show:

- predicted future value;
- uncertainty;
- forecast validation metric;
- forecast availability status.

### Reason panel

Show:

- plain-English reason;
- numerical evidence;
- data-quality flags;
- confounder evidence.

### Decision panel

Show:

```text
PASS / RETEST / REVIEW / REJECT
```

### Audit/history

Show:

- previous checkpoints;
- previous AI recommendations;
- reviewer actions;
- configuration/model versions;
- decision history.

---

# 41. Recommended Repository Structure

```text
project/
├── README.md
├── docs/
│   ├── DATA_PIPELINE_IMPLEMENTATION.md
│   ├── SRS.md
│   ├── ARCHITECTURE.md
│   ├── DATABASE.md
│   ├── MODEL_CARD.md
│   ├── DATA_CARD.md
│   ├── TEST_PLAN.md
│   ├── DEPLOYMENT.md
│   ├── RUNBOOK.md
│   └── CHANGELOG.md
│
├── configs/
│   ├── datasets/
│   │   └── <dataset-config>.yaml
│   ├── mappings/
│   │   └── <mapping-version>.yaml
│   ├── units/
│   │   └── <unit-policy>.yaml
│   ├── validation/
│   │   └── <validation-version>.yaml
│   ├── populations/
│   │   └── <population-version>.yaml
│   └── decisions/
│       └── <rule-version>.yaml
│
├── data/
│   ├── raw/
│   ├── staged/
│   ├── canonical/
│   └── quarantine/
│
├── src/
│   ├── ingestion/
│   ├── profiling/
│   ├── compatibility/
│   ├── mapping/
│   ├── validation/
│   ├── preprocessing/
│   ├── grouping/
│   ├── features/
│   ├── models/
│   │   ├── anomaly/
│   │   └── forecasting/
│   ├── evidence/
│   ├── decision/
│   ├── audit/
│   └── api/
│
├── tests/
│   ├── ingestion/
│   ├── validation/
│   ├── mapping/
│   ├── models/
│   ├── decisions/
│   └── end_to_end/
│
└── model_registry/
```

---

# 42. Implementation Order

Build in this order so the team does not create disconnected model code.

## Phase 1 — Common data foundation

1. Source adapters.
2. Dataset registry.
3. Automatic profiler.
4. Compatibility checker.
5. Mapping configuration loader.
6. Unit conversion layer.
7. Canonical internal schema.
8. Validation engine.
9. Quarantine/provenance mechanism.

## Phase 2 — Context and time

10. Component/part/lot grouping.
11. Population configuration.
12. Time normalization.
13. Progressive checkpoint state.
14. Test-condition context.

## Phase 3 — Behaviour representation

15. Baseline features.
16. Drift features.
17. Peer-relative features.
18. Multi-parameter feature representation.
19. Feature versioning.

## Phase 4 — Model branches

20. Isolation Forest.
21. GPR.
22. Independent evaluation.
23. Uncertainty/evidence outputs.

## Phase 5 — Risk and explanation

24. Rule engine.
25. Risk fusion.
26. PASS/RETEST/REVIEW/REJECT.
27. Explanation generator.
28. Evidence pack.

## Phase 6 — Product integration

29. FastAPI.
30. Dashboard.
31. Audit/history.
32. End-to-end tests.
33. Deployment/runbook.
34. Documentation.

---

# 43. Team Workstream Mapping

The agreed team split maps cleanly onto this pipeline.

## Workstream 1 — Isolation Forest + Data Engineering + Anomaly Explainability

Own:

- profiling;
- mapping/config;
- validation;
- canonical representation;
- feature engineering for anomaly branch;
- IF training;
- baseline comparison;
- anomaly metrics;
- anomaly evidence;
- anomaly explanation.

## Workstream 2 — GPR + Forecasting + Forecast Evidence

Own:

- time-series preparation;
- forecast feature design;
- baseline comparison;
- GPR;
- predictive uncertainty;
- MAE and degradation-specific evaluation;
- early-warning evaluation;
- forecast evidence.

## Workstream 3 — Frontend + Backend + Integration

Own:

- API;
- progressive state;
- dashboard;
- trajectory chart;
- anomaly result presentation;
- forecast/uncertainty presentation;
- explanation UI;
- risk fusion;
- decision panel;
- audit/history;
- end-to-end integration.

Common tasks:

- authentic dataset verification;
- testing;
- demo;
- documentation;
- configuration management.

---

# 44. Testing Strategy

Testing must cover both software correctness and decision safety.

## 42.1 Unit tests

Test:

- file readers;
- mapping;
- unit conversion;
- validation rules;
- timestamp normalization;
- grouping;
- feature formulas;
- model wrappers;
- rule engine;
- audit record creation.

## 42.2 Data-quality tests

Create tests using **real dataset examples and naturally occurring data conditions** from the authentic sources where possible.

Verify each validation case listed in Section 13.

## 42.3 Model tests

Verify:

- IF receives only approved features;
- labels are excluded from IF features;
- GPR never receives future observations;
- uncertainty is preserved;
- missing-history conditions return explicit unavailable status.

## 42.4 Decision tests

Test combinations such as:

```text
valid + normal
valid + anomaly
valid + forecast risk
valid + conflicting evidence
invalid data + anomaly-looking values
strong common-mode evidence
critical identity failure
```

Each should map to the expected recommendation policy.

## 42.5 End-to-end test

```text
raw authentic dataset
→ profile
→ map
→ validate
→ canonicalize
→ feature
→ IF/GPR
→ evidence
→ decision
→ audit
→ API response
→ UI
```

---

# 45. No-Leakage Rule

This rule is absolute:

> At any analysis time, the pipeline may only use information that would have been available at that time.

For a checkpoint `t_k`:

```text
Allowed:
t_0, t_1, ..., t_k

Not allowed:
t_(k+1), t_(k+2), future labels, future QA action
```

This applies to:

- features;
- population references when temporally constrained;
- model fitting choices for online evaluation;
- forecast targets;
- decision evidence.

---

# 46. Data Drift and Deployment Monitoring

Deployment monitoring should track:

- missingness changes;
- unit/schema changes;
- population changes;
- feature distribution drift;
- alert-rate changes;
- forecast error changes;
- data-quality failure rates;
- common-mode event frequency;
- model availability;
- reviewer override patterns.

## 44.1 Recalibration rule

When target-domain data arrive, recalibration should be treated as a controlled change:

```text
new approved data
      ↓
evaluate population shift
      ↓
revalidate feature distributions
      ↓
retest thresholds
      ↓
revalidate anomaly model
      ↓
revalidate forecast model
      ↓
engineering approval
      ↓
new version
```

---

# 47. Security and Access Boundaries

The exact production security policy is unknown and must be supplied by the real deployment environment.

The architecture should nevertheless separate:

- source-data access;
- configuration editing;
- model deployment;
- QA review;
- audit access;
- administrative changes.

Configuration and audit records should not be editable by the same uncontrolled path used for ordinary analysis.

---

# 48. Unknowns That Must Remain Explicit

The implementation team must keep these as configuration/input questions, not assumptions:

- exact ISRO data interface;
- exact production file/database/API format;
- exact real parameter names;
- exact production units;
- actual component/part/lot ID structure;
- exact checkpoint policy in all environments;
- exact chamber/test-condition metadata;
- real engineering danger limits;
- authorized defect-rate assumptions;
- approved anomaly cut-offs;
- acceptable false-positive/false-negative trade-off;
- final part types in scope;
- real-time latency target;
- deployment hardware/network/security constraints;
- audit retention period;
- QA sign-off procedure;
- exact RETEST/REVIEW workflow.

When these become available, update configuration and controlled documentation rather than rewriting the whole pipeline.

---

# 49. Future QA Question–Answer Layer

This is a **later add-on**, not a dependency of the core system. Build it after the anomaly, forecasting, decision, API, dashboard, and audit layers are stable.

## 49.1 Purpose
Allow a QA engineer to ask natural-language questions about an already analysed component, such as why it was flagged, which peers were used, what changed over time, what the forecast says, and what happened in previous runs.

## 49.2 Recommended architecture

```text
QA question
    ↓
Question understanding / intent routing
    ↓
Controlled retrieval from system data
    ├─ current analysis
    ├─ component history
    ├─ reference population
    ├─ anomaly evidence
    ├─ forecast evidence
    ├─ rules / thresholds
    └─ audit records
    ↓
Local / approved LLM (natural-language interface)
    ↓
Evidence-grounded answer
```

## 49.3 Why the LLM is not the core system
The anomaly detector, forecast engine, decision rules, and audit trail must work without the LLM. The LLM only turns retrieved system facts into a conversational answer.

## 49.4 Recommended implementation
Use a hybrid model:

```text
Safety-critical structured questions → deterministic backend handlers
Natural-language questions           → local/approved LLM + retrieval
Unknown information                  → explicit “insufficient information”
```

The LLM must never invent a measurement, engineering limit, model result, or final QA decision.

## 49.5 Data-control principle
For sensitive target-domain data, prefer a locally deployed/on-premise or otherwise explicitly approved language model so raw screening data does not leave the controlled environment merely to answer QA questions.

---

# 50. Definition of Done

The generalized data pipeline is implementation-complete when:

### Data layer

- [ ] At least one authentic dataset can be ingested.
- [ ] Dataset structure is automatically profiled.
- [ ] Compatibility status is generated.
- [ ] Mapping is configuration-driven.
- [ ] Units are normalized through explicit rules.
- [ ] Canonical representation is produced.
- [ ] Data-quality events are recorded and traceable.
- [ ] Invalid identity/context is quarantined or blocked.

### Time/context layer

- [ ] Component/part/lot grouping is configuration-driven.
- [ ] Time is normalized without losing source timestamps.
- [ ] Out-of-order arrivals are handled safely.
- [ ] Progressive checkpoint state works.
- [ ] Confounder/test context is preserved where available.

### AI layer

- [ ] IF consumes only approved features.
- [ ] GPR uses trusted history only.
- [ ] Predictive uncertainty is stored.
- [ ] Model versions are recorded.
- [ ] Evaluation is time-aware and leakage-safe.

### Decision layer

- [ ] Evidence is generated automatically.
- [ ] PASS/RETEST/REVIEW/REJECT logic is configurable.
- [ ] No unsupported engineering limit is invented.
- [ ] AI recommendation and human action are separate.

### Product layer

- [ ] API exposes analysis results.
- [ ] UI shows trajectory + anomaly + forecast + reason.
- [ ] Audit/history is available.
- [ ] Every result is reconstructable from stored versions.

---

# 51. Final Implementation Principle

The final system should be understood as:

```text
                 ┌─────────────────────────────┐
                 │ AUTHENTIC BURN-IN DATA      │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ PROFILE + COMPATIBILITY     │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ MAP + UNITS + CONTEXT       │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ VALIDATE + QUARANTINE       │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ CANONICAL INTERNAL DATA     │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ TIME + GROUP + FEATURES     │
                 └──────────────┬──────────────┘
                         ┌──────┴──────┐
                         ▼             ▼
               ┌────────────────┐ ┌────────────────┐
               │ ISOLATION      │ │ GPR            │
               │ FOREST         │ │ FORECAST       │
               │ anomaly now    │ │ future +       │
               │ + history      │ │ uncertainty    │
               └───────┬────────┘ └───────┬────────┘
                       └──────────┬────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │ EVIDENCE + EXPLANATION  │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ DECISION / RISK LAYER   │
                    │ PASS / RETEST / REVIEW  │
                    │ / REJECT                │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ HUMAN QA / ENGINEERING   │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ AUDIT + API + DASHBOARD │
                    └─────────────────────────┘
```

**Core rule:** the data pipeline is reusable; the dataset-specific details live in adapters and versioned configuration. The AI models operate only after semantic mapping and data-quality validation. The final recommendation is evidence-based and traceable, while final QA authority remains with the responsible engineering process.

---

# 52. Quick Glossary for Team Members

| Term | Simple meaning |
|---|---|
| Canonical data | One standard internal format used by the rest of the system |
| Context | Information describing what the measurement belongs to and under what conditions |
| Reference population | Comparable peer components used for a baseline |
| Policy | Configurable rule describing how something should be selected or decided |
| Selector | Code that chooses items according to a policy |
| Validator | Code that checks whether input/selection is usable |
| Fallback | Next approved option when the preferred option cannot be used |
| Leakage | Using information that would not have been available at decision time |
| Peer-relative feature | How a component differs from comparable peers |
| Lot-relative feature | How a lot differs from comparable lots |
| Isolation Forest | Unsupervised model that detects unusual feature patterns |
| GPR | Gaussian Process Regression; forecasts a numeric value with predictive uncertainty |
| Evidence | Concrete measurements/comparisons supporting a result |
| Recommendation | Software output based on evidence and approved rules |
| Disposition | Final authorized QA/engineering action |

---

# 53. Source Mapping Used for This Guide

This implementation guide preserves terminology and constraints from the project sources, especially:

- `PS26170_Master_Problem_Decomposition.pdf` — final 10-part, 74-case master breakdown.
- `Part_3_The_Data_Being_Collected.pdf` — ingestion, mapping, units, progressive data, canonical representation.
- `Part_7_Confounders_SIH_26170.docx` — confounder-aware component/lot analysis and evidence.
- `Part_10.pdf` — streaming, mapping, recalibration, generalization, human QA, auditability.
- `SIH26170_Research_Guide.pdf` — solution architecture, anomaly/forecast separation, evaluation, evidence, prototype constraints.
- `authoring-patterns.md` — semantic authoring structure.
- `diagram-workflow.md` — diagram readability, stable labels, flow, and visual QA guidance.

---

# 54. One-Sentence Architecture Summary

> **Ingest authentic screening data → profile and check compatibility → map source-specific schema/semantics/units into a canonical representation → validate identity/time/context/data quality → build time- and population-aware behaviour features → run Isolation Forest and/or GPR using only information available at that point → generate evidence and uncertainty → apply versioned decision rules → route uncertain cases to human QA → store a complete audit trail.**
