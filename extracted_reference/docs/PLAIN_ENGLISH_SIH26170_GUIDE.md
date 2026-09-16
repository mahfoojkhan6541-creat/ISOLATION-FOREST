# SIH26170 Solution Guide in Very Simple English

**Project name:** Burn-In Screening Intelligence  
**Problem statement:** SIH26170 — *AI-Driven Anomaly Detection in Component Burn-In & Screening*  
**Prepared for:** the SIH26170 project team

## 1. What the original SIH26170 problem is asking for

SIH26170 is about finding risky electronic components **before** they are used in a high-reliability system, such as a space payload. During a burn-in test, a component is operated under stress, often at a high temperature, and electrical parameters are measured over time. Examples include leakage current, standby current (`Iddq`), and propagation delay.

The original problem says that normal pass/fail checking is not enough. A component may still be below the absolute datasheet limit but may be changing in a strange or dangerous way. This is called a **latent defect**. For example, if a lot normally has leakage current near `10 µA`, a component at `45 µA` is suspicious even if the formal maximum limit is `50 µA`.[1]

The statement asks for two main modules. **Module A** must find unusual components by looking beyond static limits. **Module B** must use only the `0h` and `24h` readings to predict the `168h` value and flag risky drift early. It also asks for low false negatives, accurate 168-hour prediction, and explanations that a QA inspector can understand.[1]

> **In one line:** the project is an early-warning quality system. It does not only ask, “Is the value legal right now?” It also asks, “Does this component look unusual, and is it likely to become unsafe later?”

## 2. What has actually been built

The delivered solution is a complete **decision-support platform**, not only a machine-learning notebook. It contains a reproducible Python screening engine and an authenticated web dashboard. The Python engine is the source of truth: it validates data, creates features, trains models, evaluates them, scores components, and generates deterministic explanations. The web dashboard gives QA engineers a simpler way to use and inspect that engine.

| Built part | What it does in simple words | Why it is useful |
|---|---|---|
| Canonical data validator | Checks CSV, XLSX, XLS, and JSON measurement files before analysis | Stops bad, missing, mixed-unit, or wrongly structured data from producing misleading decisions |
| Synthetic demo generator | Creates clearly labelled artificial burn-in data | Lets the team demonstrate the complete workflow because the SIH page did not provide a public dataset link |
| Module A detector | Finds hard limit failures, unusual peer values, and abnormal early drift | Catches latent defects that ordinary pass/fail limits can miss |
| Module B predictor | Predicts the 168-hour value using only 0h and 24h information | Supports an early decision instead of waiting for the complete burn-in period |
| Conservative policy engine | Produces `PASS`, `REVIEW`, or `REJECT` | Makes the decision action clear and keeps uncertain cases visible to people |
| Explainability engine | Creates fixed reason codes and plain-English explanations | Allows a QA engineer to understand why the system acted |
| QA web dashboard | Shows validation, training, screening, inspection, model settings, and reports | Makes the solution usable as a demonstration and as a future QA workflow base |
| Database and audit layer | Keeps dataset metadata, model versions, screening runs, and structured decisions | Supports traceability and later review |
| Test suite and documentation | Tests important safety rules and explains the system | Helps judges, developers, and QA teams verify what the solution does |

## 3. What the system looks like in real life

In real use, a lab or QA team exports burn-in measurements from its test system. The file is uploaded to the platform. The platform first checks that the data makes sense. It then trains or applies the saved model, scores every component and parameter, and shows a decision for each one.

The system does **not** silently throw away uncertain parts. It has three clear outcomes.

| Outcome | Meaning in plain English | Typical next action |
|---|---|---|
| **PASS** | The component looks normal under the currently configured rules | Continue normal QA handling, while retaining the evidence |
| **REVIEW** | There is a warning, uncertainty, missing evidence, or a suspicious but not decisive signal | A QA engineer should inspect the component and decide what extra action is needed |
| **REJECT** | There is a hard limit breach or a strong combined risk signal | Hold or reject the component according to the organization’s approved QA procedure |

These are **decision-support actions**, not a replacement for formal organizational approval. The system makes its evidence visible; the responsible QA organization remains responsible for final acceptance, re-test, disposition, and use in flight hardware.

## 4. The complete flow from raw measurements to a decision

The following table explains the whole system as a simple pipeline.

| Step | What happens | Simple explanation |
|---|---|---|
| 1. Collect | The lab measures values at `0h`, `24h`, `96h`, and `168h` | These are the burn-in observations for each component and parameter |
| 2. Validate | The platform checks columns, numbers, units, time points, limits, and duplicates | It prevents “garbage in, garbage out” |
| 3. Group peers | The platform compares a component with similar components | A component should be compared to its own lot/device/parameter group, not to unrelated parts |
| 4. Build early features | It calculates value changes, early slope, peer deviation, percentiles, and data-quality flags | These features describe whether the component is behaving normally or strangely |
| 5. Module A | It checks limits, peer deviation, and suspicious drift patterns | This is the dynamic outlier detector |
| 6. Module B | It predicts the 168-hour value from only 0h and 24h evidence | This is the early drift predictor |
| 7. Apply policy | It combines the evidence into PASS, REVIEW, or REJECT | The exact rule and thresholds are stored with the decision |
| 8. Explain | It creates reason codes and a readable explanation | A QA inspector can see why the action was taken |
| 9. Audit | It stores dataset metadata, model version, run metadata, and results | The team can trace a decision back to its source |

## 5. Data the system understands

The platform supports **CSV**, **XLSX/XLS**, and **JSON** input. The preferred format is “long format”: one row means one measured parameter at one time for one component.

| Required field | Example | What it means |
|---|---|---|
| `component_id` | `L01-C021` | Unique component identifier |
| `lot_id` | `LOT-01` | Manufacturing lot or peer-group identifier |
| `parameter` | `Iddq` or `LeakageCurrent` | Electrical property being measured |
| `time_hours` | `0`, `24`, `96`, `168` | Burn-in time at which the value was measured |
| `value` | `55.262` | Measured value |
| `unit` | `uA`, `ns`, `V` | Unit for the parameter |
| `stress_temperature_c` | `125` | Burn-in stress temperature, if available |
| `spec_min` | `0` | Lower allowed limit |
| `spec_max` | `50` | Upper allowed limit |

The platform also accepts useful optional information, including device type, stress condition, test identifier, and an optional historical label such as `healthy` or `defective`. Labels are helpful for evaluating recall and false-negative rate, but the system can still run when labels are not available.

### Important data rules

The same parameter should not mix units inside one comparable group. For example, do not combine `µA` and `mA` without converting them first. A component/parameter needs `0h` and `24h` values for an early Module B decision. The `168h` measurement is needed for training and retrospective evaluation because it is the future value the model is trying to predict.

> **Very important:** the platform never uses the real `96h` or `168h` value when making an early 24-hour decision. This prevents “seeing the answer in advance.”

## 6. Module A explained: the dynamic anomaly detector

Module A answers this question:

> “Is this component already unusual compared with its own specification and with similar components?”

It uses several layers instead of one simple rule.

### A. Hard specification limit check

If the current reading is beyond the supplied maximum or minimum limit, the component is a clear failure. This produces a **REJECT** action.

Example: if the maximum leakage limit is `50 µA` and the 24-hour leakage is `55 µA`, the part is rejected immediately.

### B. Peer comparison

The system compares a component with similar items in its group, such as the same lot, device type, and parameter. It uses the group’s **median** and **median absolute deviation (MAD)**. These are robust statistics: one extreme component does not distort them as much as a normal average and standard deviation.

Example: if the peer group is near `10 µA` and one component is `45 µA`, the component receives a strong peer-deviation warning even if `45 µA` is still below a static limit of `50 µA`.

### C. Early slope comparison

The system looks at how much the value changed from `0h` to `24h`. It also compares that change rate with the normal early change rate of peers.

Example: a part may be within the limit at `24h`, but its leakage may be rising much faster than all comparable parts. That is a sign of possible latent weakness.

### D. Risk fusion

The system combines peer-level risk and slope-level risk using saved, versioned policy weights. A strong single issue can cause REVIEW. Several strong issues together can cause REJECT.

### Why Module A is better than static limits alone

Static limits answer only: “Has the part crossed the red line?” Module A also answers: “Is this part heading toward the red line, or behaving unlike its peers?” This directly addresses the SIH26170 requirement for **dynamic outlier detection**.[1]

## 7. Module B explained: the early 168-hour drift predictor

Module B answers this question:

> “Based only on what we knew at 0h and 24h, where is this parameter likely to be at 168h?”

The system trains interpretable regression models. It compares a regular linear ridge model and a polynomial ridge model. The model with better performance on a held-out lot group is selected. The evaluation records MAE, bias, interval coverage, recall, false-negative rate, precision, and action counts.

### The 24-hour rule

For an early decision, Module B uses only information available at or before 24 hours:

| Used at 24h | Not used at 24h |
|---|---|
| 0h value | Actual 96h value |
| 24h value | Actual 168h value |
| Change from 0h to 24h | Later-time slopes |
| Early slope | Any future measurement |
| Peer information calculated from 24h values | Hidden future ground truth |
| Optional stress metadata | Retrospective outcomes |

This is one of the most important design choices. If a model used the real 96h or 168h reading while claiming to make a 24h prediction, it would be unfair and useless in real life. The built test suite specifically changes later values and checks that the early forecast and action do not change.

### Forecast interval

The platform shows not only a single predicted 168-hour number but also a **90% prediction interval**. This tells the QA engineer how uncertain the prediction is.

For example, the system may say:

> “Predicted value at 168h: `59.1 µA ± 1.0 µA`.”

This means the platform is not pretending to know the future perfectly. It shows a likely range and lets the safety policy consider that uncertainty.

### Safety slope

The default policy calculates how much change is still safely possible between 24h and 168h. For an upper limit, it asks:

> “From the current 24-hour value, how fast may the value rise and still remain below the maximum limit by 168h?”

If the observed early drift is already faster than this safe rate, the part is flagged. This directly matches the SIH26170 requirement to flag a component when predicted 168-hour drift exceeds a calculated safety slope.[1]

## 8. How the system decides PASS, REVIEW, or REJECT

The decision policy is intentionally conservative because the problem statement says missing a defective component is catastrophic.[1]

| Situation | Typical action | Reason shown to QA |
|---|---|---|
| Measurement is inside limits, close to peers, normal slope, safe forecast | PASS | No decisive anomaly detected under the active policy |
| Peer deviation is high, slope is unusual, data is incomplete, or forecast is uncertain | REVIEW | Human attention is needed before a part is silently accepted |
| Current value exceeds a hard limit | REJECT | `ABSOLUTE_LIMIT_EXCEEDED` |
| Forecast plus uncertainty crosses a limit | REVIEW or REJECT depending on combined risk | `PREDICTED_168H_LIMIT_BREACH` |
| Early slope is too high for the remaining safe margin | REVIEW or REJECT | `SAFETY_SLOPE_EXCEEDED` |
| Component is far from its peers | REVIEW or REJECT | `PEER_DEVIATION_HIGH` |
| Early drift is different from peer drift | REVIEW or REJECT | `EARLY_DRIFT_ABOVE_PEER_SLOPE` |

The exact threshold values and weights are included in the saved model artifact. This is important because a decision must be reproducible later. Two people running the same artifact on the same measurements should obtain the same result.

## 9. Explainability: how a QA inspector understands a decision

The system does not use a generative language model to make the quality decision. It uses fixed, deterministic calculation rules and fixed explanation templates. This is deliberate.

For every component/parameter result, the inspector can see:

| Evidence shown | Why it matters |
|---|---|
| 0h, 24h, 96h, and 168h trace | Shows the real measured behaviour over time |
| 24h value | Shows what was actually known at the early decision point |
| Peer median and peer band | Shows what similar components look like |
| Specification limits | Shows the allowed operating range |
| Predicted 168h value | Shows where the model expects the parameter to go |
| 90% interval | Shows uncertainty around that prediction |
| Action | Shows PASS, REVIEW, or REJECT clearly |
| Reason codes | Gives short, stable technical reasons |
| Plain-English explanation | Turns the numerical evidence into a readable QA explanation |

For example, an explanation can say that the 24-hour value exceeded the supplied limit, was far from its peers, had an abnormal early slope, and had a forecast that crossed the limit. The same facts are also shown visually in the inspector.

## 10. How to use the web dashboard

The dashboard is intended for an authenticated QA user. Sign in and use the left navigation.

| Dashboard area | What to do there | What you will see |
|---|---|---|
| **Overview** | Start here | Main screening counts, the early-risk queue, 168-hour MAE, and the visible 24-hour decision contract |
| **Datasets** | Upload a file for validation | File structure checks, clean-row preview, quarantined-row preview, and data-quality feedback |
| **Train & evaluate** | Upload a validated dataset and choose **Upload & train** | A trained deterministic model, grouped-validation comparison, metrics, saved policy, and a new authenticated workspace |
| **Batch screening** | Filter PASS, REVIEW, or REJECT results | Component list, 24h-to-forecast comparison, risk score, action, and leading reason |
| **Component inspector** | Choose **Inspect** for a component | Timeline, peer band, spec limits, forecast, interval, explanation, and reason codes |
| **Model registry** | Review the active model | Model type, training hash, validation information, and policy settings |
| **Reports** | Export a concise summary | Screening counts, evidence description, limitations, and a downloadable demonstration summary |

### Recommended dashboard workflow

1. Prepare a measurement file in the required format.
2. Open **Datasets** and upload it first.
3. Read the validation summary. Fix quarantined records instead of ignoring them.
4. Open **Train & evaluate** and select **Upload & train** only after the data is suitable.
5. Check the grouped validation metrics. Do not judge the model only by one accuracy number.
6. Open **Batch screening** and focus first on REJECT and REVIEW items.
7. Open **Component inspector** for each important item.
8. Read the measurement trace, peer comparison, forecast, interval, and reason codes.
9. Record the human QA decision outside or alongside the system according to the organization’s approved process.

## 11. How to use the Python pipeline

The Python pipeline is included so the project is reproducible without relying only on the dashboard. It can validate data, generate a synthetic demo set, train, evaluate, score, and write reports.

The main commands are structured around these actions:

| CLI action | Purpose |
|---|---|
| `generate-demo` | Create clearly labelled synthetic burn-in measurements for demonstration and testing |
| `validate-data` | Validate an input file, create clean data, and quarantine invalid records |
| `train` | Train the interpretable early-drift model and save a versioned artifact |
| `evaluate` | Measure prediction and screening performance on the available labelled data |
| `predict` | Score a batch and produce PASS/REVIEW/REJECT decisions with explanations |
| `report` | Create a readable screening summary from the results |

The exact setup and example command sequence are documented in the project’s `README.md` and `docs/DEMO_SCRIPT.md`. The important point is that the dashboard and the Python service use the same screening logic, so the result is not a separate “demo-only” calculation.

## 12. How each part maps to the original SIH26170 statement

| Original SIH26170 requirement | Delivered implementation |
|---|---|
| Analyze measurements at 0h, 24h, 96h, and 168h | Canonical validator and feature pipeline support the stated staged measurement times |
| Detect unusual components, not only static failures | Module A checks hard limits, peer deviation, early slope, and combined dynamic risk |
| Example: 45 µA is suspicious even if max is 50 µA | Robust peer comparison identifies large deviation from similar components |
| Use 0h and 24h to predict 168h | Module B has an enforced early-only feature set and leakage-prevention tests |
| Flag when predicted drift exceeds safety slope | Configurable safety-slope policy supports remaining-limit, absolute-threshold, and peer-reference modes |
| Penalize false negatives heavily | Conservative calibration weights a missed labelled defect much more heavily than unnecessary review |
| Measure prediction MAE | Evaluation records MAE and related prediction metrics |
| Explain classification to QA inspector | Deterministic reason codes, explanations, trace chart, peer band, limits, forecast, and uncertainty are shown in the inspector |
| Build a usable solution | Authenticated web platform, reproducible Python CLI, database audit records, reports, tests, and documentation |

## 13. Testing that has been completed

The project includes automated tests because this is a QA decision-support system. The tests are not a claim that it is ready for flight use; they prove important implementation rules.

| Test area | What is checked |
|---|---|
| Reproducibility | The same source data and configuration give the same decisions |
| Leakage prevention | Changing 96h or 168h values does not change the early 24h forecast/action |
| Validation | Missing fields, unsupported files, empty data, unit mismatch, invalid structure, and incomplete time information are handled safely |
| Boundaries | A value at a specification limit is handled differently from a value beyond that limit |
| Policy | Safety-slope modes and conservative false-negative calibration are tested |
| Python/dashboard parity | Results from the Python service are compared with the dashboard’s saved demo results, including action, risk, forecast, interval, reason codes, and explanation |
| API | Authenticated screening access is tested |
| Build | TypeScript checks, application tests, Python tests, and the production build complete successfully |

## 14. Why the solution uses these methods

The solution intentionally uses interpretable methods rather than an opaque black-box-only model. There are several practical reasons.

First, the original challenge explicitly asks for explainability. A QA inspector needs to know what measurement, peer comparison, trend, limit, and policy rule led to the action. Second, burn-in datasets can be limited, uneven, and grouped by lot. A simple-looking random train/test split can give misleadingly good results, so the solution uses grouped validation by lot when possible. Third, high-reliability screening has a special risk: letting a bad component escape can be worse than sending an extra component for human review. That is why REVIEW is a first-class action and false-negative risk is treated conservatively.

The platform uses robust peer statistics because normal averages can be distorted by outliers. It uses ridge regression because it is stable with correlated early features and easier to explain. It compares a linear and a nonlinear polynomial candidate rather than assuming one model is always better. It records the actual model artifact and policy so past decisions can be reconstructed.

## 15. What this solution is, and what it is not

### What it is

This is a complete and working **prototype QA screening platform** for SIH26170. It demonstrates the required Module A and Module B logic, a real training/evaluation/scoring flow, data validation, deterministic explanations, visual inspection, saved audit metadata, and a judge-friendly web experience.

### What it is not

It is not yet a certified production system for flight acceptance. The public SIH statement did not provide a dataset link, so the supplied demo data is clearly marked synthetic. The synthetic metrics demonstrate that the software works; they must not be presented as real spacecraft-component performance.

Before operational use, the team must retrain on authorized representative measurement data, confirm units and peer-group definitions, validate results by component family and lot, obtain QA approval of limits and policy thresholds, define user roles and sign-off workflow, and perform cybersecurity and deployment review.

## 16. A simple example

Imagine 100 similar components. Most have leakage current around `10 µA` at 24 hours. One component has `45 µA`.

1. A static rule may say it still passes because the limit is `50 µA`.
2. Module A compares it with peers and sees that `45 µA` is far from normal.
3. Module A also sees that its value rose much faster from 0h to 24h than other components.
4. Module B predicts where it may be by 168 hours and gives a range, not just one number.
5. If the prediction or its uncertainty crosses `50 µA`, or the early slope is too high to remain safe, the system flags it.
6. The inspector sees **why**: the actual 24h value, the peer gap, slope warning, forecast, interval, limit line, and reason codes.
7. The component becomes REVIEW or REJECT instead of silently passing because it has not crossed the static limit yet.

That is the main value of this project.

## 17. Final simple summary

> **This solution turns burn-in measurements into an early, explainable QA decision. It catches obvious failures, looks for subtle abnormal drift, predicts the likely 168-hour value using only early evidence, and gives a human-readable reason for every PASS, REVIEW, or REJECT action.**

It directly addresses SIH26170 by combining dynamic anomaly detection, early time-series forecasting, false-negative-aware policy, and clear QA explainability in one usable platform.

## References

[1] [Smart India Hackathon — SIH26170: AI-Driven Anomaly Detection in Component Burn-In & Screening](https://www.sih.gov.in/)
