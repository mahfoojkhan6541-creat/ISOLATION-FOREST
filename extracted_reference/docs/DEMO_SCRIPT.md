# Five-Minute QA Demonstration Script

Begin by stating that the displayed dataset is **synthetic demonstration data**. Explain that the system proves the workflow, traceability, and leakage protection; it does not claim real ISRO hardware performance.

| Step | Demonstration action | What to explain |
|---:|---|---|
| 1 | Open the Overview screen. | It separates pass, review, and reject counts and shows the active model and warning that this is decision support. |
| 2 | Open Dataset Validation and select the synthetic file. | The schema lists each canonical field and records any invalid measurement in quarantine rather than dropping it. |
| 3 | Open Train & Evaluate. | The model compares a linear and nonlinear transparent candidate on a grouped split; results display MAE, interval coverage, recall, and false-negative rate. |
| 4 | Open Batch Screening. | Every row uses only 0h and 24h inputs for the early forecast, then receives PASS, REVIEW, or REJECT. |
| 5 | Select a `REVIEW` or `REJECT` result. | The Component Inspector exposes the timeline, peer band, specification limit, forecast interval, early slope, reason codes, and exact policy rule. |
| 6 | Open Model Registry and Reports. | The artifact stores its version, training data hash, coefficients, candidate metrics, and threshold policy; the report captures the same decision evidence. |

Close by emphasizing the operational workflow: validate data, select or train a governed model, screen early measurements, direct uncertain cases to manual review, and retain the model version and explanation with the record.
