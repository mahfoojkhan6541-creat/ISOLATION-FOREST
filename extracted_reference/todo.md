# Project TODO

- [x] Define and validate canonical CSV, XLSX, and JSON burn-in schemas, including component, lot, parameter, time, unit, stress conditions, specification limits, and optional labels.
- [x] Build a reproducible Python CLI for validation, synthetic demo-data generation, training, evaluation, batch scoring, and report generation.
- [x] Engineer leakage-safe, interpretable time-series and peer-relative features from 0h, 24h, 96h, and 168h measurements.
- [x] Implement Module A layered anomaly detection with absolute limits, peer deviation, temporal signatures, calibrated risk fusion, and PASS/REVIEW/REJECT actions.
- [x] Implement Module B 168-hour prediction using only 0h and 24h information for early decisions, uncertainty intervals, and configurable safety-slope policies.
- [x] Compare transparent and nonlinear models with grouped validation and record the required performance, calibration, and policy metrics.
- [x] Produce deterministic reason codes and plain-English explanations for every screening action.
- [x] Build an authenticated, polished QA dashboard with overview, dataset validation, training, screening, inspector, model registry, and reporting views.
- [x] Show accessible interactive timelines, peer bands, specification limits, forecasts, confidence intervals, decision summaries, filters, loading, empty, error, and partial-data states.
- [x] Add parity, boundary, malformed-input, missing-time-point, unit-mismatch, leakage-prevention, reproducibility, API, and end-to-end workflow tests.
- [x] Write the model card, data contract, setup guide, demo script, methodology, limitations, and implementation rationale in clear English.
- [x] Preserve the stated constraints: Module B early decisions use only data available by 24h; screening actions are PASS, REVIEW, or REJECT; outputs are deterministic and interpretable.
- [x] Add inspector peer bands and specification-limit visuals, with a clear partial-data state for incomplete measurement series.
- [x] Add malformed-upload tests and strengthen full deterministic parity checks between the service path and saved CLI outputs.
- [ ] Fix the authenticated /datasets TRPC "Failed to fetch" error and verify the dataset validation workflow.
