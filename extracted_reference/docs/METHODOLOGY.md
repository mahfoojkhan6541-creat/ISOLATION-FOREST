# Screening Methodology and Rationale

## Feature design and leakage control

The feature table is constructed at the **component and parameter** level. The system calculates values at the four defined time points, early value change, relative change, early slope, later slopes for retrospective diagnostics, curvature, peer robust z-scores, peer percentiles, and missing-measurement indicators.

For an early Module B action, the actual model matrix contains only `value_0h`, `value_24h`, their derived early-change features, peer features computed from 24h values, and optional stress metadata. It excludes 96h and 168h measurement values. The test suite alters those later values and proves that the early forecast and action remain identical.

## Module A: conservative dynamic anomaly detection

Module A combines four deterministic elements. First, it checks supplied specification limits. Second, it measures peer-relative deviation with a median and median absolute deviation (MAD), which are less distorted by extreme values than a simple mean and standard deviation. Third, it evaluates the early slope against the peer slope distribution. Finally, it fuses normalized level and slope risk using versioned weights.

The policy selects `PASS`, `REVIEW`, or `REJECT`. A hard specification violation is immediately `REJECT`. A forecasted limit breach, a safety-slope breach, a material peer deviation, an abnormal early slope, or poor data quality produces at least `REVIEW`; a combined high-risk forecast and anomaly condition can produce `REJECT`. The exact reason codes, measured values, robust deviations, forecast, interval, and policy settings are persisted with every outcome.

## Module B: 168-hour forecast and uncertainty

Two interpretable candidates are compared on a lot-grouped holdout: linear ridge regression and polynomial ridge regression. Ridge regularization makes the fitted coefficients more stable in the presence of correlated early features. The candidate with lower grouped validation MAE is selected. The 90% prediction interval is based on the selected model's training residual quantile, and interval coverage is reported during evaluation.

The default `remaining_limit` safety-slope policy is calculated separately for each record. For an upper limit, the allowed remaining slope is `(spec_max - value_24h) / 144`; for a lower limit, it is `(spec_min - value_24h) / 144`. This policy is configurable. QA may instead supply absolute slope thresholds or use a peer-reference slope policy. The saved model artifact includes the mode and values actually used, so historic actions can be recreated.

## Validation and threshold calibration

Training uses a grouped holdout by lot when possible; it does not split measurements of a common lot across training and validation. Candidate reports contain MAE, bias, RMSE, 90% interval coverage, recall, false-negative rate, precision, action counts, and the threshold policy. When valid healthy/defective labels are available in the training partition, review thresholds are calibrated with a false negative weighted 100 times more heavily than a false positive. When labels are not sufficient, the conservative default policy remains in use and is explicitly marked `conservative_default`.

This high false-negative cost is intentional: the SIH statement describes a missed latent defect as catastrophic. It creates a larger human-review queue, so QA must review the recorded precision, action distribution, and capacity before adopting a threshold operationally.

## Explainability

Explanations are deterministic templates generated from the actual record and policy. There is no language model in the decision path. The inspector sees the 24-hour value, peer deviation, early slope, predicted 168-hour value, uncertainty interval, applicable specification limit, action, reason codes, and plain-English explanation. This makes a decision stable, auditable, and understandable by a QA inspector.
