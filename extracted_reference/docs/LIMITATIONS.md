# Limitations and Operational Safeguards

The SIH problem statement supplied for this project does not contain a public dataset link or device-specific acceptance policy. The repository therefore includes synthetic data only for reproducible engineering tests. The model must be retrained and independently validated on authorized, representative data before any operational use.

| Risk | Safeguard in this solution | Required operational control |
|---|---|---|
| Future-data leakage | Tests prove changes to 96h/168h do not alter early decisions. | Review feature changes before every model promotion. |
| Missed latent defect | Calibration weights false negatives far above false positives and retains a REVIEW state. | Set review capacity and acceptance thresholds with the reliability authority. |
| Unit or schema mistake | Invalid and inconsistent records are quarantined. | Maintain a controlled unit dictionary and ingest mapping review. |
| Lot/process shift | Peer baseline groups by lot, device type, parameter, and unit. | Require adequate peer-group size and monitor process changes. |
| Forecast uncertainty | Prediction intervals and uncertainty-driven review reasons are shown. | Do not convert a forecast into a guarantee of future behavior. |
| Policy drift | Policy and calibration information are stored in every artifact. | Require versioned QA approval for all policy changes. |

The system must not be used to silently auto-accept components, to infer unspecified engineering limits, or to represent synthetic evaluation results as measured hardware performance.
