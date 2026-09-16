# SIH26170 Burn-In Screening Intelligence

This repository implements an **explainable screening aid** for the SIH26170 challenge, *AI-Driven Anomaly Detection in Component Burn-In & Screening*. It has two coordinated parts. The `ml/` directory contains the reproducible Python source of truth for data validation, training, scoring, evaluation, and reporting. The web application provides an authenticated QA workspace for inspecting the same deterministic decisions.

> **Safety statement:** This software supports screening decisions. It does not replace approved component qualification, failure analysis, QA sign-off, or any flight-acceptance process. Synthetic demonstration metrics are not evidence of field or flight reliability.

## Why this solution is structured this way

Burn-in is used to expose early-life defects under elevated operating stress, while environmental screening aims to reveal latent weaknesses before use.[1] [2] The SIH statement requires both dynamic peer-relative anomaly detection and a 0h/24h-to-168h early forecast. The implementation therefore uses a layered, conservative policy rather than a single opaque classifier.

| Layer | What it checks | Why it is included |
|---|---|---|
| Data validation | Schema, units, time points, duplicate readings, numeric values, and specification ranges. | A model result is not trustworthy when the underlying measurement record is ambiguous or malformed. |
| Absolute-limit gate | Whether the 24-hour value already violates a supplied upper or lower limit. | Obvious nonconformance should remain an explicit engineering rule. |
| Peer-relative anomaly signals | Robust deviation from the lot/device/parameter median and early drift slope. | A component can be abnormal for its peer group even if it remains inside an absolute datasheet limit. |
| Module B forecast | A transparent ridge forecaster predicts 168-hour value from only 0h and 24h information. | This meets the early-decision constraint without reading future measurements. |
| Conservative policy | The fused risk, forecast interval, and safety-slope policy return `PASS`, `REVIEW`, or `REJECT`. | A `REVIEW` state makes uncertainty visible instead of forcing a weak binary decision. |

## Quick start

The following commands run the full clearly labelled synthetic demonstration. The data is generated solely for development and testing; do not treat its metrics as real ISRO or flight-hardware performance.

```bash
cd /home/ubuntu/sih26170-screening
python3 -m pip install -r ml/requirements.txt
mkdir -p artifacts
python3 ml/cli.py generate-demo-data --output artifacts/synthetic_burnin_demo.csv
python3 ml/cli.py train --input artifacts/synthetic_burnin_demo.csv --model-output artifacts/demo_model.json
python3 ml/cli.py evaluate --input artifacts/synthetic_burnin_demo.csv --model artifacts/demo_model.json --output artifacts/demo_metrics.json
python3 ml/cli.py predict --input artifacts/synthetic_burnin_demo.csv --model artifacts/demo_model.json --output artifacts/demo_scores.csv
python3 ml/cli.py report --input artifacts/synthetic_burnin_demo.csv --model artifacts/demo_model.json --output artifacts/demo_report.html
```

Run the Python tests with `python3 -m pytest ml/tests -q`. The test suite includes reproducibility, early-decision leakage prevention, malformed-record handling, unit mismatch handling, and configurable safety-slope policy checks.

## CLI reference

| Command | Input | Output | Purpose |
|---|---|---|---|
| `validate-data` | CSV, XLSX, XLS, or JSON | Clean and quarantine CSV files plus a JSON summary. | Checks the canonical data contract. |
| `generate-demo-data` | Seed and output path. | Clearly labelled synthetic CSV. | Supports reproducible development and demo runs. |
| `train` | Validated measurement file. | Versioned JSON model artifact. | Fits and compares linear and polynomial ridge candidates with grouped validation. |
| `evaluate` | Measurement file and model artifact. | JSON metrics. | Reports MAE, bias, RMSE, interval coverage, action counts, recall, false-negative rate, and precision when labels exist. |
| `predict` | Early measurement file and model artifact. | Auditable component scores. | Produces Module A and Module B screening results. |
| `report` | Measurement file and model artifact. | HTML QA report. | Produces a printable screening summary and component-level explanations. |

## References

[1] [JEDEC, *JESD22-A108C: Temperature, Bias, and Operating Life*](https://www.jedec.org/standards-documents/docs/jesd-22-a108c)

[2] [NASA S3VI Reliability and Safety Knowledge Base, *Burn-In*](https://s3vi.ndc.nasa.gov/ssri-kb/topics/47/)
