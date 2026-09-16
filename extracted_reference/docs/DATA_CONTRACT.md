# Canonical Burn-In Measurement Data Contract

The pipeline accepts **CSV, XLSX/XLS, and JSON**. It accepts either canonical long-format records or a simple wide format using `value_0h`, `value_24h`, `value_96h`, and `value_168h`. All inputs are normalized into long-format measurements before any model step.

| Field | Required | Type | Example | Validation rule | Why it matters |
|---|---:|---|---|---|---|
| `component_id` | Yes | Text | `L01-C018` | Non-empty; unique with parameter/time. | Gives every screening decision a traceable item identity. |
| `lot_id` | Recommended | Text | `LOT-01` | Defaults to `UNSPECIFIED_LOT` when absent. | Stops a lot-wide process shift from being mistaken for a component outlier. |
| `device_type` | Recommended | Text | `ASIC-A` | Defaults to `UNSPECIFIED_DEVICE` when absent. | Keeps unlike component families out of the same peer baseline. |
| `parameter` | Yes | Text | `Iddq` | Non-empty. | Defines the measured characteristic being compared and forecast. |
| `time_hours` | Yes | Number | `24` | Must be exactly 0, 24, 96, or 168. | Preserves a fixed, auditable time-series contract. |
| `value` | Yes | Number | `12.44` | Must be finite numeric data. | Is the actual measured parameter value. |
| `unit` | Yes | Text | `uA` | Non-empty; one unit per component/parameter series. | Prevents unsafe comparisons of incomparable values such as µA and mA. |
| `temperature_c` | No | Number | `125` | Optional numeric value. | Records the stress condition that may affect drift. |
| `voltage_v` | No | Number | `3.3` | Optional numeric value. | Records the electrical stress condition that may affect drift. |
| `spec_min` | No | Number | `0` | Must not exceed `spec_max`. | Supports explicit lower-limit checks and lower-slope policy. |
| `spec_max` | No | Number | `50` | Must not be below `spec_min`. | Supports explicit upper-limit checks and upper-slope policy. |
| `label` | No | Text | `healthy` or `defective` | Retained as supplied. | Enables supervised threshold evaluation when trustworthy labels exist. |

## Canonical long-format example

```csv
component_id,lot_id,device_type,parameter,time_hours,value,unit,temperature_c,voltage_v,spec_min,spec_max,label
L01-C001,LOT-01,ASIC-A,Iddq,0,9.84,uA,125,3.3,0,50,healthy
L01-C001,LOT-01,ASIC-A,Iddq,24,10.12,uA,125,3.3,0,50,healthy
L01-C001,LOT-01,ASIC-A,Iddq,96,10.76,uA,125,3.3,0,50,healthy
L01-C001,LOT-01,ASIC-A,Iddq,168,11.21,uA,125,3.3,0,50,healthy
```

## Accepted wide-format example

```csv
component_id,lot_id,device_type,parameter,unit,temperature_c,voltage_v,spec_min,spec_max,label,value_0h,value_24h,value_96h,value_168h
L01-C001,LOT-01,ASIC-A,Iddq,uA,125,3.3,0,50,healthy,9.84,10.12,10.76,11.21
```

## Validation outcomes

Invalid rows are never silently discarded. The validator writes them to a quarantine output with one or more reason codes. `UNSUPPORTED_TIME_POINT`, `NON_NUMERIC_MEASUREMENT`, `MISSING_REQUIRED_IDENTIFIER`, `DUPLICATE_MEASUREMENT`, `UNIT_MISMATCH_WITHIN_COMPONENT`, and `INVALID_SPECIFICATION_RANGE` are currently enforced. A series with only 0h and 24h data is valid for early screening, but cannot be used to train or retrospectively evaluate the 168-hour forecast.
