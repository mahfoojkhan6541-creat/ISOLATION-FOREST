import sqlite3
import json

conn = sqlite3.connect("data/audit_traceability.db")
cursor = conn.cursor()

for run_id in ["run_d1_ae1d8eda", "run_d2_f1c7d232"]:
    cursor.execute("SELECT count(*) FROM decisions WHERE run_id = ?", (run_id,))
    cnt = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM forecast_results WHERE run_id = ?", (run_id,))
    fcnt = cursor.fetchone()[0]
    print(f"Run {run_id}: {cnt} decisions, {fcnt} forecasts")

cursor.execute("""
SELECT d.component_id, d.checkpoint, d.recommendation, d.triggered_rule, d.plain_english_reason, d.evidence_json,
       a.anomaly_score, a.anomaly_status,
       f.forecast_mean, f.forecast_std, f.lower_2sigma, f.upper_2sigma, f.forecast_horizon
FROM decisions d
LEFT JOIN anomaly_results a ON d.run_id = a.run_id AND d.component_id = a.component_id
LEFT JOIN forecast_results f ON d.run_id = f.run_id AND d.component_id = f.component_id
WHERE d.run_id = 'run_d1_ae1d8eda'
LIMIT 5
""")
print("\nSample D1 components:")
for r in cursor.fetchall():
    print(f"ID: {r[0]}, Checkpoint: {r[1]}, Rec: {r[2]}, Rule: {r[3]}, AnomScore: {r[6]}, ForecastMean: {r[8]}, ForecastStd: {r[9]}")
