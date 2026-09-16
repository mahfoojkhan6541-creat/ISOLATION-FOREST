import os
import sqlite3
import json
import datetime
from typing import Dict, Any, List, Optional


class AuditStorage:
    """Relational SQLite storage implementing Section 38 & 39 provenance chain."""

    def __init__(self, db_path: str = "data/audit_traceability.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Pipeline Runs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                started_at TEXT,
                completed_at TEXT,
                status TEXT,
                config_version TEXT,
                summary_json TEXT
            );
            """)

            # 2. Anomaly Results table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                component_id TEXT,
                checkpoint REAL,
                anomaly_score REAL,
                anomaly_status TEXT,
                model_version TEXT,
                population_id TEXT,
                created_at TEXT
            );
            """)

            # 3. Forecast Results table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                component_id TEXT,
                param_key TEXT,
                forecast_horizon REAL,
                forecast_mean REAL,
                forecast_std REAL,
                lower_2sigma REAL,
                upper_2sigma REAL,
                forecast_status TEXT,
                model_version TEXT,
                created_at TEXT
            );
            """)

            # 4. Decisions table (AI recommendation)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                component_id TEXT,
                checkpoint REAL,
                recommendation TEXT,
                triggered_rule TEXT,
                confidence TEXT,
                plain_english_reason TEXT,
                evidence_json TEXT,
                created_at TEXT
            );
            """)

            # 5. Human QA Actions table (Strictly separated per Section 38.2)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_qa_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                component_id TEXT,
                checkpoint REAL,
                ai_recommendation TEXT,
                human_action TEXT,
                reviewer_id TEXT,
                sign_off_notes TEXT,
                timestamp TEXT
            );
            """)

            conn.commit()

    def record_run_start(self, run_id: str, dataset_id: str, config_version: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO pipeline_runs (run_id, dataset_id, started_at, status, config_version)
            VALUES (?, ?, ?, ?, ?)
            """, (run_id, dataset_id, datetime.datetime.now(datetime.timezone.utc).isoformat(), "RUNNING", config_version))
            conn.commit()

    def record_run_complete(self, run_id: str, summary: Dict[str, Any], status: str = "COMPLETED"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE pipeline_runs
            SET completed_at = ?, status = ?, summary_json = ?
            WHERE run_id = ?
            """, (datetime.datetime.now(datetime.timezone.utc).isoformat(), status, json.dumps(summary, default=str), run_id))
            conn.commit()

    def record_anomaly_result(
        self,
        run_id: str,
        component_id: str,
        checkpoint: Any,
        score: float,
        status: str,
        model_version: str,
        population_id: str
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO anomaly_results (run_id, component_id, checkpoint, anomaly_score, anomaly_status, model_version, population_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, str(component_id), float(checkpoint) if checkpoint is not None else 0.0, float(score), status, model_version, population_id, datetime.datetime.now(datetime.timezone.utc).isoformat()))
            conn.commit()

    def record_forecast_result(
        self,
        run_id: str,
        component_id: str,
        param_key: str,
        forecast_info: Dict[str, Any]
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            interval = forecast_info.get("interval") or {}
            cursor.execute("""
            INSERT INTO forecast_results (run_id, component_id, param_key, forecast_horizon, forecast_mean, forecast_std, lower_2sigma, upper_2sigma, forecast_status, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                str(component_id),
                param_key,
                forecast_info.get("forecast_horizon"),
                forecast_info.get("forecast_mean"),
                forecast_info.get("forecast_std"),
                interval.get("lower"),
                interval.get("upper"),
                forecast_info.get("forecast_status", "unavailable"),
                forecast_info.get("model_version", "gpr_v1"),
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            ))
            conn.commit()

    def record_decision(
        self,
        run_id: str,
        component_id: str,
        checkpoint: Any,
        decision_result: Dict[str, Any],
        explanation: str,
        evidence_pack: Dict[str, Any]
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO decisions (run_id, component_id, checkpoint, recommendation, triggered_rule, confidence, plain_english_reason, evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                str(component_id),
                float(checkpoint) if checkpoint is not None else 0.0,
                decision_result.get("recommendation", "REVIEW"),
                decision_result.get("triggered_rule", ""),
                decision_result.get("confidence", "MODERATE"),
                explanation,
                json.dumps(evidence_pack, default=str),
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            ))
            conn.commit()

    def record_human_qa_action(
        self,
        run_id: str,
        component_id: str,
        checkpoint: Any,
        ai_recommendation: str,
        human_action: str,
        reviewer_id: str,
        notes: str = ""
    ):
        """Records final authorized human disposition (Section 38.2)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO human_qa_actions (run_id, component_id, checkpoint, ai_recommendation, human_action, reviewer_id, sign_off_notes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                str(component_id),
                float(checkpoint) if checkpoint is not None else 0.0,
                ai_recommendation,
                human_action,
                reviewer_id,
                notes,
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            ))
            conn.commit()

    def query_audit_trail(self, component_id: str) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT checkpoint, anomaly_score, anomaly_status, model_version, created_at FROM anomaly_results WHERE component_id = ? ORDER BY checkpoint", (str(component_id),))
            anomalies = cursor.fetchall()

            cursor.execute("SELECT checkpoint, recommendation, triggered_rule, plain_english_reason, created_at FROM decisions WHERE component_id = ? ORDER BY checkpoint", (str(component_id),))
            decisions = cursor.fetchall()

            cursor.execute("SELECT human_action, reviewer_id, sign_off_notes, timestamp FROM human_qa_actions WHERE component_id = ?", (str(component_id),))
            qa_actions = cursor.fetchall()

            return {
                "component_id": str(component_id),
                "anomalies": anomalies,
                "decisions": decisions,
                "human_qa_actions": qa_actions
            }
