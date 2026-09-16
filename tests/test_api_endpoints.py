import pytest
from fastapi.testclient import TestClient
from src.api.service import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "SIH26170" in data["service"]


def test_api_profile_and_compatibility():
    # Register D1
    reg_resp = client.post("/datasets/register", json={
        "dataset_id": "D1_API_TEST",
        "file_path": "data/D1.csv",
        "domain": "burn_in",
        "source_format": "csv"
    })
    assert reg_resp.status_code == 200

    # Profile D1
    prof_resp = client.post("/datasets/profile", json={"dataset_id": "D1_API_TEST"})
    assert prof_resp.status_code == 200
    pdata = prof_resp.json()
    assert pdata["profile"]["dimensions"]["records"] > 1000

    # Check compatibility
    compat_resp = client.post("/datasets/compatibility", json={"dataset_id": "D1_API_TEST"})
    assert compat_resp.status_code == 200
    cdata = compat_resp.json()
    assert cdata["compatibility"]["status"] in ["COMPATIBLE", "PARTIALLY_COMPATIBLE"]


def test_api_analyze_and_qa_action():
    # Run rapid analysis on a small sample of D1
    analyze_resp = client.post("/analyze", json={
        "dataset_id": "D1",
        "mapping_version": "d1_mapping",
        "config_version": "test_v1",
        "sample_limit": 15
    })
    assert analyze_resp.status_code == 200
    adata = analyze_resp.json()
    assert adata["status"] == "completed"
    assert "run_id" in adata
    run_id = adata["run_id"]

    # Test audit run endpoint
    audit_resp = client.get(f"/audit/{run_id}")
    assert audit_resp.status_code == 200
    aud_data = audit_resp.json()
    assert aud_data["decisions_count"] > 0

    # Test Human QA action recording (Section 38.2)
    sample_cid = aud_data["decisions"][0][0]
    sample_chk = aud_data["decisions"][0][1]
    sample_rec = aud_data["decisions"][0][2]

    qa_resp = client.post("/audit/qa-action", json={
        "run_id": run_id,
        "component_id": str(sample_cid),
        "checkpoint": float(sample_chk),
        "ai_recommendation": sample_rec,
        "human_action": "CONFIRM_PASS",
        "reviewer_id": "QA_ENG_LEAD",
        "notes": "Verified per screening protocol"
    })
    assert qa_resp.status_code == 200
    assert qa_resp.json()["status"] == "recorded"
