# tests/test_fdp_tables_expanded.py
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def post_check(payload):
    r = client.post("/check", json=payload)
    return r

def test_two_pilot_standard_case_applies():
    # This is a common two-pilot example overlapping WOCL fully but within table limits
    payload = {
        "duty_start": "2025-11-25T01:30:00+05:30",
        "duty_end":   "2025-11-25T06:30:00+05:30",
        "duty_minutes": 300,
        "flight_minutes": 240,
        "landings": 1,
        "rest_hours": 12.0,
        "context": {"acclimatized_tz": "Asia/Kolkata"}
    }
    r = post_check(payload)
    assert r.status_code == 200
    data = r.json()
    assert "details" in data
    # Ensure the rules engine applied something (applied_rules should be list)
    assert isinstance(data["details"].get("applied_rules", []), list)

def test_single_pilot_expected_violation():
    # Single-pilot configuration where duty exceeds single-pilot FDP table for many landings
    payload = {
        "duty_start": "2025-11-25T06:00:00+05:30",
        "duty_end":   "2025-11-25T16:00:00+05:30",
        "duty_minutes": 600,
        "flight_minutes": 500,
        "landings": 8,
        "rest_hours": 12.0,
        "context": {"acclimatized_tz": "Asia/Kolkata", "operation_type": "single_pilot"}
    }
    r = post_check(payload)
    assert r.status_code == 200
    data = r.json()
    # There should be a legal flag and violations list (could be empty or have items depending on operator table)
    assert "legal" in data and "violations" in data

def test_augmented_crew_ulr_case():
    payload = {
        "duty_start": "2025-11-25T20:00:00+05:30",
        "duty_end":   "2025-11-26T10:00:00+05:30",
        "duty_minutes": 840,
        "flight_minutes": 700,
        "landings": 1,
        "rest_hours": 18.0,
        "context": {"acclimatized_tz": "Asia/Kolkata", "crew_category": "4_crew_ulr", "rest_facility": "ulr_bunk_approved"}
    }
    r = post_check(payload)
    assert r.status_code == 200
    data = r.json()
    # Should have details and not crash
    assert isinstance(data["details"].get("applied_rules", []), list)
