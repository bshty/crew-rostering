# tests/test_rule_loader.py
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_get_rules():
    resp = client.get("/rules")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # at least the file names you added should appear as rules if valid
