# tests/test_cumulative.py
import datetime
from backend.legality import evaluate_rule_for_payload
from backend.legality import CheckRequest

def make_check_req(duty_minutes=0, flight_minutes=0, landings=0, rest_hours=12.0):
    return CheckRequest(
        duty_minutes=duty_minutes,
        flight_minutes=flight_minutes,
        landings=landings,
        rest_hours=rest_hours
    )

def test_cumulative_rule_exists():
    # Load the json rule and ensure it's present by reading the file
    import json, os
    p = os.path.join(os.getcwd(), "rules", "cumulative_rules.json")
    with open(p, "r", encoding="utf-8") as f:
        rules = json.load(f)
    assert "logic" in rules
    assert isinstance(rules["logic"]["table"], list)

def test_dummy_cumulative_evaluate():
    # Use evaluate_rule_for_payload with a sample cumulative rule row (functionality check)
    import json, os
    p = os.path.join(os.getcwd(), "rules", "cumulative_rules.json")
    with open(p, "r", encoding="utf-8") as f:
        rules = json.load(f)
    # pick first table row (window7)
    row = rules["logic"]["table"][0]
    req = make_check_req(flight_minutes=100)
    # evaluate_rule_for_payload expects a full rule dict; construct a minimal rule with simple logic to pass through
    rule = {
        "id": "TEST_CUM",
        "logic": {
            "type": "simple",
            "value": 1000
        }
    }
    # Should not raise; returns None since simple rule compares flight_minutes <= value
    from backend.legality import evaluate_rule_for_payload
    v = evaluate_rule_for_payload(rule, req)
    assert v is None
