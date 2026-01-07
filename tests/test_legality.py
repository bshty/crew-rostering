# tests/bench_legality.py
import time
import random
import datetime
import importlib
import backend.legality as legality

# small deterministic ruleset
BENCH_RULES = {
    "meta": {"version": "bench-v1"},
    "two_pilot": {"fdp_table": [{"max_flight_time_hours": 9, "max_fdp_minutes": 600, "max_landings": 3}]},
    "rest_rules": {"defaults": {"min_rest_before_fdp_hours": 10}},
    "positioning_rules": {"count_as_duty_default": True, "count_positioning_as_landing_if_minutes_over": 45},
}

def inject_rules(r):
    legality.RULES = r
    importlib.reload(legality)

def gen_random_payload():
    now = datetime.datetime.datetime.utcnow().replace(hour=6, minute=0, second=0, microsecond=0)
    start = now + datetime.timedelta(days=random.randint(-2,2), hours=random.randint(-10,10))
    duty_mins = random.randint(60, 10*60)
    end = start + datetime.timedelta(minutes=duty_mins)
    flight_mins = random.randint(0, min(duty_mins, 8*60))
    payload = legality.CheckRequest(
        crew_id="BENCH",
        duty_start=start.isoformat(),
        duty_end=end.isoformat(),
        duty_minutes=f"{duty_mins//60:02d}:{duty_mins%60:02d}",
        flight_minutes=f"{flight_mins//60:02d}:{flight_mins%60:02d}",
        landings=random.randint(0,4),
        rest_hours=f"{random.randint(8,14):02d}:00",
        context={}
    )
    return payload

def run_stress(n=5000):
    inject_rules(BENCH_RULES)
    rule = {"id": "fdp_tables", "logic": {"type": "fdp_tables", "tables": [{"source": "bench", "bands":[{"band_hours":9,"max_fdp_minutes":600,"max_landings":3}]}]}}
    start = time.time()
    for i in range(n):
        p = gen_random_payload()
        legality.normalize_payload_times(p)
        legality.evaluate_rule_for_payload(rule, p, {"fdp_tables": rule})
    dur = time.time() - start
    print(f"Ran {n} checks in {dur:.2f}s — {n/dur:.1f} checks/sec (avg {dur/n*1000:.2f} ms/check)")

if __name__ == "__main__":
    run_stress(2000)
