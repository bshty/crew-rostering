#!/usr/bin/env python3
"""
Robust converter: finds a FDP_TABLES JSON in common locations and converts it
to the engine-friendly format (band_hours / max_landings / max_fdp_minutes).
Save as scripts/convert_fdp.py and run from your project root.
"""

import json
from pathlib import Path
import sys

CANDIDATE_NAMES = ["FDP_TABLES*", "FDP_TABLES_DGCA*"]
CANDIDATE_DIRS = [
    Path("backend") / "rules",
    Path("backend"),
    Path("."),           # project root
    Path("/mnt/data"),   # linux-like upload location (if present)
]

OUT = Path("backend") / "rules" / "FDP_TABLES_DGCA_V2_converted.json"

RANGE_TO_HOURS = {
    "up_to_8_hours": 8,
    "over_8_to_9_hours": 9,
    "over_9_to_10_hours": 10,
    "over_8_hours": 9,
    "up_to_7_hours": 7,
    "over_7_to_8_hours": 8
}

def find_src():
    looked = []
    # check each dir for files matching candidate names
    for d in CANDIDATE_DIRS:
        try:
            d = Path(d)
            looked.append(str(d.resolve()) if d.exists() else str(d))
            if not d.exists():
                continue
            for pattern in CANDIDATE_NAMES:
                for p in d.glob(pattern):
                    if p.is_file():
                        return p, looked
        except Exception as e:
            looked.append(f"error_checking_{d}: {e}")
    # also do a shallow search in the repository root (not recursive deep to avoid slowness)
    root = Path(".")
    try:
        for pattern in CANDIDATE_NAMES:
            for p in root.glob("**/" + pattern):
                if p.is_file():
                    looked.append("recursive_search")
                    return p, looked
    except Exception as e:
        looked.append(f"recursive_search_error: {e}")

    return None, looked

def to_int_minutes(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(float(v))
    except Exception:
        try:
            s = str(v)
            if ":" in s:
                h,m = s.split(":")
                return int(h)*60 + int(m)
        except Exception:
            pass
    return None

def main():
    src, looked = find_src()
    if not src:
        print("ERROR: Could not find a FDP_TABLES file. I looked in:")
        for p in looked:
            print(" -", p)
        print("\nPlease copy your FDP JSON into one of those folders (or tell me the exact path).")
        sys.exit(2)

    print("Found FDP file at:", src)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:
        print("ERROR: Failed to read/parse JSON at", src, ":", e)
        sys.exit(3)

    # build converted object
    converted = {
        "id": "FDP_TABLES_DGCA_V2_CONVERTED",
        "title": data.get("title", "Converted FDP Tables"),
        "dgca_reference": data.get("dgca_reference"),
        "logic": {"type": "fdp_tables", "tables": []}
    }

    orig_tables = data.get("logic", {}).get("tables", {}) or data.get("tables") or {}
    # handle dict-of-tables and list-of-tables shapes
    if isinstance(orig_tables, dict):
        items = orig_tables.items()
    elif isinstance(orig_tables, list):
        items = [(i, t) for i, t in enumerate(orig_tables)]
    else:
        items = []

    for name, table_obj in items:
        new_table = {"source": str(name), "bands": []}
        bands = table_obj.get("bands", []) if isinstance(table_obj, dict) else []
        for band in bands:
            ftr = band.get("flight_time_range")
            lmap = band.get("landings_to_max_fdp_minutes") or band.get("landings_map")
            if ftr and isinstance(lmap, dict):
                bh = RANGE_TO_HOURS.get(ftr, None)
                for lk, v in lmap.items():
                    try:
                        ml = int(lk)
                    except Exception:
                        continue
                    mm = to_int_minutes(v)
                    if mm is None:
                        continue
                    new_table["bands"].append({"band_hours": bh, "max_landings": ml, "max_fdp_minutes": mm})
            else:
                # try numeric fields if present
                bh = band.get("band_hours") or band.get("max_flight_time_hours")
                ml = band.get("max_landings") or band.get("landings")
                mm = band.get("max_fdp_minutes")
                if mm is None and band.get("max_fdp_hours") is not None:
                    mm = to_int_minutes(band.get("max_fdp_hours") * 60)
                if mm is None:
                    mm = to_int_minutes(band.get("max_fdp_minutes")) 
                if mm is None:
                    continue
                new_table["bands"].append({"band_hours": bh, "max_landings": ml, "max_fdp_minutes": mm})

        if new_table["bands"]:
            converted["logic"]["tables"].append(new_table)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    print("Wrote converted file to:", OUT.resolve())

if __name__ == "__main__":
    main()
