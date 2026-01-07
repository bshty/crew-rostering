# backend/validate_rules.py
# Copy this file into your project and run:
#   python backend/validate_rules.py
# It will validate every .json in backend/rules and print errors with file/line/col.

import json
from pathlib import Path
import sys
import traceback

RULES_DIR = Path(__file__).resolve().parent.joinpath("rules")

def validate_json_file(p: Path):
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception as e:
        print(f"{p.name}: ERROR reading file: {e}")
        return False
    try:
        json.loads(txt)
        print(f"{p.name}: OK")
        return True
    except json.JSONDecodeError as e:
        # Print friendly error with line/col and snippet
        print(f"{p.name}: JSON parse error: {e.msg} (line {e.lineno}, col {e.colno})")
        # show a small snippet around the error location
        lines = txt.splitlines()
        ln = e.lineno - 1
        start = max(0, ln-2)
        end = min(len(lines), ln+2)
        print("---- context ----")
        for i in range(start, end):
            marker = ">>" if i==ln else "  "
            print(f"{marker} {i+1:4d}: {lines[i]}")
        print("-----------------")
        return False
    except Exception:
        print(f"{p.name}: Unexpected error while parsing JSON:")
        traceback.print_exc()
        return False

def main():
    if not RULES_DIR.exists():
        print("Rules folder not found:", RULES_DIR.resolve())
        sys.exit(1)
    files = sorted(RULES_DIR.glob("*.json"))
    if not files:
        print("No .json files found in:", RULES_DIR.resolve())
        sys.exit(0)
    ok_count = 0
    bad_count = 0
    for f in files:
        ok = validate_json_file(f)
        if ok:
            ok_count += 1
        else:
            bad_count += 1
    print(f"\nSummary: {ok_count} OK, {bad_count} INVALID ({len(files)} files checked)")
    if bad_count:
        sys.exit(2)

if __name__ == "__main__":
    main()
