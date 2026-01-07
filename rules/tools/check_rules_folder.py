import json
from pathlib import Path

# Path to your rules directory
p = Path("rules")

print("Looking for folder at:", p.resolve())

if not p.exists():
    # try backend/rules instead
    p = Path("backend") / "rules"
    print("Trying fallback:", p.resolve())

if not p.exists():
    print("ERROR: Could not find any rules folder.")
    quit()

files = list(p.glob("*.json"))
print("Found JSON files:", [f.name for f in files])

for f in files:
    try:
        json.loads(f.read_text(encoding="utf-8"))
        print(f"{f.name} → OK")
    except Exception as e:
        print(f"{f.name} → ERROR:", e)
