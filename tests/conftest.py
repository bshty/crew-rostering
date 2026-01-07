# tests/conftest.py
# Ensure project root is on sys.path so `import backend` works reliably in pytest.
import sys
from pathlib import Path

# Resolve project root as the parent of the tests folder
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # put project root at front so local packages take precedence
    sys.path.insert(0, str(ROOT))
