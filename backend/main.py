# backend/main.py
"""
DGCA Rule Engine - FastAPI main file.

Loads DGCA rules from backend/rules, exposes:
- GET  /               -> "Rule Engine Ready!" + rules count
- GET  /rules          -> list rule summaries
- GET  /rules/{id}     -> full rule detail
- POST /rules/reload   -> reload rules from disk
- POST /check          -> legality checker endpoint
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load rule loader
from .load_rules import load_rules_from_folder
try:
    from .load_rules import RuleSpec  # optional
except Exception:
    RuleSpec = None  # type: ignore

# Legality checker
from .legality import router as legality_router

log = logging.getLogger("uvicorn.error")

# Path to backend/rules folder
BASE_DIR = Path(__file__).resolve().parent
RULES_DIR = BASE_DIR / "rules"

# Ensure folder exists
RULES_DIR.mkdir(parents=True, exist_ok=True)


# ---------- RESPONSE MODELS ----------
class RuleSummary(BaseModel):
    id: str
    title: Optional[str] = None
    dgca_reference: Optional[Any] = None
    enabled: Optional[bool] = None
    version: Optional[str] = None


class RuleDetail(BaseModel):
    id: str
    title: Optional[str] = None
    dgca_reference: Optional[Any] = None
    enabled: Optional[bool] = None
    version: Optional[str] = None
    scope: Optional[Dict[str, Any]] = None
    logic: Optional[Dict[str, Any]] = None
    actions: Optional[Dict[str, Any]] = None
    tests: Optional[List[Dict[str, Any]]] = None


# ---------- NORMALIZE LOADER OUTPUT ----------
def _normalize_loaded_rules(loaded: Any, invalid: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rules_map: Dict[str, Any] = {}
    invalid_list: List[Dict[str, Any]] = invalid or []

    # Loader returned dict
    if isinstance(loaded, dict):
        rules_map = loaded.copy()
        return rules_map, invalid_list

    # Loader returned list
    if isinstance(loaded, list):
        for idx, item in enumerate(loaded):
            # Pydantic RuleSpec
            if hasattr(item, "id"):
                rid = getattr(item, "id", f"rule_{idx}")
                rules_map[rid] = item
                continue

            # Plain dict
            if isinstance(item, dict):
                rid = item.get("id") or item.get("title") or f"rule_{idx}"
                rules_map[rid] = item
                continue

            # Filename string
            if isinstance(item, str):
                candidate = Path(item)
                possible = candidate if candidate.exists() else (RULES_DIR / candidate)
                try:
                    raw = json.loads(possible.read_text(encoding="utf-8"))
                    rid = raw.get("id") or possible.stem
                    rules_map[rid] = raw
                except Exception:
                    rules_map[f"file_{idx}"] = {"file": str(possible)}
                continue

            # Unknown shape fallback
            rules_map[f"rule_{idx}"] = item

        return rules_map, invalid_list

    # Single object with id
    if hasattr(loaded, "id"):
        rid = getattr(loaded, "id")
        rules_map[rid] = loaded
        return rules_map, invalid_list

    rules_map["rules_loaded"] = loaded
    return rules_map, invalid_list


# ---------- LIFESPAN STARTUP ----------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        loaded, invalid = load_rules_from_folder(RULES_DIR)
    except Exception as e:
        log.exception("load_rules_from_folder failed: %s", e)
        loaded, invalid = {}, [{"file": "loader_exception", "error": str(e)}]

    rules_map, invalid_list = _normalize_loaded_rules(loaded, invalid)

    app.state.rules = rules_map
    app.state.invalid_rules = invalid_list

    log.info("Rule loader startup: %d valid, %d invalid", len(rules_map), len(invalid_list))

    yield


# Create FastAPI app
try:
    app = FastAPI(title="DGCA Rule Engine (MVP)", lifespan=_lifespan)
except TypeError:
    app = FastAPI(title="DGCA Rule Engine (MVP)")
    
    @app.on_event("startup")
    def _startup_rule_loader():
        try:
            loaded, invalid = load_rules_from_folder(RULES_DIR)
        except Exception as e:
            log.exception("Loader error: %s", e)
            loaded, invalid = {}, [{"file": "loader_exception", "error": str(e)}]

        rules_map, invalid_list = _normalize_loaded_rules(loaded, invalid)
        app.state.rules = rules_map
        app.state.invalid_rules = invalid_list

        log.info("Rule loader startup (fallback): %d valid, %d invalid", len(rules_map), len(invalid_list))


# Include legality checker
app.include_router(legality_router)


# ---------- ROOT ----------
@app.get("/")
def root():
    rules_map = getattr(app.state, "rules", {})
    return {
        "message": "Rule Engine Ready!",
        "rules_loaded": len(rules_map),
    }


# ---------- LIST RULES ----------
@app.get("/rules", response_model=List[RuleSummary])
def get_rules():
    rules_map = getattr(app.state, "rules", {})
    summaries: List[RuleSummary] = []

    def _sort_key(x):
        return getattr(x, "id", str(x))

    for r in sorted(rules_map.values(), key=_sort_key):
        if hasattr(r, "id"):
            summaries.append(RuleSummary(
                id=r.id,
                title=getattr(r, "title", None),
                dgca_reference=getattr(r, "dgca_reference", None),
                enabled=getattr(r, "enabled", None),
                version=getattr(r, "version", None),
            ))
        elif isinstance(r, dict):
            summaries.append(RuleSummary(
                id=r.get("id", "unknown"),
                title=r.get("title"),
                dgca_reference=r.get("dgca_reference"),
                enabled=r.get("enabled"),
                version=r.get("version"),
            ))
        else:
            summaries.append(RuleSummary(id=str(r)))

    return summaries


# ---------- GET RULE DETAIL ----------
@app.get("/rules/{rule_id}", response_model=RuleDetail)
def get_rule_detail(rule_id: str):
    rules_map: Dict[str, Any] = getattr(app.state, "rules", {})
    rule = rules_map.get(rule_id)

    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

    if hasattr(rule, "id"):
        return RuleDetail(
            id=rule.id,
            title=getattr(rule, "title", None),
            dgca_reference=getattr(rule, "dgca_reference", None),
            enabled=getattr(rule, "enabled", None),
            version=getattr(rule, "version", None),
            scope=getattr(rule, "scope", None),
            logic=getattr(rule, "logic", None),
            actions=getattr(rule, "actions", None),
            tests=getattr(rule, "tests", None),
        )

    if isinstance(rule, dict):
        return RuleDetail(
            id=rule.get("id", rule_id),
            title=rule.get("title"),
            dgca_reference=rule.get("dgca_reference"),
            enabled=rule.get("enabled"),
            version=rule.get("version"),
            scope=rule.get("scope"),
            logic=rule.get("logic"),
            actions=rule.get("actions"),
            tests=rule.get("tests"),
        )

    return RuleDetail(id=rule_id)


# ---------- RELOAD RULES ----------
@app.post("/rules/reload")
def reload_rules():
    try:
        loaded, invalid = load_rules_from_folder(RULES_DIR)
    except Exception as e:
        log.exception("reload loader error: %s", e)
        loaded, invalid = {}, [{"file": "loader_exception", "error": str(e)}]

    rules_map, invalid_list = _normalize_loaded_rules(loaded, invalid)
    app.state.rules = rules_map
    app.state.invalid_rules = invalid_list

    return {
        "loaded": len(rules_map),
        "invalid": invalid_list,
    }
