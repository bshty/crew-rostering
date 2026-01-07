# backend/load_rules.py
"""
Robust rule loader for DGCA rule JSON files.

Provides:
 - VALID_RULES: validated rule objects (RuleSpec)
 - INVALID_REPORTS: parse/validation errors
 - MERGED_RULES: merged plain dict from all raw JSON
 - RULES: alias to MERGED_RULES for legality engine
"""
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import datetime

# ---- pydantic compatibility (v1 & v2) ----
try:
    from pydantic import BaseModel, ValidationError, field_validator as _field_validator  # v2
    IS_PYDANTIC_V2 = True
except Exception:
    from pydantic import BaseModel, ValidationError, validator as _field_validator  # v1
    IS_PYDANTIC_V2 = False

log = logging.getLogger("rule_loader")
log.setLevel(logging.INFO)


# ---------------------------------------------------------
# RuleSpec Model
# ---------------------------------------------------------
class RuleSpec(BaseModel):
    id: str
    title: str
    logic: Dict[str, Any]
    dgca_reference: Optional[Any] = None
    enabled: bool = True
    version: Optional[str] = None
    notes: Optional[Dict[str, Any]] = None

    @_field_validator("id")
    def _id_not_empty(cls, v):
        if not v or not isinstance(v, str) or v.strip() == "":
            raise ValueError("id must be non-empty string")
        return v


# ---------------------------------------------------------
# Helper: Extract rule objects from mixed JSON formats
# ---------------------------------------------------------
def _iter_rule_objects_from_raw(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []

    # List of rules
    if isinstance(raw, list):
        return raw

    # Dict cases
    if isinstance(raw, dict):
        # wrapper { "rules": [ ... ] }
        if "rules" in raw and isinstance(raw["rules"], list):
            return raw["rules"]

        # dict-of-rule-objects keyed by id
        values = list(raw.values())
        if (
            values
            and all(isinstance(v, dict) for v in values)
            and any(("id" in v or "logic" in v or "title" in v) for v in values)
        ):
            out = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    vr = dict(v)
                    if "id" not in vr:
                        vr["id"] = k
                    out.append(vr)
            return out

        # Single rule
        return [raw]

    return []


# ---------------------------------------------------------
# Deep-merge with array concat and dedupe
# ---------------------------------------------------------
def deep_merge_with_array_concat(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dicts. When encountering lists on same key, concat and dedupe by repr.
    Otherwise behave like _deep_merge_dicts.
    """
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_with_array_concat(out[k], v)
        elif k in out and isinstance(out[k], list) and isinstance(v, list):
            combined = out[k] + v
            seen = set()
            dedup = []
            for item in combined:
                try:
                    key = json.dumps(item, sort_keys=True)
                except Exception:
                    key = repr(item)
                if key not in seen:
                    dedup.append(item)
                    seen.add(key)
            out[k] = dedup
        else:
            out[k] = v
    return out


# ---------------------------------------------------------
# Small normalization helpers
# ---------------------------------------------------------
def hours_to_minutes_safe(h: Any) -> Optional[int]:
    try:
        if h is None:
            return None
        # numeric hours
        if isinstance(h, (int, float)):
            return int(round(float(h) * 60))
        s = str(h).strip()
        if s == "":
            return None
        # accept strings like "8.5", "8.5h", "8:30"
        if ":" in s:
            parts = s.split(":")
            hh = int(parts[0])
            mm = int(parts[1]) if len(parts) > 1 else 0
            return hh * 60 + mm
        s2 = s.lower().replace("hours", "").replace("hour", "").replace("h", "").strip()
        val = float(s2)
        return int(round(val * 60))
    except Exception:
        return None


def normalize_rest_facility_key(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = str(s).strip().lower()
    if "ulr" in s2 and ("bunk" in s2 or "ulr_bunk" in s2):
        return "ulr_bunk_approved"
    if "ulr" in s2:
        return "ulr"
    if "bunk" in s2:
        return "bunk"
    if "isolat" in s2 or "isolated" in s2:
        return "isolated_rest_seat"
    if "basic" in s2:
        return "basic_rest_seat"
    return s2.replace(" ", "_")


def flight_time_range_to_hours(rng: Optional[str]) -> Optional[int]:
    """
    Convert 'up_to_8_hours' or 'over_8_to_9_hours' into an integer hour upper bound for canonical rows.
    Uses same mapping/heuristic as legality.lookup (kept simple).
    """
    if not rng:
        return None
    s = str(rng).lower()
    if "up_to_8" in s or "up_to_8_hours" in s or "up_to_8" in s:
        return 8
    if "up_to_7" in s or "up_to_7_hours" in s:
        return 7
    if "over_8_to_9" in s or "over_8_to_9_hours" in s:
        return 9
    if "over_9_to_10" in s:
        return 10
    if "over_8_to_11" in s:
        return 11
    if "over_11_to_14" in s:
        return 14
    if "ulr_above_14" in s:
        return 24
    # fallback: attempt to parse digits
    import re
    m = re.search(r'(\d{1,2})', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


# ---------------------------------------------------------
# Map a single rule object into the canonical merged_plain structure
# ---------------------------------------------------------
def map_rule_to_merged(merged: Dict[str, Any], rule_obj: Dict[str, Any], source_file: str) -> None:
    """
    Map different 'logic.type' shapes into canonical keys that the legality engine expects.
    This mutates 'merged' in place.
    """
    rid = rule_obj.get("id") or "<no-id>"
    logic = rule_obj.get("logic", {}) or {}
    ltype = logic.get("type")
    # attach provenance
    rule_obj_copy = dict(rule_obj)
    rule_obj_copy["_source_file"] = source_file

    # ensure meta exists
    meta = merged.setdefault("meta", {})
    meta.setdefault("loaded_from_files", [])
    if source_file not in meta["loaded_from_files"]:
        meta["loaded_from_files"].append(source_file)

    def _is_encroachment_table_name(name: Optional[str], tbl: Optional[Dict[str, Any]]) -> bool:
        if not name and not tbl:
            return False
        name_s = (str(name or "")).lower()
        # check presence of keywords in name or notes/description
        if any(k in name_s for k in ("encroach", "encroachment", "example", "night_encroach")):
            return True
        notes_txt = ""
        if isinstance(tbl, dict):
            notes_txt = str(tbl.get("notes") or tbl.get("note") or tbl.get("description") or "")
        if any(k in notes_txt.lower() for k in ("encroach", "encroachment", "example", "night_encroach")):
            return True
        return False

    if ltype == "fdp_tables":
        # keep raw table under fdp_tables by id/title
        key = rule_obj.get("id", f"fdp_{len(merged.get('fdp_tables', {}))}")
        merged.setdefault("fdp_tables", {})
        # ensure we attach _source_file and mark example rows when appropriate
        raw_copy = dict(rule_obj_copy)
        raw_copy.setdefault("_source_file", source_file)
        raw_copy["_is_encroachment_example"] = _is_encroachment_table_name(key, raw_copy) or bool(raw_copy.get("is_encroachment_example"))
        merged["fdp_tables"][key] = raw_copy

        # Additionally try to canonicalize into crew-specific canonical 'fdp_table' lists
        tables = logic.get("tables") or {}
        # tables may be dict-of-tables or list
        if isinstance(tables, dict):
            iter_tables = tables.items()
        elif isinstance(tables, list):
            iter_tables = [(f"{rid}_{i}", t) for i, t in enumerate(tables)]
        else:
            iter_tables = []

        for tname, t in iter_tables:
            if not isinstance(t, dict):
                continue
            crew_type = (t.get("crew_type") or t.get("table_type") or "").strip().lower()
            # keep the raw t under fdp_tables as well and mark if example
            t_copy = dict(t)
            t_copy.setdefault("_source_file", source_file)
            t_copy["_is_encroachment_example"] = _is_encroachment_table_name(tname, t_copy) or bool(t_copy.get("is_encroachment_example"))
            merged["fdp_tables"].setdefault(tname, t_copy)
            # canonicalize into merged[crew_type]["fdp_table"]
            if crew_type in ("two_pilot", "single_pilot", "augmented", "cabin"):
                canon_list = merged.setdefault(crew_type, {}).setdefault("fdp_table", [])
                # bands
                bands = t.get("bands") or []
                for b in bands:
                    # map landings_to_max_fdp_minutes if present
                    lmap = b.get("landings_to_max_fdp_minutes") or b.get("landings_map") or {}
                    if isinstance(lmap, dict) and lmap:
                        ft_hours = flight_time_range_to_hours(b.get("flight_time_range") or b.get("flight_time_range"))
                        for lands_k, mm_v in lmap.items():
                            try:
                                lands = int(lands_k)
                            except Exception:
                                try:
                                    lands = int(float(lands_k))
                                except Exception:
                                    continue
                            try:
                                mm = int(mm_v) if mm_v is not None else None
                            except Exception:
                                try:
                                    mm = int(float(mm_v))
                                except Exception:
                                    mm = None
                            row = {
                                "max_flight_time_hours": ft_hours,
                                "max_landings": lands,
                                "max_fdp_minutes": mm,
                                "__source_table": tname,
                                "__source_file": source_file,
                                "__is_example": t_copy.get("_is_encroachment_example", False)
                            }
                            canon_list.append(row)
                    else:
                        # fallback single-band row
                        ft_hours = flight_time_range_to_hours(b.get("flight_time_range"))
                        mm = None
                        if b.get("max_fdp_minutes") is not None:
                            try:
                                mm = int(b.get("max_fdp_minutes"))
                            except Exception:
                                try:
                                    mm = int(float(b.get("max_fdp_minutes")))
                                except Exception:
                                    mm = None
                        ml = b.get("max_landings")
                        if ml is not None:
                            try:
                                ml = int(ml)
                            except Exception:
                                try:
                                    ml = int(float(ml))
                                except Exception:
                                    ml = None
                        row = {
                            "max_flight_time_hours": ft_hours,
                            "max_landings": ml,
                            "max_fdp_minutes": mm,
                            "__source_table": tname,
                            "__source_file": source_file,
                            "__is_example": t_copy.get("_is_encroachment_example", False)
                        }
                        canon_list.append(row)

            else:
                # If no explicit crew type, keep as generic
                merged.setdefault("fdp_tables", {}).setdefault(tname, t_copy)

    elif ltype == "rest":
        # canonical rest rules saved under 'rest_rules'
        rblock = rule_obj.get("rest_rules") or rule_obj.get("logic") or rule_obj
        merged["rest_rules"] = rblock
        merged.setdefault("rules_index", {})[rid] = {"file": source_file, "type": "rest", "raw": rule_obj_copy}

    elif ltype == "positioning":
        pr = rule_obj.get("logic") or rule_obj
        # try to extract the configuration keys commonly used by engine
        positioning_cfg = {}
        # map known fields if present
        for k in ("count_as_duty_default", "count_positioning_as_landing_if_minutes_over", "count_positioning_as_duty"):
            if isinstance(pr, dict) and k in pr:
                positioning_cfg[k] = pr[k]
        # if not present look inside rule notes/defaults
        merged["positioning_rules"] = positioning_cfg or pr
        merged.setdefault("rules_index", {})[rid] = {"file": source_file, "type": "positioning", "raw": rule_obj_copy}

    elif ltype == "cumulative_windows":
        table = rule_obj.get("logic", {}).get("table") or rule_obj.get("table") or []
        cl = merged.setdefault("cumulative_limits", {})
        for row in table:
            try:
                w = int(row.get("window_days"))
                metric = row.get("metric")
                if metric and w:
                    cl[f"{w}d"] = cl.get(f"{w}d", {})
                    cl[f"{w}d"][metric] = int(row.get("max_minutes"))
            except Exception:
                continue
        merged.setdefault("rules_index", {})[rid] = {"file": source_file, "type": "cumulative", "raw": rule_obj_copy}

    elif ltype == "standby":
        merged["standby_rules"] = rule_obj.get("logic", {}) or rule_obj
        merged.setdefault("rules_index", {})[rid] = {"file": source_file, "type": "standby", "raw": rule_obj_copy}

    else:
        # generic fallback - store under rules_index
        merged.setdefault("rules_index", {})[rid] = {"file": source_file, "type": ltype or "unknown", "raw": rule_obj_copy}


# ---------------------------------------------------------
# Main Loader
# ---------------------------------------------------------
def load_rules_from_folder(folder: Path) -> Tuple[Dict[str, RuleSpec], List[Dict[str, Any]]]:
    """
    Loads all rule JSON files from folder
    Returns:
        (VALID_RULES, INVALID_REPORTS)
    AND stores MERGED_RULES (dict) + RULES (alias) at module level.
    """
    valid: Dict[str, RuleSpec] = {}
    invalid: List[Dict[str, Any]] = []
    merged_plain: Dict[str, Any] = {}

    folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        log.warning(f"Rules folder does not exist: {folder}")
        globals()["VALID_RULES"] = valid
        globals()["INVALID_REPORTS"] = invalid
        globals()["MERGED_RULES"] = merged_plain
        globals()["RULES"] = merged_plain
        return valid, invalid

    # Load *.json files deterministically
    for f in sorted(folder.glob("*.json")):
        fname = f.name
        try:
            text = f.read_text(encoding="utf-8")
        except Exception as e:
            invalid.append({"file": fname, "error": f"read_error: {e}"})
            log.error(f"Failed to read {fname}: {e}")
            continue

        try:
            parsed = json.loads(text)
        except Exception as e:
            invalid.append({"file": fname, "error": f"json_parse_error: {e}"})
            log.error(f"JSON parse error in {fname}: {e}")
            continue

        # Merge raw contents (keep raw merged for traceability)
        if isinstance(parsed, dict):
            merged_plain = deep_merge_with_array_concat(merged_plain, parsed)

        # Extract rule objects and validate via RuleSpec
        rule_objs = _iter_rule_objects_from_raw(parsed)

        for idx, raw_rule in enumerate(rule_objs):
            try:
                if IS_PYDANTIC_V2 and hasattr(RuleSpec, "model_validate"):
                    r = RuleSpec.model_validate(raw_rule)
                else:
                    r = RuleSpec.parse_obj(raw_rule)

                if r.enabled:
                    if r.id in valid:
                        invalid.append({"file": fname, "index": idx, "error": f"duplicate rule id: {r.id}", "existing_from": valid[r.id].__dict__.get("_source_file") if hasattr(valid[r.id], "__dict__") else None})
                        log.error(f"Duplicate rule id {r.id} in {fname}")
                    else:
                        # attach provenance to the pydantic object for trace
                        try:
                            # pydantic v1/v2: safe attach attribute
                            setattr(r, "_source_file", fname)
                        except Exception:
                            pass
                        valid[r.id] = r
                        log.info(f"Loaded rule {r.id} from {fname}")

                        # map the validated raw_rule into canonical merged_plain shapes
                        try:
                            map_rule_to_merged(merged_plain, raw_rule, fname)
                        except Exception:
                            log.exception("Failed mapping rule to merged for %s from %s", raw_rule.get("id"), fname)

            except ValidationError as e:
                invalid.append(
                    {"file": fname, "index": idx, "error": f"validation_error: {e.json() if hasattr(e, 'json') else str(e)}"}
                )
            except Exception as e:
                invalid.append({"file": fname, "index": idx, "error": f"unexpected_error: {e}"})
                log.exception(f"Unexpected error validating rule in {fname}: {e}")

    # Post-process normalization pass on merged_plain
    try:
        # ensure meta entries
        meta = merged_plain.setdefault("meta", {})
        meta.setdefault("generated_at", datetime.datetime.utcnow().isoformat() + "Z")
        meta.setdefault("source_files", meta.get("loaded_from_files", []))

        # convert any max_fdp_hours -> max_fdp_minutes in canonical lists
        for crew_key in ("two_pilot", "single_pilot", "augmented", "cabin"):
            if merged_plain.get(crew_key) and isinstance(merged_plain[crew_key], dict):
                fdplist = merged_plain[crew_key].get("fdp_table", [])
                normalized = []
                for row in fdplist:
                    r = dict(row)
                    # normalize max_fdp_minutes
                    if r.get("max_fdp_minutes") is None and r.get("max_fdp_hours") is not None:
                        mm = hours_to_minutes_safe(r.get("max_fdp_hours"))
                        if mm is not None:
                            r["max_fdp_minutes"] = mm
                    # ensure numeric types for known keys
                    if r.get("max_fdp_minutes") is not None:
                        try:
                            r["max_fdp_minutes"] = int(r["max_fdp_minutes"])
                        except Exception:
                            r["max_fdp_minutes"] = None
                    if r.get("max_landings") is not None:
                        try:
                            r["max_landings"] = int(r["max_landings"])
                        except Exception:
                            r["max_landings"] = None
                    if r.get("max_flight_time_hours") is not None:
                        try:
                            r["max_flight_time_hours"] = int(r["max_flight_time_hours"])
                        except Exception:
                            try:
                                r["max_flight_time_hours"] = int(float(r["max_flight_time_hours"]))
                            except Exception:
                                r["max_flight_time_hours"] = None
                    normalized.append(r)
                merged_plain[crew_key]["fdp_table"] = normalized

        # normalize augmented rest_facility map keys
        if merged_plain.get("fdp_tables") and isinstance(merged_plain["fdp_tables"], dict):
            for tname, t in merged_plain["fdp_tables"].items():
                # if rows exist
                rows = t.get("rows") or t.get("bands") or []
                # try to canonicalize rest_fac maps that might exist in rows
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, dict):
                            rf = r.get("max_fdp_by_rest_facility_minutes") or r.get("max_fdp_by_rest_facility_minutes_map")
                            if isinstance(rf, dict):
                                newmap = {}
                                for k, v in rf.items():
                                    nk = normalize_rest_facility_key(k)
                                    try:
                                        newmap[nk] = int(v) if v is not None else None
                                    except Exception:
                                        try:
                                            newmap[nk] = int(float(v))
                                        except Exception:
                                            newmap[nk] = None
                                # store normalized map back
                                r["max_fdp_by_rest_facility_minutes"] = newmap

        # canonical positioning rules
        if "positioning_rules" in merged_plain:
            pr = merged_plain["positioning_rules"] or {}
            # ensure sensible defaults
            if not isinstance(pr, dict):
                merged_plain["positioning_rules"] = {"count_as_duty_default": False}
            else:
                pr.setdefault("count_as_duty_default", pr.get("count_as_duty_default", False))
                pr.setdefault("count_positioning_as_landing_if_minutes_over", pr.get("count_positioning_as_landing_if_minutes_over", 30))

        # canonical rest_rules numeric conversion (hours)
        if "rest_rules" in merged_plain and isinstance(merged_plain["rest_rules"], dict):
            rr = merged_plain["rest_rules"]
            # defaults
            defaults = rr.get("defaults", {})
            for k in ("min_rest_before_fdp_hours", "min_rest_other_duties_hours"):
                try:
                    rr.setdefault("defaults", {})
                    if k in defaults:
                        rr["defaults"][k] = float(defaults[k])
                    else:
                        rr["defaults"].setdefault(k, 10.0)
                except Exception:
                    rr["defaults"][k] = 10.0
            # rest_after_long_fdp
            raf = rr.get("rest_after_long_fdp", {})
            try:
                raf.setdefault("threshold_fdp_hours", float(raf.get("threshold_fdp_hours", 10.0)))
                raf.setdefault("min_rest_if_prev_fdp_over_threshold_hours", float(raf.get("min_rest_if_prev_fdp_over_threshold_hours", 12.0)))
            except Exception:
                rr["rest_after_long_fdp"] = {"threshold_fdp_hours": 10.0, "min_rest_if_prev_fdp_over_threshold_hours": 12.0}

        # quick validation checks appended to invalid if critical items missing
        try:
            if "meta" not in merged_plain or not merged_plain["meta"].get("source_files"):
                invalid.append({"stage": "postprocess", "error": "no source_files discovered in meta"})
            # ensure 7d cumulative if user provided cumulative_limits in any rule
            if merged_plain.get("cumulative_limits"):
                if "7d" not in merged_plain["cumulative_limits"]:
                    # optional, but warn
                    log.info("cumulative_limits present but 7d entry missing")
        except Exception:
            pass

    except Exception as e:
        log.exception("Post-process normalization failed: %s", e)
        invalid.append({"stage": "postprocess", "error": str(e)})

    # Store module level
    globals()["VALID_RULES"] = valid
    globals()["INVALID_REPORTS"] = invalid
    globals()["MERGED_RULES"] = merged_plain
    globals()["RULES"] = merged_plain  # alias

    log.info(
        f"Rule loader summary: {len(valid)} valid rules, {len(invalid)} invalid, merged keys: {len(merged_plain)}"
    )

    return valid, invalid


# ---------------------------------------------------------
# Eager load on import
# ---------------------------------------------------------
RULES_DIR = Path(__file__).parent / "rules"

try:
    VALID_RULES, INVALID_REPORTS = load_rules_from_folder(RULES_DIR)
except Exception as e:
    log.exception(f"Failed to eager-load rules: {e}")
    VALID_RULES = {}
    INVALID_REPORTS = []
    MERGED_RULES = {}
    RULES = {}

__all__ = [
    "load_rules_from_folder",
    "VALID_RULES",
    "INVALID_REPORTS",
    "MERGED_RULES",
    "RULES",
    "RuleSpec",
]
