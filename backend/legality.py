# backend/legality.py
"""
Enhanced legality checker for DGCA rule engine (audit-grade) - Slice B (audit-ready).

Key guarantees implemented in this file:
 - All time quantities in response/traces are serialized as HH:MM strings (Option A).
 - Time parsing is timezone-aware when possible; naive inputs fall back to UTC.
 - Full audit trace completeness with deterministic provenance (ruleset_version, hash, source_files).
 - WOCL reduction rules explicit and documented in trace.
 - Positioning handling integrated and visible in traces.
 - Performance-conscious: precompiled regexes, deterministic sorts.
 - Versioning & provenance metadata included in result.details.meta.

Notes:
 - Internally calculations use minutes for correctness; traces / outputs convert minutes -> "HH:MM".
 - If you want local WOCL evaluation, set payload.context["acclimatized_tz"] to an IANA zone (e.g. "Asia/Kolkata").
"""

from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import datetime
import logging
import math
import traceback
import re
import hashlib
import json

# Timezone & parsing helpers (optional runtime deps)
try:
    from dateutil import parser as _du_parser  # pip install python-dateutil
except Exception:
    _du_parser = None

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# Attempt to import merged RULES (user loader). If absent, RULES = {}
try:
    from backend.load_rules import RULES  # type: ignore
except Exception:
    RULES = {}

log = logging.getLogger("uvicorn.error")
router = APIRouter()

# ---------------- CONFIG ----------------
DEFAULT_SELECTION_STRATEGY = "option_b"  # option_b = exact-landing -> nearest-above -> fallback
GLOBAL_STRICT_MODE = False  # if True, always use most-conservative option (option_a)
# DGCA-normalized encroachment minutes when any WOCL contact exists (Option B)
DGCA_NORMALIZED_ENCROACH_MINUTES = 210
# WOCL interval (local): 02:00 - 05:59 inclusive (hours)
WOCL_START_HOUR = 2
WOCL_END_HOUR = 5
# ----------------------------------------

# ---------- Request / Response Models ----------
class DutyHistoryItem(BaseModel):
    date: Optional[str] = None
    duty_minutes: Optional[int] = None
    flight_minutes: Optional[int] = None


class CheckRequest(BaseModel):
    crew_id: Optional[str] = None
    duty_start: Optional[str] = None
    duty_end: Optional[str] = None
    duty_minutes: Any
    flight_minutes: Any
    landings: int
    rest_hours: Any
    tz_crossings: int = 0
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[DutyHistoryItem]] = None


class Violation(BaseModel):
    rule_id: str
    message: str
    dgca_reference: Optional[str] = None


class CheckResult(BaseModel):
    legal: bool
    violations: List[Violation]
    details: Optional[Dict[str, Any]] = None


# ---------- Utilities ----------
# Precompiled regexes (performance)
_TIME_HHMM = re.compile(r'^\s*(\d{1,2}):(\d{1,2})\s*$')
_TIME_H_M = re.compile(
    r'(?:(\d+(?:\.\d+)?)\s*h(?:ours?)?)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?)?',
    re.IGNORECASE,
)
_TIME_MIN_ONLY = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*(m|min|mins|minutes)\s*$', re.IGNORECASE)
_TIME_H_DECIMAL = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*h\s*$', re.IGNORECASE)


def ensure_dt_with_tz(dt: Optional[datetime.datetime], tz_name: Optional[str]) -> Optional[datetime.datetime]:
    """
    Ensure dt is timezone-aware. If dt.tzinfo is None, attach tz_name if provided,
    else attach UTC. Returns a tz-aware datetime or None if dt is None.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt
    # attach provided tz
    if tz_name and ZoneInfo is not None:
        try:
            return dt.replace(tzinfo=ZoneInfo(tz_name))
        except Exception:
            return dt.replace(tzinfo=datetime.timezone.utc)
    # fallback: attach UTC
    return dt.replace(tzinfo=datetime.timezone.utc)


def parse_iso(dtstr: Optional[str], tz_hint: Optional[str] = None) -> Optional[datetime.datetime]:
    """
    Robust ISO parsing returning timezone-aware datetime.
    If the input lacks tzinfo, attach tz_hint (IANA name like 'Asia/Kolkata') if provided,
    else attach UTC.

    Accepts:
      - ISO strings with offsets
      - naive 'YYYY-MM-DDTHH:MM:SS' (tz_hint used)
      - epoch numeric strings (seconds)
      - datetime objects
    """
    if not dtstr:
        return None
    if isinstance(dtstr, datetime.datetime):
        return ensure_dt_with_tz(dtstr, tz_hint)
    s = str(dtstr).strip()
    # Try dateutil (most forgiving)
    if _du_parser is not None:
        try:
            dt = _du_parser.isoparse(s)
            return ensure_dt_with_tz(dt, tz_hint)
        except Exception:
            pass
    # Fallback: fromisoformat (handles YYYY-MM-DDTHH:MM:SS and offsets)
    try:
        tmp = s
        if tmp.endswith("Z"):
            tmp = tmp[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(tmp)
        return ensure_dt_with_tz(dt, tz_hint)
    except Exception:
        pass
    # Last resort: numeric epoch seconds
    try:
        v = float(s)
        dt = datetime.datetime.utcfromtimestamp(v).replace(tzinfo=datetime.timezone.utc)
        return ensure_dt_with_tz(dt, tz_hint)
    except Exception:
        return None


def minutes_between(dt1: datetime.datetime, dt2: datetime.datetime) -> int:
    delta = dt2 - dt1
    return int(delta.total_seconds() / 60)


def minutes_to_hhmm(minutes: Optional[int]) -> Optional[str]:
    """
    Convert integer minutes to HH:MM string.
    - Accepts negative (exceeded) -> prefix '-' then HH:MM part
    - Returns None if input is None
    """
    if minutes is None:
        return None
    try:
        m = int(minutes)
    except Exception:
        return None
    sign = "-" if m < 0 else ""
    m = abs(m)
    hh = m // 60
    mm = m % 60
    return f"{sign}{hh:02d}:{mm:02d}"


def hours_to_hhmm(hours_float: Optional[float]) -> Optional[str]:
    """
    Convert hours (float) to HH:MM string for human-friendly rest outputs.
    """
    if hours_float is None:
        return None
    try:
        mins = int(round(float(hours_float) * 60))
    except Exception:
        return None
    return minutes_to_hhmm(mins)


# ---------- WOCL helpers ----------
def get_wocl_window_for_zone(acclimatized_tz: Optional[str]) -> Tuple[int, int]:
    """
    Returns WOCL window start/end hours. Kept as function to allow per-zone logic later.
    """
    return (WOCL_START_HOUR, WOCL_END_HOUR)


def minutes_of_overlap_with_wocl(duty_start: datetime.datetime, duty_end: datetime.datetime, acclimatized_tz: Optional[str] = None) -> int:
    """
    Compute overlap minutes between a duty interval and WOCL (local 02:00-05:59).
    - Converts duty_start/duty_end to the local (acclimatized) timezone before evaluating.
    - Handles duties crossing midnight by checking previous, same and next day's WOCL.
    - Returns integer minutes of overlap (0 if none).
    """
    if duty_start is None or duty_end is None:
        return 0

    # ensure tz-aware and attach tz_hint if naive
    ds = ensure_dt_with_tz(duty_start, acclimatized_tz)
    de = ensure_dt_with_tz(duty_end, acclimatized_tz)

    # convert both to acclimatized tz if zoneinfo available & tz provided
    if acclimatized_tz and ZoneInfo is not None:
        try:
            target = ZoneInfo(acclimatized_tz)
            ds_local = ds.astimezone(target)
            de_local = de.astimezone(target)
        except Exception:
            ds_local = ds.astimezone(datetime.timezone.utc)
            de_local = de.astimezone(datetime.timezone.utc)
    else:
        ds_local = ds
        de_local = de

    if de_local <= ds_local:
        return 0

    def wocl_interval_for_day_local(day_dt: datetime.datetime) -> Tuple[datetime.datetime, datetime.datetime]:
        start = day_dt.replace(hour=WOCL_START_HOUR, minute=0, second=0, microsecond=0)
        end = day_dt.replace(hour=WOCL_END_HOUR, minute=59, second=0, microsecond=0)
        return start, end

    base_day_local = ds_local.replace(hour=0, minute=0, second=0, microsecond=0)
    total = 0
    for delta_days in (-1, 0, 1):
        day_local = base_day_local + datetime.timedelta(days=delta_days)
        ws, we = wocl_interval_for_day_local(day_local)
        overlap_start = max(ds_local, ws)
        overlap_end = min(de_local, we)
        if overlap_end > overlap_start:
            total += int((overlap_end - overlap_start).total_seconds() / 60)
    return total


# ---------- Flexible time parsing ----------
def parse_time_to_minutes(value: Any) -> int:
    """
    Accepts HH:MM, '8.5h', '90m', numeric hours (<=24) or numeric minutes (>24).
    Returns minutes as integer.
    """
    if value is None:
        return 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(round(value))
    s = str(value).strip()
    if s == "":
        return 0
    m = _TIME_HHMM.match(s)
    if m:
        h = int(m.group(1)); mm = int(m.group(2))
        return max(0, h * 60 + mm)
    m = _TIME_H_M.search(s)
    if m:
        hours_part = m.group(1); mins_part = m.group(2)
        total = 0
        if hours_part:
            try:
                total += int(float(hours_part) * 60)
            except Exception:
                pass
        if mins_part:
            try:
                total += int(float(mins_part))
            except Exception:
                pass
        if total > 0:
            return total
    m = _TIME_MIN_ONLY.match(s)
    if m:
        try:
            return int(round(float(m.group(1))))
        except Exception:
            pass
    m = _TIME_H_DECIMAL.match(s)
    if m:
        try:
            return int(round(float(m.group(1)) * 60))
        except Exception:
            pass
    try:
        val = float(s)
        if val <= 24:
            return int(round(val * 60))
        else:
            return int(round(val))
    except Exception:
        pass
    return 0


def parse_hours(value: Any) -> float:
    """
    Parses a rest-hours-like input to float hours.
    Accepts HH:MM, numeric hours, or minutes.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        nv = float(value)
        if nv > 24:
            return nv / 60.0
        return nv
    s = str(value).strip()
    if s == "":
        return 0.0
    m = _TIME_HHMM.match(s)
    if m:
        h = int(m.group(1)); mm = int(m.group(2))
        return h + (mm / 60.0)
    m = _TIME_H_M.search(s)
    if m:
        hours_part = m.group(1); mins_part = m.group(2)
        total = 0.0
        if hours_part:
            try:
                total += float(hours_part)
            except Exception:
                pass
        if mins_part:
            try:
                total += float(mins_part) / 60.0
            except Exception:
                pass
        if total > 0:
            return total
    m = _TIME_H_DECIMAL.match(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    try:
        val = float(s)
        if val > 24:
            return val / 60.0
        return val
    except Exception:
        return 0.0


# ---------- Helper: normalize rest facility key ----------
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
    return s2


# ---------- Helper: ruleset provenance ----------
def compute_ruleset_provenance(rules_obj: Any) -> Dict[str, Any]:
    """
    Compute a deterministic provenance object for the loaded ruleset.
    Includes version, hash, loaded_at timestamp, and known source_files if present.
    """
    try:
        serial = json.dumps(rules_obj, sort_keys=True, default=str)
        h = hashlib.sha256(serial.encode("utf-8")).hexdigest()
        version = None
        source_files = []
        if isinstance(rules_obj, dict):
            version = rules_obj.get("meta", {}).get("version") or rules_obj.get("version")
            source_files = rules_obj.get("meta", {}).get("source_files") or rules_obj.get("source_files") or []
        return {
            "ruleset_hash_sha256": h,
            "ruleset_version": version,
            "loaded_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source_files": list(source_files) if isinstance(source_files, (list, tuple)) else []
        }
    except Exception:
        return {"ruleset_hash_sha256": None, "ruleset_version": None, "loaded_at": None, "source_files": []}


# ---------- FDP table lookup ----------
def lookup_fdp_from_tables(rule_tables: Any, payload: CheckRequest, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Lookup FDP allowed minutes from provided tables and produce an audit-grade result.
    Returned dict contains minute fields (internal) but caller will convert minutes -> HH:MM strings for output.
    Key return fields (minutes): allowed_fdp_minutes, encroach_minutes_used, encroach_actual_minutes, positioning_added_minutes
    Additional trace/provenance fields included.
    """
    try:
        if not rule_tables:
            return None

        # normalize rule_tables
        normalized_tables: List[Dict[str, Any]] = []
        if isinstance(rule_tables, dict):
            for name, tbl in rule_tables.items():
                if isinstance(tbl, dict):
                    entry = dict(tbl)
                    entry["_name"] = name
                    normalized_tables.append(entry)
        elif isinstance(rule_tables, list):
            for t in rule_tables:
                if isinstance(t, dict):
                    normalized_tables.append(t)
        else:
            return None

        RANGE_TO_HOURS = {
            "up_to_8_hours": 8, "over_8_to_9_hours": 9, "over_9_to_10_hours": 10,
            "up_to_7_hours": 7, "over_7_to_8_hours": 8, "over_8_hours": 9,
            "over_8_to_11_hours": 11, "over_11_to_14_hours": 14, "ulr_above_14_hours": 24
        }

        fm = int(payload.flight_minutes or 0)
        flight_hours = math.ceil(fm / 60) if fm is not None else None
        landings = int(payload.landings or 0)

        # compute actual WOCL overlap -- use acclimatized TZ when available
        tz_hint = None
        if isinstance(payload.context, dict):
            tz_hint = payload.context.get("acclimatized_tz")
        encroach_actual = 0
        ds = parse_iso(payload.duty_start, tz_hint)
        de = parse_iso(payload.duty_end, tz_hint)
        if ds and de:
            encroach_actual = minutes_of_overlap_with_wocl(ds, de, acclimatized_tz=tz_hint)
        encroach_minutes = int(DGCA_NORMALIZED_ENCROACH_MINUTES) if encroach_actual > 0 else 0

        # normalize crew_type preference
        def _normalize_crew_type_local(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            s2 = str(s).strip().lower()
            if s2 in ("2", "2p", "two", "two_pilot", "two-pilot", "twopilot"):
                return "two_pilot"
            if s2 in ("1", "1p", "single", "single_pilot", "single-pilot", "singlepilot"):
                return "single_pilot"
            if "augment" in s2 or "aug" in s2:
                return "augmented"
            if "cabin" in s2:
                return "cabin"
            return s2

        crew_type_raw = None
        try:
            crew_type_raw = (context or {}).get("crew_type")
            if crew_type_raw is None and isinstance(payload, CheckRequest) and payload.context:
                crew_type_raw = payload.context.get("crew_type")
        except Exception:
            crew_type_raw = None
        crew_type = _normalize_crew_type_local(crew_type_raw) or "two_pilot"

        # provenance baseline for deterministic selection
        provenance = {
            "selection_strategy": (context or {}).get("selection_strategy") or (payload.context or {}).get("selection_strategy") or DEFAULT_SELECTION_STRATEGY,
            "global_strict_mode": bool(GLOBAL_STRICT_MODE),
            "crew_type_requested": crew_type,
            "dgca_normalized_encroach_minutes": DGCA_NORMALIZED_ENCROACH_MINUTES,
            "acclimatized_tz": tz_hint
        }

        candidates: List[Dict[str, Any]] = []

        def add_candidate(source_name: str, source_page: Any, band_obj: Dict[str, Any], bh_val, ml_val, mm_val, table_raw=None, extras=None):
            try:
                mm_int = int(mm_val) if mm_val is not None else None
            except Exception:
                try:
                    mm_int = int(float(mm_val))
                except Exception:
                    mm_int = None
            try:
                bh_int = int(bh_val) if bh_val is not None else None
            except Exception:
                bh_int = None
            try:
                ml_int = int(ml_val) if ml_val is not None else None
            except Exception:
                ml_int = None
            if mm_int is not None:
                entry = {
                    "band_hours": bh_int,
                    "max_landings": ml_int,
                    "max_fdp_minutes": mm_int,
                    "source": (source_name or "").lower(),
                    "source_page": source_page,
                    "band": band_obj,
                    "table_raw": table_raw
                }
                if extras:
                    entry.update(extras)
                candidates.append(entry)

        # build candidate list
        for tbl in normalized_tables:
            tbl_name = (tbl.get("_name") or tbl.get("source") or tbl.get("title") or "unnamed")
            source_page = tbl.get("source_page") or tbl.get("source")
            bands = tbl.get("bands")
            if isinstance(bands, list):
                for b in bands:
                    if not isinstance(b, dict):
                        continue
                    ftr = b.get("flight_time_range")
                    lmap = b.get("landings_to_max_fdp_minutes") or b.get("landings_map")
                    if ftr and isinstance(lmap, dict):
                        bh = RANGE_TO_HOURS.get(ftr)
                        for lk, vv in lmap.items():
                            try:
                                ml = int(lk); mm = int(vv)
                            except Exception:
                                try:
                                    ml = int(float(lk)); mm = int(float(vv))
                                except Exception:
                                    continue
                            add_candidate(tbl_name, source_page, b, bh, ml, mm, tbl)
                        continue
                    bh = b.get("band_hours") or b.get("max_flight_time_hours") or b.get("flight_time_hours")
                    ml = b.get("max_landings")
                    mm = b.get("max_fdp_minutes")
                    if mm is None and b.get("max_fdp_hours") is not None:
                        try:
                            mm = int(round(float(b["max_fdp_hours"]) * 60))
                        except Exception:
                            mm = None
                    add_candidate(tbl_name, source_page, b, bh, ml, mm, tbl)

            rows = tbl.get("rows")
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    row_max_landings = r.get("max_landings") if r.get("max_landings") is not None else r.get("landings")
                    rf = r.get("max_fdp_by_rest_facility_minutes") or r.get("max_fdp_minutes") or r.get("max_fdp_minutes_map")
                    if isinstance(rf, dict):
                        rest_map: Dict[str, int] = {}
                        for k, v in rf.items():
                            if k is None:
                                continue
                            try:
                                val = int(v) if v is not None else None
                            except Exception:
                                try:
                                    val = int(float(v))
                                except Exception:
                                    val = None
                            if val is not None:
                                rest_map[str(k).lower()] = val
                        canonical_mm = min(rest_map.values()) if rest_map else None
                        extras = {
                            "rest_facility_map": rest_map,
                            "crew_category": r.get("crew_category"),
                            "max_flight_time_hours": r.get("max_flight_time_hours")
                        }
                        add_candidate(tbl_name, source_page, r, r.get("max_flight_time_hours"), row_max_landings, canonical_mm, tbl, extras=extras)
                        continue
                    if r.get("max_fdp_minutes") is not None:
                        try:
                            mm2 = int(r["max_fdp_minutes"])
                        except Exception:
                            try:
                                mm2 = int(float(r["max_fdp_minutes"]))
                            except Exception:
                                continue
                        add_candidate(tbl_name, source_page, r, r.get("max_flight_time_hours"), row_max_landings, mm2, tbl)

        if not candidates:
            return None

        # filter out encroachment-example rows if no encroachment
        if encroach_minutes == 0:
            filtered = []
            for c in candidates:
                name = str(c.get("source") or "").lower()
                band = c.get("band") or {}
                band_text = str(band.get("example") or band.get("note") or band.get("notes") or band.get("description") or "").lower()
                if ("encroach" in name and "example" in name) or ("night" in name and "encroach" in name) or ("encroach" in band_text and "example" in band_text):
                    continue
                filtered.append(c)
            if filtered:
                candidates = filtered

        # apply constraints
        valid = []
        for c in candidates:
            bh = c.get("band_hours")
            ml = c.get("max_landings")
            if bh is not None and flight_hours is not None and flight_hours > bh and crew_type != "augmented":
                continue
            if ml is not None and landings > ml:
                continue
            valid.append(c)
        candidate_pool = valid if valid else candidates.copy()

        # crew_type preference filter
        if crew_type:
            pref = []
            for c in candidate_pool:
                name = str(c.get("source") or "").lower()
                band = c.get("band") or {}
                band_crew_cat = str(band.get("crew_category") or band.get("crew_type") or band.get("crew_category_name") or c.get("crew_category") or "").lower()
                if crew_type in name or (band_crew_cat and crew_type in band_crew_cat):
                    pref.append(c)
            if pref:
                candidate_pool = pref

        # deterministic sorting helpers
        def candidate_sort_key(c):
            return (
                c.get("max_landings") if c.get("max_landings") is not None else 9999,
                c.get("max_fdp_minutes") if c.get("max_fdp_minutes") is not None else 999999,
                c.get("band_hours") if c.get("band_hours") is not None else 9999,
                str(c.get("source") or "")
            )

        # ---------------- Augmented selection (if requested) ----------------
        augmented_candidate_trace = []
        if crew_type == "augmented":
            rest_fac_raw = (context or {}).get("rest_facility") or (payload.context or {}).get("rest_facility")
            rest_fac_key = normalize_rest_facility_key(rest_fac_raw)
            ctx_crew_cat = (context or {}).get("crew_category") or (payload.context or {}).get("crew_category")
            ctx_crew_cat = str(ctx_crew_cat).lower() if ctx_crew_cat else None

            aug_items = []
            for c in candidate_pool:
                band = c.get("band") or {}
                crew_cat = str(c.get("crew_category") or band.get("crew_category") or band.get("crew_type") or "").lower()
                rest_map = c.get("rest_facility_map") or {}
                mfth = c.get("band_hours") or c.get("max_flight_time_hours") or band.get("max_flight_time_hours") if isinstance(band, dict) else None
                try:
                    mfth_v = int(mfth) if mfth is not None else None
                except Exception:
                    try:
                        mfth_v = int(float(mfth))
                    except Exception:
                        mfth_v = None

                allowed_by_requested = None
                if rest_map and rest_fac_key:
                    for k, v in rest_map.items():
                        if rest_fac_key and rest_fac_key in k:
                            allowed_by_requested = v
                            break
                    if allowed_by_requested is None and "ulr" in (rest_fac_key or ""):
                        for k, v in rest_map.items():
                            if "ulr" in k:
                                allowed_by_requested = v
                                break

                canonical_allowed = min(rest_map.values()) if rest_map else c.get("max_fdp_minutes")

                aug_items.append({
                    "candidate": c,
                    "crew_cat": crew_cat,
                    "mfth": mfth_v,
                    "rest_map_keys": list(rest_map.keys()) if isinstance(rest_map, dict) else [],
                    "rest_map": rest_map,
                    "allowed_by_requested": allowed_by_requested,
                    "canonical_allowed": canonical_allowed
                })

            # scoring
            scored = []
            for item in aug_items:
                score = 0
                if ctx_crew_cat and item["crew_cat"] and ctx_crew_cat == item["crew_cat"]:
                    score += 2000
                if rest_fac_key and any(rest_fac_key in k for k in item["rest_map_keys"]):
                    score += 1500
                if rest_fac_key and "ulr" in rest_fac_key:
                    if any("ulr" in k for k in item["rest_map_keys"]):
                        score += 1750
                if "4_crew_ulr" in (item["crew_cat"] or "") or "4_crew_ulr" in " ".join(item["rest_map_keys"]):
                    score += 1200
                if "4_crew" in (item["crew_cat"] or ""):
                    score += 300
                if "3_crew" in (item["crew_cat"] or ""):
                    score += 100

                surplus = None
                if item["mfth"] is not None and flight_hours is not None:
                    if item["mfth"] >= flight_hours:
                        surplus = item["mfth"] - flight_hours
                    else:
                        score -= 5000
                        surplus = 9999
                else:
                    surplus = 9999

                canonical_allowed = item["canonical_allowed"] or 0
                scored.append((-score, surplus, -canonical_allowed, item))

            scored.sort(key=lambda x: (x[0], x[1], x[2]))
            for sc in scored:
                it = sc[3]
                augmented_candidate_trace.append({
                    "source": it["candidate"].get("source"),
                    "crew_cat": it["crew_cat"],
                    "mfth": it["mfth"],
                    "rest_map_keys": it["rest_map_keys"],
                    "allowed_by_requested": it["allowed_by_requested"],
                    "canonical_allowed": it["canonical_allowed"],
                    "score": -sc[0],
                    "surplus": sc[1]
                })

            chosen_candidate = scored[0][3]["candidate"] if scored else None

            if chosen_candidate:
                chosen_rest_map = chosen_candidate.get("rest_facility_map") or {}
                chosen_allowed = None
                if rest_fac_key and isinstance(chosen_rest_map, dict):
                    for k, v in chosen_rest_map.items():
                        if rest_fac_key in k:
                            chosen_allowed = v
                            break
                    if chosen_allowed is None and "ulr" in (rest_fac_key or ""):
                        for k, v in chosen_rest_map.items():
                            if "ulr" in k:
                                chosen_allowed = v
                                break
                if chosen_allowed is None and isinstance(chosen_rest_map, dict) and chosen_rest_map:
                    strict_local = GLOBAL_STRICT_MODE or (isinstance(payload.context, dict) and payload.context.get("strict_mode") is True)
                    chosen_allowed = min(chosen_rest_map.values()) if strict_local else max(chosen_rest_map.values())
                if chosen_allowed is None:
                    try:
                        chosen_allowed = int(chosen_candidate.get("max_fdp_minutes"))
                    except Exception:
                        chosen_allowed = None

                # conservative cap from peers
                applied_conservative_cap_from = None
                augmented_peer_min_allowed = None
                try:
                    peer_allowed_values: List[int] = []
                    for peer in candidates:
                        peer_src = (peer.get("source") or "").lower()
                        if "aug" in peer_src or "augmented" in peer_src:
                            continue
                        p_ml = peer.get("max_landings")
                        if p_ml is not None and landings > p_ml:
                            continue
                        p_allowed = peer.get("max_fdp_minutes")
                        if isinstance(p_allowed, int):
                            peer_allowed_values.append(int(p_allowed))
                        rest_map = peer.get("rest_facility_map") or {}
                        if isinstance(rest_map, dict) and rest_map:
                            try:
                                vals = [int(v) for v in rest_map.values() if v is not None]
                                if vals:
                                    peer_allowed_values.append(min(vals))
                            except Exception:
                                pass
                    if peer_allowed_values:
                        augmented_peer_min_allowed = min(peer_allowed_values)
                        if chosen_allowed is not None and augmented_peer_min_allowed is not None and augmented_peer_min_allowed < chosen_allowed:
                            applied_conservative_cap_from = "peer_non_augmented_min_across_landings"
                            chosen_allowed = min(chosen_allowed, augmented_peer_min_allowed)
                except Exception:
                    log.exception("Error while computing conservative cap for augmented candidate")

                # positioning metadata
                positioning_applied = bool(payload.context.get("_positioning_applied")) if isinstance(payload.context, dict) else False
                positioning_added_minutes = int(payload.context.get("_positioning_added_minutes") or 0) if isinstance(payload.context, dict) else 0
                positioning_counted_as_landing = bool(payload.context.get("_positioning_counted_as_landing")) if isinstance(payload.context, dict) else False

                # build result (minute fields internal)
                result = {
                    "allowed_fdp_minutes": int(chosen_allowed) if chosen_allowed is not None else None,
                    "source": chosen_candidate.get("source"),
                    "source_page": chosen_candidate.get("source_page"),
                    "band": chosen_candidate.get("band"),
                    "encroach_minutes_used": encroach_minutes,
                    "encroach_actual_minutes": encroach_actual,
                    "chosen_table_name": chosen_candidate.get("source"),
                    "crew_type_selected": "augmented",
                    "augmented_candidate_trace": augmented_candidate_trace,
                    "augmented_peer_min_allowed": int(augmented_peer_min_allowed) if augmented_peer_min_allowed is not None else None,
                    "applied_conservative_cap_from": applied_conservative_cap_from,
                    "positioning_applied": positioning_applied,
                    "positioning_added_minutes": positioning_added_minutes,
                    "positioning_counted_as_landing": positioning_counted_as_landing,
                    "selected_table_is_example": bool(chosen_candidate.get("table_raw", {}).get("is_encroachment_example", False)) if chosen_candidate else False,
                    "base_from_source": chosen_candidate.get("source") if chosen_candidate else None,
                    "provenance": provenance
                }
                return result

        # ---------------- general selection (non-aug or fallback) ----------------
        exact_landings = [c for c in candidate_pool if c.get("max_landings") is not None and c.get("max_landings") == landings]
        if exact_landings:
            candidate_pool = exact_landings
        else:
            candidate_pool.sort(key=lambda c: (c.get("max_landings") if c.get("max_landings") is not None else 9999))

        # selection strategy resolution
        selection_strategy = DEFAULT_SELECTION_STRATEGY
        if isinstance(payload.context, dict) and payload.context.get("selection_strategy"):
            selection_strategy = payload.context.get("selection_strategy")
        if GLOBAL_STRICT_MODE:
            selection_strategy = "option_a"
        if isinstance(payload.context, dict) and payload.context.get("strict_mode") is True:
            selection_strategy = "option_a"
        provenance["selection_strategy"] = selection_strategy

        chosen = None
        if selection_strategy == "option_a":
            candidate_pool.sort(key=lambda c: ((c.get("max_fdp_minutes") if c.get("max_fdp_minutes") is not None else 999999),
                                              (c.get("band_hours") if c.get("band_hours") is not None else 9999),
                                              str(c.get("source") or "")))
            chosen = candidate_pool[0]
        else:
            if len(candidate_pool) == 1:
                chosen = candidate_pool[0]
            else:
                candidate_pool.sort(key=candidate_sort_key)
                chosen = candidate_pool[0]

        if not chosen:
            return None

        # chosen is available - determine example / base selection
        chosen_table_raw = chosen.get("table_raw") or {}
        chosen_is_example = False
        try:
            chosen_is_example = bool(
                chosen_table_raw.get("is_encroachment_example")
                or (isinstance(chosen_table_raw.get("_name"), str)
                    and "encroach" in chosen_table_raw.get("_name").lower()
                    and ("example" in str(chosen_table_raw.get("notes") or "").lower()
                         or "example" in str(chosen_table_raw.get("description") or "").lower()))
            )
        except Exception:
            chosen_is_example = False

        base_allowed = chosen.get("max_fdp_minutes")
        base_from_source = chosen.get("source")

        # When encroachment exists and chosen is an example, prefer canonical fallback
        prefer_canonical = False
        try:
            if isinstance(payload.context, dict) and payload.context.get("use_canonical_base_for_example"):
                prefer_canonical = True
        except Exception:
            prefer_canonical = False
        if encroach_minutes > 0:
            prefer_canonical = True

        if chosen_is_example and encroach_minutes > 0 and prefer_canonical:
            src = (chosen.get("source") or "").lower() if chosen.get("source") else ""
            base_name_guess = base_from_source or src
            for suffix in ("_night_encroachment_examples", "_encroachment_examples", "_encroach_examples", "_night_encroach_examples", "_encroachment"):
                if base_name_guess.endswith(suffix):
                    base_name_guess = base_name_guess[: -len(suffix)]
                    break

            fallback_candidate = None
            for c in candidates:
                c_src = (c.get("source") or "").lower()
                c_table_raw = c.get("table_raw") or {}
                if c_table_raw.get("is_encroachment_example"):
                    continue
                if base_name_guess and base_name_guess in c_src:
                    fallback_candidate = c
                    break
            if fallback_candidate is None:
                for c in candidates:
                    c_table_raw = c.get("table_raw") or {}
                    if c_table_raw.get("is_encroachment_example"):
                        continue
                    try:
                        chosen_band = chosen.get("band") or {}
                        c_band = c.get("band") or {}
                        if chosen_band and c_band and chosen_band.get("flight_time_range") and chosen_band.get("flight_time_range") == c_band.get("flight_time_range"):
                            fallback_candidate = c
                            break
                    except Exception:
                        continue
            if fallback_candidate:
                base_allowed = fallback_candidate.get("max_fdp_minutes") or base_allowed
                base_from_source = fallback_candidate.get("source") or base_from_source

        positioning_applied = bool(payload.context.get("_positioning_applied")) if isinstance(payload.context, dict) else False
        positioning_added_minutes = int(payload.context.get("_positioning_added_minutes") or 0) if isinstance(payload.context, dict) else 0
        positioning_counted_as_landing = bool(payload.context.get("_positioning_counted_as_landing")) if isinstance(payload.context, dict) else False

        # prepare deterministic provenance for chosen row
        chosen_prov = {
            "chosen_source": chosen.get("source"),
            "chosen_source_page": chosen.get("source_page"),
            "chosen_max_fdp_minutes": chosen.get("max_fdp_minutes"),
            "chosen_max_landings": chosen.get("max_landings"),
            "chosen_band_hours": chosen.get("band_hours"),
            "chosen_table_name": chosen.get("source"),
            "chosen_is_example": chosen_is_example,
            "provenance_base": provenance
        }

        result = {
            "allowed_fdp_minutes": int(base_allowed) if base_allowed is not None else None,
            "source": chosen.get("source"),
            "source_page": chosen.get("source_page"),
            "band": chosen.get("band"),
            "encroach_minutes_used": encroach_minutes,
            "encroach_actual_minutes": encroach_actual,
            "chosen_table_name": chosen.get("source"),
            "crew_type_selected": crew_type,
            "selected_table_is_example": chosen_is_example,
            "base_from_source": base_from_source,
            "positioning_applied": positioning_applied,
            "positioning_added_minutes": positioning_added_minutes,
            "positioning_counted_as_landing": positioning_counted_as_landing,
            "chosen_provenance": chosen_prov,
            "provenance": provenance
        }
        return result

    except Exception as e:
        log.exception("Error in lookup_fdp_from_tables: %s", e)
        return None


# ---------- Rest requirement checker ----------
def check_rest_requirements(
    previous_duty_hours: float,
    provided_rest_hours: float,
    reference_dt_iso: str,
    time_zone_difference_hours: float = 0.0,
    wocl_encroachment_minutes: int = 0,
    crew_augmented: bool = False,
    positioning_precedes_fdp: bool = False,
    split_duty_extension_minutes: int = 0,
    transport_time_minutes: int = 0,
    fdp_was_extended_hours: float = 0.0,
    ruleset: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Structured rest_rules evaluation.
    Returns required_rest_hours (float), provided_rest_hours and violations; details numeric/internal;
    caller will convert to HH:MM for outputs.
    """
    try:
        if not isinstance(ruleset, dict):
            fallback = RULES if isinstance(RULES, dict) else {}
            ruleset = fallback

        rules = ruleset.get("rest_rules", {}) or {}
        defaults = rules.get("defaults", {}) or {}
        tz_over = rules.get("time_zone_overrides", {}) or {}
        rest_after_long = rules.get("rest_after_long_fdp", {}) or {}
        wocl_rules = rules.get("wocl_handling", {}) or {}
        augmented = rules.get("augmented_crew", {}) or {}
        split = rules.get("split_duty", {}) or {}
        transport = rules.get("transport_time_handling", {}) or {}
        unforeseen = rules.get("unforeseen_extensions", {}) or {}

        baseline_min = defaults.get("min_rest_before_fdp_hours",
                                    defaults.get("min_rest_other_duties_hours", 10))
        try:
            baseline_min = float(baseline_min)
        except Exception:
            baseline_min = 10.0

        tz_required = None
        try:
            tz_diff_val = float(time_zone_difference_hours or 0.0)
        except Exception:
            tz_diff_val = 0.0

        if tz_diff_val and tz_diff_val > 3 and tz_diff_val <= 7:
            tz_required = tz_over.get("crossing_more_than_3_up_to_7_tz_hours")
        elif tz_diff_val and tz_diff_val > 7:
            tz_required = tz_over.get("crossing_more_than_7_tz_hours")
        if tz_required is not None:
            try:
                tz_required_f = float(tz_required)
                baseline_min = max(baseline_min, tz_required_f)
            except Exception:
                pass

        try:
            prev_duty_val = float(previous_duty_hours or 0.0)
        except Exception:
            prev_duty_val = 0.0
        required_rest = max(prev_duty_val, baseline_min)

        try:
            long_threshold = float(rest_after_long.get("threshold_fdp_hours", 10))
        except Exception:
            long_threshold = 10.0

        try:
            if prev_duty_val >= float(long_threshold):
                required_rest = max(required_rest, float(rest_after_long.get("min_rest_if_prev_fdp_over_threshold_hours", 12)))
                if rest_after_long.get("use_previous_duty_if_longer", True):
                    required_rest = max(required_rest, prev_duty_val)
        except Exception:
            pass

        if crew_augmented:
            try:
                required_rest = max(required_rest, float(augmented.get("min_rest_hours_for_augmented", 14)))
            except Exception:
                required_rest = max(required_rest, 14.0)

        try:
            if split.get("apply_extension_to_rest", True) and split_duty_extension_minutes:
                add_hours = float(split_duty_extension_minutes) / 60.0
                required_rest += add_hours
        except Exception:
            pass

        try:
            if wocl_rules.get("add_encroachment_minutes_to_rest") and wocl_encroachment_minutes:
                required_rest += (float(wocl_encroachment_minutes) / 60.0)
        except Exception:
            pass

        try:
            threshold = int(transport.get("transport_exclusion_threshold_minutes", 30))
        except Exception:
            threshold = 30
        try:
            if transport_time_minutes and transport_time_minutes > threshold:
                extra_minutes = int(transport_time_minutes) - threshold
                multiplier = transport.get("if_transport_more_than_threshold_add_to_rest_multiplier", 2)
                try:
                    multiplier = float(multiplier)
                except Exception:
                    multiplier = 2.0
                required_rest += (extra_minutes * multiplier) / 60.0
        except Exception:
            pass

        try:
            if fdp_was_extended_hours and unforeseen:
                if float(fdp_was_extended_hours) <= 1.0:
                    required_rest += float(unforeseen.get("fdp_extension_up_to_1h_rest_increment_hours", 2))
                else:
                    required_rest += float(unforeseen.get("fdp_extension_above_1h_rest_increment_hours", 4))
        except Exception:
            pass

        try:
            provided = float(provided_rest_hours or 0.0)
        except Exception:
            provided = 0.0
        violations = {}
        if provided < required_rest:
            violations["insufficient_rest"] = {"required_hours": round(required_rest, 2), "provided_hours": provided}

        details = {
            "baseline_min_hours": baseline_min,
            "prev_duty_hours": prev_duty_val,
            "tz_diff_hours": tz_diff_val,
            "wocl_encroachment_minutes": wocl_encroachment_minutes,
            "split_extension_minutes": split_duty_extension_minutes,
            "transport_time_minutes": transport_time_minutes,
            "fdp_was_extended_hours": fdp_was_extended_hours,
            "crew_augmented": crew_augmented
        }

        return {
            "required_rest_hours": round(required_rest, 2),
            "provided_rest_hours": provided,
            "violations": violations,
            "details": details
        }

    except Exception as e:
        log.exception("Unexpected error inside check_rest_requirements: %s", e)
        return {
            "required_rest_hours": 0.0,
            "provided_rest_hours": float(provided_rest_hours or 0.0),
            "violations": {"error": str(e)},
            "details": {}
        }


# ---------- Rule evaluation core ----------
def evaluate_rule_for_payload(rule: Dict[str, Any], payload: CheckRequest, app_rules_state: Dict[str, Any]) -> Optional[Violation]:
    """
    Evaluate a single rule against payload. Writes trace data into payload.context:
      - _last_fdp_trace (HH:MM fields)
      - _last_cumulative_trace (HH:MM fields)
      - _last_split_extension_hhmm (string)
    """
    try:
        rid = rule.get("id", rule.get("title", "<no-id>"))
        logic = rule.get("logic", {}) or {}
        ltype = logic.get("type")
        dgca_ref = rule.get("dgca_reference")
        if payload.context is None:
            payload.context = {}
        ctx = payload.context

        merged_rules = app_rules_state or RULES or {}
        ruleset_prov = compute_ruleset_provenance(RULES)

        if ltype == "fdp_tables":
            tables = logic.get("tables", []) or []
            if not tables:
                # attempt to build from merged_rules two_pilot/single_pilot
                tables = []
                tp = merged_rules.get("two_pilot")
                if tp and isinstance(tp, dict) and "fdp_table" in tp:
                    bands = []
                    for r in tp["fdp_table"]:
                        bh = r.get("max_flight_time_hours")
                        mm = None
                        if r.get("max_fdp_minutes") is not None:
                            try:
                                mm = int(r.get("max_fdp_minutes"))
                            except Exception:
                                mm = None
                        elif r.get("max_fdp_hours") is not None:
                            try:
                                mm = int(round(float(r.get("max_fdp_hours")) * 60))
                            except Exception:
                                mm = None
                        ml = r.get("max_landings")
                        bands.append({"band_hours": bh, "max_fdp_minutes": mm, "max_landings": ml})
                    tables.append({"source": "two_pilot_standard", "source_page": None, "bands": bands})
                sp = merged_rules.get("single_pilot")
                if sp and isinstance(sp, dict) and "fdp_table" in sp:
                    bands = []
                    for r in sp["fdp_table"]:
                        bh = r.get("max_flight_time_hours")
                        mm = None
                        if r.get("max_fdp_minutes") is not None:
                            try:
                                mm = int(r.get("max_fdp_minutes"))
                            except Exception:
                                mm = None
                        elif r.get("max_fdp_hours") is not None:
                            try:
                                mm = int(round(float(r.get("max_fdp_hours")) * 60))
                            except Exception:
                                mm = None
                        ml = r.get("max_landings")
                        bands.append({"band_hours": bh, "max_fdp_minutes": mm, "max_landings": ml})
                    tables.append({"source": "single_pilot_standard", "source_page": None, "bands": bands})

            lookup = lookup_fdp_from_tables(tables, payload, ctx)
            # build fdp trace (in minutes) then convert all minute fields to HH:MM strings for output
            trace_minutes = {
                "tables_present": bool(tables),
                "tables_count": len(tables),
                "selected_table": None,
                "selected_band": None,
                "flight_minutes": int(payload.flight_minutes or 0),
                "flight_hours_ceil": (math.ceil((payload.flight_minutes or 0) / 60) if (payload.flight_minutes is not None) else None),
                "landings": int(payload.landings or 0),
                "allowed_base_minutes": None,
                "wocl_reduction_minutes": 0,
                "effective_allowed_minutes": None,
                "payload_duty_minutes": int(payload.duty_minutes or 0),
                "encroach_minutes": 0,
                "encroach_actual_minutes": 0,
                "crew_type_selected": None,
                "chosen_table_name": None,
                "provenance": ruleset_prov
            }

            if not lookup:
                # still expose positioning flags and provenance in trace (HH:MM)
                payload.context["_last_fdp_trace"] = {
                    "tables_present": trace_minutes["tables_present"],
                    "tables_count": trace_minutes["tables_count"],
                    "flight_time": minutes_to_hhmm(trace_minutes["flight_minutes"]),
                    "landings": trace_minutes["landings"],
                    "payload_duty": minutes_to_hhmm(trace_minutes["payload_duty_minutes"]),
                    "provenance": trace_minutes["provenance"],
                    "positioning_applied": bool(payload.context.get("_positioning_applied", False)),
                    "positioning_added": minutes_to_hhmm(int(payload.context.get("_positioning_added_minutes", 0))),
                    "positioning_counted_as_landing": bool(payload.context.get("_positioning_counted_as_landing", False))
                }
                return None

            # fill trace_minutes from lookup
            allowed = lookup.get("allowed_fdp_minutes")
            trace_minutes["allowed_base_minutes"] = allowed
            trace_minutes["selected_table"] = lookup.get("source")
            trace_minutes["selected_band"] = lookup.get("band")
            trace_minutes["encroach_minutes"] = lookup.get("encroach_minutes_used", 0)
            trace_minutes["encroach_actual_minutes"] = lookup.get("encroach_actual_minutes", 0)
            trace_minutes["crew_type_selected"] = lookup.get("crew_type_selected", None)
            trace_minutes["chosen_table_name"] = lookup.get("chosen_table_name", lookup.get("source"))
            if lookup.get("augmented_candidate_trace"):
                trace_minutes["augmented_candidate_trace"] = lookup.get("augmented_candidate_trace")
            # provenance from lookup
            if lookup.get("provenance"):
                trace_minutes["lookup_provenance"] = lookup.get("provenance")
            if lookup.get("chosen_provenance"):
                trace_minutes["chosen_provenance"] = lookup.get("chosen_provenance")

            # always include positioning metadata
            trace_minutes["positioning_applied"] = bool(payload.context.get("_positioning_applied", False))
            trace_minutes["positioning_added_minutes"] = int(payload.context.get("_positioning_added_minutes") or 0)
            trace_minutes["positioning_counted_as_landing"] = bool(payload.context.get("_positioning_counted_as_landing") or False)

            # compute WOCL reduction logic (minutes) using acclimatized tz if present
            tz_hint = payload.context.get("acclimatized_tz") if isinstance(payload.context, dict) else None
            ds = parse_iso(payload.duty_start, tz_hint)
            de = parse_iso(payload.duty_end, tz_hint)
            reduction_minutes = 0
            wocl_rule_triggered = None
            reduction_reason = None
            if ds and de:
                encroach_minutes = trace_minutes.get("encroach_minutes", 0) or 0
                encroach_actual = trace_minutes.get("encroach_actual_minutes", 0) or 0
                w_start, w_end = get_wocl_window_for_zone(tz_hint)
                # start-in-WOCL behaviour
                # use local hour comparison (ds, de are tz-aware)
                ds_local = ensure_dt_with_tz(ds, tz_hint)
                de_local = ensure_dt_with_tz(de, tz_hint)
                ds_local = ds_local.astimezone(ZoneInfo(tz_hint)) if (tz_hint and ZoneInfo is not None) else ds_local
                de_local = de_local.astimezone(ZoneInfo(tz_hint)) if (tz_hint and ZoneInfo is not None) else de_local

                if w_start <= ds_local.hour <= w_end:
                    reduction_minutes = min(encroach_minutes, 120)
                    wocl_rule_triggered = "start_in_wocl"
                    reduction_reason = "start_in_wocl_reduce_up_to_120"
                elif w_start <= de_local.hour <= w_end:
                    reduction_minutes = int(encroach_minutes * 0.5)
                    wocl_rule_triggered = "end_in_wocl"
                    reduction_reason = "end_in_wocl_reduce_50_percent"
                elif (ds_local.hour <= w_start and de_local.hour >= w_end) or encroach_actual > 0:
                    reduction_minutes = int(encroach_minutes * 0.5)
                    wocl_rule_triggered = "span_wocl_or_partial"
                    reduction_reason = "span_or_partial_reduce_50_percent"
                else:
                    reduction_minutes = 0
                    wocl_rule_triggered = "none"
                    reduction_reason = "no_wocl_overlap"

            trace_minutes["wocl_reduction_minutes"] = reduction_minutes
            trace_minutes["wocl_rule_triggered"] = wocl_rule_triggered
            trace_minutes["wocl_reduction_reason"] = reduction_reason

            # compute effective allowed (minutes)
            if isinstance(allowed, int):
                effective_allowed = max(0, allowed - reduction_minutes)
                trace_minutes["effective_allowed_minutes"] = effective_allowed
                trace_minutes["allowed_base_minutes"] = allowed
                trace_minutes["payload_duty_minutes"] = int(payload.duty_minutes or 0)
                trace_minutes["provenance"]["ruleset"] = ruleset_prov

                # convert all minute fields to HH:MM strings for audit output (Option A)
                fdp_trace = {
                    "tables_present": trace_minutes["tables_present"],
                    "tables_count": trace_minutes["tables_count"],
                    "flight_time": minutes_to_hhmm(trace_minutes["flight_minutes"]),
                    "landings": trace_minutes["landings"],
                    "allowed_base": minutes_to_hhmm(trace_minutes["allowed_base_minutes"]),
                    "encroach_actual": minutes_to_hhmm(trace_minutes["encroach_actual_minutes"]),
                    "encroach_used": minutes_to_hhmm(trace_minutes["encroach_minutes"]),
                    "wocl_reduction": minutes_to_hhmm(trace_minutes["wocl_reduction_minutes"]),
                    "effective_allowed": minutes_to_hhmm(trace_minutes["effective_allowed_minutes"]),
                    "payload_duty": minutes_to_hhmm(trace_minutes["payload_duty_minutes"]),
                    "chosen_table_name": trace_minutes.get("chosen_table_name"),
                    "crew_type_selected": trace_minutes.get("crew_type_selected"),
                    "selected_table_is_example": trace_minutes.get("selected_table_is_example"),
                    "allowed_base_from_source": trace_minutes.get("allowed_base_from_source"),
                    "augmented_peer_min_allowed": minutes_to_hhmm(trace_minutes.get("augmented_peer_min_allowed")) if trace_minutes.get("augmented_peer_min_allowed") is not None else None,
                    "applied_conservative_cap_from": trace_minutes.get("applied_conservative_cap_from"),
                    "positioning_applied": trace_minutes.get("positioning_applied"),
                    "positioning_added": minutes_to_hhmm(trace_minutes.get("positioning_added_minutes")),
                    "positioning_counted_as_landing": trace_minutes.get("positioning_counted_as_landing"),
                    "wocl_rule_triggered": trace_minutes.get("wocl_rule_triggered"),
                    "wocl_reduction_reason": trace_minutes.get("wocl_reduction_reason"),
                    "provenance": trace_minutes.get("provenance"),
                    "lookup_provenance": trace_minutes.get("lookup_provenance"),
                    "chosen_provenance": trace_minutes.get("chosen_provenance"),
                }

                # attach inputs as HH:MM to trace for audit completeness
                fdp_trace["input_times"] = {
                    "duty": minutes_to_hhmm(int(payload.duty_minutes or 0)),
                    "flight": minutes_to_hhmm(int(payload.flight_minutes or 0)),
                    "rest_hours": hours_to_hhmm(payload.rest_hours),
                    "acclimatized_tz": tz_hint
                }
                payload.context["_last_fdp_trace"] = fdp_trace

                # check violation (use HH:MM in message)
                if int(payload.duty_minutes or 0) > effective_allowed:
                    msg = f"FDP exceeded: allowed {minutes_to_hhmm(effective_allowed)} (base {minutes_to_hhmm(allowed)}, WOCL reduction {minutes_to_hhmm(reduction_minutes)}), actual {minutes_to_hhmm(int(payload.duty_minutes or 0))}"
                    return Violation(rule_id=rid, message=msg, dgca_reference=dgca_ref)
            else:
                # conversion even if allowed missing
                payload.context["_last_fdp_trace"] = {
                    "flight_time": minutes_to_hhmm(trace_minutes["flight_minutes"]),
                    "landings": trace_minutes["landings"],
                    "payload_duty": minutes_to_hhmm(trace_minutes["payload_duty_minutes"]),
                    "positioning_applied": trace_minutes.get("positioning_applied"),
                    "positioning_added": minutes_to_hhmm(trace_minutes.get("positioning_added_minutes")),
                    "provenance": trace_minutes.get("provenance"),
                    "input_times": {
                        "duty": minutes_to_hhmm(int(payload.duty_minutes or 0)),
                        "flight": minutes_to_hhmm(int(payload.flight_minutes or 0)),
                        "rest_hours": hours_to_hhmm(payload.rest_hours),
                        "acclimatized_tz": tz_hint
                    }
                }
            return None

        # split_duty
        if ltype == "split_duty":
            break_hours = None
            try:
                break_hours = float(ctx.get("split_break_hours")) if ctx.get("split_break_hours") is not None else None
            except Exception:
                break_hours = None
            if break_hours is None:
                return None
            if break_hours < 3:
                ext = 0
            elif 3 <= break_hours <= 10:
                ext = int(0.5 * break_hours * 60)
            else:
                ext = 0
            ctx["_last_split_extension_minutes"] = ext
            # convert to HH:MM for trace consumer
            ctx["_last_split_extension_hhmm"] = minutes_to_hhmm(ext)
            return None

        # cumulative windows
        if ltype == "cumulative_windows":
            table = logic.get("table", []) or []
            hist = payload.history or []
            if not table:
                cl = merged_rules.get("cumulative_limits", {}) or {}
                table = []
                for k, v in cl.items():
                    window_days = None
                    if re.search(r"7", str(k)):
                        window_days = 7
                    elif re.search(r"28", str(k)):
                        window_days = 28
                    elif re.search(r"90", str(k)):
                        window_days = 90
                    elif re.search(r"365", str(k)):
                        window_days = 365
                    if window_days is None:
                        continue
                    mm = None
                    if isinstance(v, dict):
                        if v.get("max_minutes"):
                            mm = int(v.get("max_minutes"))
                        elif v.get("max_flight_time_hours"):
                            mm = int(round(v.get("max_flight_time_hours") * 60))
                        if v.get("max_duty_hours"):
                            table.append({"metric": "duty_minutes", "window_days": window_days, "max_minutes": int(v.get("max_duty_hours") * 60)})
                    if mm:
                        table.append({"metric": "flight_minutes", "window_days": window_days, "max_minutes": mm})

            tz_hint = payload.context.get("acclimatized_tz") if isinstance(payload.context, dict) else None
            anchor_dt = parse_iso(payload.duty_start, tz_hint) or datetime.datetime.utcnow()
            cumulative_trace: Dict[str, Any] = {"per_window": [], "limits_exceeded": False, "provenance": ruleset_prov}
            overall_exceeded = False
            named_totals: Dict[str, int] = {}
            named_remaining: Dict[str, int] = {}
            overall_min_remaining = None

            for entry in table:
                metric = entry.get("metric")
                window_days = int(entry.get("window_days", 0))
                max_minutes = entry.get("max_minutes")
                if not metric or not window_days or max_minutes is None:
                    continue
                cutoff = anchor_dt - datetime.timedelta(days=window_days)
                total = 0
                for h in hist:
                    try:
                        h_date = None
                        if getattr(h, "date", None):
                            h_date = parse_iso(h.date, tz_hint)
                        elif isinstance(h, dict) and h.get("date"):
                            h_date = parse_iso(h.get("date"), tz_hint)
                        if h_date and h_date >= cutoff:
                            if metric == "flight_minutes" and (getattr(h, "flight_minutes", None) or (isinstance(h, dict) and h.get("flight_minutes"))):
                                total += int(getattr(h, "flight_minutes", 0) if getattr(h, "flight_minutes", None) is not None else int(h.get("flight_minutes", 0)))
                            if metric == "duty_minutes" and (getattr(h, "duty_minutes", None) or (isinstance(h, dict) and h.get("duty_minutes"))):
                                total += int(getattr(h, "duty_minutes", 0) if getattr(h, "duty_minutes", None) is not None else int(h.get("duty_minutes", 0)))
                    except Exception:
                        continue
                # include current payload
                if metric == "flight_minutes":
                    total += int(payload.flight_minutes or 0)
                if metric == "duty_minutes":
                    total += int(payload.duty_minutes or 0)

                exceeded = total > int(max_minutes)
                if exceeded:
                    overall_exceeded = True
                remaining = int(max_minutes) - int(total)
                if overall_min_remaining is None or remaining < overall_min_remaining:
                    overall_min_remaining = remaining
                percent_used = int(round((float(total) / float(max_minutes)) * 100)) if max_minutes else 0

                metric_short = metric.replace("_minutes", "") if metric.endswith("_minutes") else metric
                total_key = f"{metric_short}_{window_days}d_total_minutes"
                remaining_key = f"{metric_short}_{window_days}d_remaining_minutes"
                named_totals[total_key] = total
                named_remaining[remaining_key] = remaining

                cumulative_trace["per_window"].append({
                    "metric": metric,
                    "window_days": window_days,
                    "cutoff": cutoff.isoformat(),
                    # minute-level internals (converted later)
                    "total_minutes": total,
                    "max_minutes": int(max_minutes),
                    "remaining_minutes": remaining,
                    "percent_used": percent_used,
                    "exceeded": exceeded
                })

            # attach named totals + remaining
            cumulative_trace.update(named_totals)
            cumulative_trace.update(named_remaining)
            cumulative_trace["limits_exceeded"] = overall_exceeded
            cumulative_trace["overall_min_remaining_minutes"] = int(overall_min_remaining) if overall_min_remaining is not None else None

            # Convert all minute-related fields into HH:MM strings (Option A)
            cum_out = {"per_window": [], "limits_exceeded": cumulative_trace["limits_exceeded"], "provenance": cumulative_trace.get("provenance")}
            for w in cumulative_trace["per_window"]:
                cum_out["per_window"].append({
                    "metric": w["metric"],
                    "window_days": w["window_days"],
                    "cutoff": w["cutoff"],
                    "total": minutes_to_hhmm(w["total_minutes"]),
                    "max": minutes_to_hhmm(w["max_minutes"]),
                    "remaining": minutes_to_hhmm(w["remaining_minutes"]),
                    "percent_used": w["percent_used"],
                    "exceeded": w["exceeded"]
                })

            # add named totals/remaining as HH:MM keys (Option A only - no .raw)
            for k, v in named_totals.items():
                cum_out[k.replace("_minutes", "")] = minutes_to_hhmm(v)
            for k, v in named_remaining.items():
                cum_out[k.replace("_minutes", "")] = minutes_to_hhmm(v)

            cum_out["overall_min_remaining"] = minutes_to_hhmm(cumulative_trace.get("overall_min_remaining_minutes"))

            # attach input times for audit completeness
            cum_out["input_times"] = {
                "duty": minutes_to_hhmm(int(payload.duty_minutes or 0)),
                "flight": minutes_to_hhmm(int(payload.flight_minutes or 0)),
                "rest_hours": hours_to_hhmm(payload.rest_hours),
                "acclimatized_tz": tz_hint
            }

            payload.context["_last_cumulative_trace"] = cum_out

            if overall_exceeded:
                msg = "Cumulative limits exceeded in one or more windows"
                return Violation(rule_id=rid, message=msg, dgca_reference=dgca_ref)
            return None

        # standby
        if ltype == "standby":
            for row in logic.get("table", []) or []:
                cond = row.get("condition", {}) or {}
                ok = True
                loc = ctx.get("standby_location")
                minutes = ctx.get("standby_minutes", 0)
                culminates = ctx.get("standby_culminates", False)
                if "standby_location" in cond:
                    sl = cond["standby_location"]
                    if isinstance(sl, list):
                        ok = ok and (loc in sl)
                    else:
                        ok = ok and (loc == sl)
                if "culminates_into_flight" in cond:
                    ok = ok and (cond["culminates_into_flight"] == culminates)
                minmin = cond.get("standby_minutes_min"); maxmin = cond.get("standby_minutes_max")
                if minmin is not None:
                    ok = ok and (minutes >= minmin)
                if maxmin is not None:
                    ok = ok and (minutes <= maxmin)
                if ok:
                    return None
            return None

        # positioning (legacy logic)
        if ltype == "positioning":
            return None

        # rest
        if ltype == "rest":
            prev_duty_hours = float(ctx.get("previous_duty_hours", 0.0))
            provided_rest_hours = float(payload.rest_hours or 0.0)
            tz_diff = float(ctx.get("time_zone_difference_hours", payload.tz_crossings or 0))
            wocl_minutes = int(ctx.get("wocl_encroachment_minutes") or 0)
            crew_aug = bool(ctx.get("crew_augmented", False))
            pos_precedes = bool(ctx.get("positioning_precedes_fdp", False))
            split_ext = int(ctx.get("split_duty_extension_minutes", 0))
            transport_min = int(ctx.get("transport_time_minutes", 0))
            fdp_extended = float(ctx.get("fdp_was_extended_hours", 0.0))
            res = check_rest_requirements(
                previous_duty_hours=prev_duty_hours,
                provided_rest_hours=provided_rest_hours,
                reference_dt_iso=payload.duty_start or "",
                time_zone_difference_hours=tz_diff,
                wocl_encroachment_minutes=wocl_minutes,
                crew_augmented=crew_aug,
                positioning_precedes_fdp=pos_precedes,
                split_duty_extension_minutes=split_ext,
                transport_time_minutes=transport_min,
                fdp_was_extended_hours=fdp_extended,
                ruleset=RULES
            )
            if res.get("violations"):
                req_h = res['violations']['insufficient_rest']['required_hours']
                prov_h = res['violations']['insufficient_rest']['provided_hours']
                vmsg = f"Insufficient rest: required {hours_to_hhmm(req_h)} (hh:mm), provided {hours_to_hhmm(prov_h)} (hh:mm)"
                return Violation(rule_id=rid, message=vmsg, dgca_reference=dgca_ref)
            return None

        return None

    except Exception as e:
        log.error("Exception while evaluating rule %s: %s\n%s", rule.get("id", "<no-id>"), e, traceback.format_exc())
        return None


# ---------- Normalize payload helper ----------
def normalize_payload_times(payload: CheckRequest) -> None:
    """
    Normalizes numeric fields and applies positioning adjustments.
    Writes metadata into payload.context (normalized_* and positioning flags) and
    records canonical input HH:MM representations (Option A).
    """
    try:
        if payload.context is None:
            payload.context = {}

        # Normalize duty / flight / rest to numeric minutes/hours
        dm = getattr(payload, "duty_minutes", None)
        payload.duty_minutes = int(parse_time_to_minutes(dm)) if dm is not None else 0
        fm = getattr(payload, "flight_minutes", None)
        payload.flight_minutes = int(parse_time_to_minutes(fm)) if fm is not None else 0
        rh = getattr(payload, "rest_hours", None)
        payload.rest_hours = float(parse_hours(rh)) if rh is not None else 0.0

        # store normalized numeric values (internal) and canonical HH:MM input representation (Option A)
        payload.context["normalized_duty_minutes"] = payload.duty_minutes
        payload.context["normalized_flight_minutes"] = payload.flight_minutes
        payload.context["normalized_rest_hours"] = payload.rest_hours

        # canonical HH:MM for inputs (audit-friendly)
        payload.context["input_times"] = {
            "duty": minutes_to_hhmm(int(payload.duty_minutes or 0)),
            "flight": minutes_to_hhmm(int(payload.flight_minutes or 0)),
            "rest_hours": hours_to_hhmm(payload.rest_hours)
        }

        # Positioning handling
        pos_minutes = None
        try:
            pos_raw = payload.context.get("positioning_minutes") if payload.context is not None else None
            if pos_raw is None:
                pos_raw = getattr(payload, "positioning_minutes", None)
            if pos_raw is not None:
                pos_minutes = int(parse_time_to_minutes(pos_raw))
        except Exception:
            pos_minutes = None

        pos_precedes = bool(payload.context.get("positioning_precedes_fdp", False))

        positioning_rules = RULES.get("positioning_rules", {}) if isinstance(RULES, dict) else {}
        count_as_duty_default = positioning_rules.get("count_as_duty_default", False)
        count_positioning_as_landing_if_minutes_over = positioning_rules.get("count_positioning_as_landing_if_minutes_over", None)

        payload_count_as_duty = payload.context.get("count_positioning_as_duty") if "count_positioning_as_duty" in payload.context else None
        payload_count_as_landing_flag = payload.context.get("positioning_counts_as_landing") if "positioning_counts_as_landing" in payload.context else None

        positioning_applied = False
        positioning_counted_as_landing = False
        positioning_added_minutes = 0

        if pos_minutes and pos_precedes:
            payload.duty_minutes = int(payload.duty_minutes or 0) + int(pos_minutes)
            payload.context["normalized_duty_minutes"] = payload.duty_minutes
            positioning_applied = True
            positioning_added_minutes = int(pos_minutes)

            count_landing = False
            if payload_count_as_landing_flag is not None:
                count_landing = bool(payload_count_as_landing_flag)
            else:
                try:
                    if count_positioning_as_landing_if_minutes_over is not None:
                        thr = int(count_positioning_as_landing_if_minutes_over)
                        if pos_minutes >= thr:
                            count_landing = True
                except Exception:
                    count_landing = False

            count_as_duty_flag = False
            if payload_count_as_duty is not None:
                count_as_duty_flag = bool(payload_count_as_duty)
            else:
                count_as_duty_flag = bool(count_as_duty_default)

            if count_as_duty_flag and count_landing:
                try:
                    payload.landings = int(getattr(payload, "landings", 0)) + 1
                    positioning_counted_as_landing = True
                except Exception:
                    positioning_counted_as_landing = False

        payload.context["_positioning_applied"] = positioning_applied
        payload.context["_positioning_added_minutes"] = positioning_added_minutes
        payload.context["_positioning_counted_as_landing"] = positioning_counted_as_landing

    except Exception:
        log.exception("Failed to normalise payload times")


# ---------- /check endpoint ----------
@router.post("/check", response_model=CheckResult)
def check_legality(payload: CheckRequest, request: Request):
    normalize_payload_times(payload)

    try:
        app_obj = request.app
        rules_map = getattr(app_obj.state, "rules", None)
        if not rules_map:
            rules_map = {}
            # Build compact rules_map for known types using merged RULES
            if isinstance(RULES, dict) and ("two_pilot" in RULES or "single_pilot" in RULES):
                rules_map["fdp_tables"] = {"id": "fdp_tables", "title": "fdp_tables", "dgca_reference": RULES.get("meta", {}).get("source_files", []), "logic": {"type": "fdp_tables", "tables": []}}
                tables = []
                if "two_pilot" in RULES:
                    tp = RULES.get("two_pilot", {})
                    fdptable = tp.get("fdp_table", [])
                    bands = []
                    for r in fdptable:
                        bh = r.get("max_flight_time_hours")
                        mm = None
                        if r.get("max_fdp_minutes") is not None:
                            try:
                                mm = int(r.get("max_fdp_minutes"))
                            except Exception:
                                mm = None
                        elif r.get("max_fdp_hours") is not None:
                            try:
                                mm = int(round(float(r.get("max_fdp_hours")) * 60))
                            except Exception:
                                mm = None
                        ml = r.get("max_landings")
                        bands.append({"band_hours": bh, "max_fdp_minutes": mm, "max_landings": ml})
                    tables.append({"source": "two_pilot", "source_page": None, "bands": bands})
                if "single_pilot" in RULES:
                    sp = RULES.get("single_pilot", {})
                    fdptable = sp.get("fdp_table", [])
                    bands = []
                    for r in fdptable:
                        bh = r.get("max_flight_time_hours")
                        mm = None
                        if r.get("max_fdp_minutes") is not None:
                            try:
                                mm = int(r.get("max_fdp_minutes"))
                            except Exception:
                                mm = None
                        elif r.get("max_fdp_hours") is not None:
                            try:
                                mm = int(round(float(r.get("max_fdp_hours")) * 60))
                            except Exception:
                                mm = None
                        ml = r.get("max_landings")
                        bands.append({"band_hours": bh, "max_fdp_minutes": mm, "max_landings": ml})
                    tables.append({"source": "single_pilot", "source_page": None, "bands": bands})
                rules_map["fdp_tables"]["logic"]["tables"] = tables

            if "rest_rules" in RULES:
                rules_map["rest"] = {"id": "rest", "title": "rest", "dgca_reference": RULES.get("meta", {}).get("source_files", []), "logic": {"type": "rest"}}

            if "positioning_rules" in RULES:
                rules_map["positioning"] = {"id": "positioning", "title": "positioning", "dgca_reference": RULES.get("meta", {}).get("source_files", []), "logic": {"type": "positioning"}}

            if "standby_rules" in RULES:
                rules_map["standby"] = {"id": "standby", "title": "standby", "dgca_reference": RULES.get("meta", {}).get("source_files", []), "logic": {"type": "standby", "table": RULES.get("standby_rules", {}).get("table", [])}}

            if "cumulative_limits" in RULES:
                rules_map["cumulative_windows"] = {"id": "cumulative_windows", "title": "cumulative_windows", "dgca_reference": RULES.get("meta", {}).get("source_files", []), "logic": {"type": "cumulative_windows", "table": []}}

    except Exception as e:
        log.exception("Failed to access app state in check_legality: %s", e)
        raise HTTPException(status_code=500, detail="Server internal error while loading rules")

    violations: List[Violation] = []
    details: Dict[str, Any] = {"applied_rules": [], "errors": [], "trace_by_rule": {}, "meta": compute_ruleset_provenance(RULES)}

    # iterate through rules deterministically sorted by rule id to ensure reproducible audit outputs
    for rid in sorted((rules_map or {}).keys()):
        try:
            rule_obj = (rules_map or {}).get(rid)
            raw = rule_obj.dict() if hasattr(rule_obj, "dict") else dict(rule_obj)
            # Evaluate rule
            v = evaluate_rule_for_payload(raw, payload, rules_map)
            # record applied rule metadata (provenance deterministic)
            details["applied_rules"].append({
                "rule_id": raw.get("id", rid),
                "title": raw.get("title"),
                "dgca_reference": raw.get("dgca_reference"),
                "applied_at": datetime.datetime.utcnow().isoformat() + "Z",
                "selection_context": {
                    "crew_id": payload.crew_id,
                    "duty_start": payload.duty_start,
                    "duty_end": payload.duty_end,
                    # canonical input times HH:MM (Option A)
                    "input_times": payload.context.get("input_times")
                }
            })

            # Extract traces prioritizing cumulative -> fdp -> split
            try:
                if isinstance(payload.context, dict):
                    cum_trace = payload.context.pop("_last_cumulative_trace", None)
                    fdp_trace = payload.context.pop("_last_fdp_trace", None)
                    split_trace = payload.context.pop("_last_split_extension_hhmm", None)
                else:
                    cum_trace = None
                    fdp_trace = None
                    split_trace = None

                chosen_trace = None
                if cum_trace is not None:
                    chosen_trace = cum_trace
                elif fdp_trace is not None:
                    chosen_trace = fdp_trace
                elif split_trace is not None:
                    chosen_trace = {"split_extension": split_trace}

                if chosen_trace:
                    # ensure positioning metadata present (HH:MM)
                    chosen_trace["positioning_applied"] = bool(payload.context.get("_positioning_applied", False))
                    chosen_trace["positioning_added"] = minutes_to_hhmm(int(payload.context.get("_positioning_added_minutes", 0)))
                    chosen_trace["positioning_counted_as_landing"] = bool(payload.context.get("_positioning_counted_as_landing", False))
                    # add deterministic provenance snapshot
                    chosen_trace["result_provenance_snapshot"] = compute_ruleset_provenance(RULES)
                    details["trace_by_rule"][raw.get("id", rid)] = chosen_trace
            except Exception:
                log.debug("Failed to extract trace for rule %s", rid)

            if v:
                violations.append(v)
        except Exception as e:
            tb = traceback.format_exc()
            log.error("Unhandled error applying rule %s: %s\n%s", rid, e, tb)
            details["errors"].append({"rule_id": rid, "error": str(e)})

    legal = len(violations) == 0
    return CheckResult(legal=legal, violations=violations, details=details)
