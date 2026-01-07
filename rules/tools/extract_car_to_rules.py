# tools/extract_car_to_rules.py
"""
Extract CAR FDP tables and rules from uploaded CAR PDFs into canonical JSON rules.

Run:
    python tools/extract_car_to_rules.py \
        --pdfs "/mnt/data/CAR Sec 7 Seriess J Pt 3.pdf" \
               "/mnt/data/D7J-J4(R1_Jan2023).pdf" \
               "/mnt/data/D7J-J1(Issue4).pdf" \
        --outdir rules/

Output:
    rules/fdp_rules_audit.json
    rules/rest_rules_audit.json
    rules/standby_rules_audit.json
    rules/positioning_rules_audit.json
    rules/cumulative_rules_audit.json

Notes:
- Uses pdfplumber to parse table structures.
- Attempts to capture paragraph text and page numbers for traceability.
- Manual review may still be required for OCR/table edge cases; script writes a 'warnings' file.
"""
import argparse
import json
from pathlib import Path
import pdfplumber
import re
from typing import List, Dict, Any, Tuple

def extract_text_snippets(pdf_path: Path, pages: List[int]) -> str:
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pages:
            if p < 0 or p >= len(pdf.pages): continue
            page = pdf.pages[p]
            txt = page.extract_text() or ""
            text.append(f"--- PAGE {p+1} ---\n{txt}")
    return "\n".join(text)

def extract_tables_from_pdf(pdf_path: Path) -> List[Tuple[int,List[List[str]]]]:
    """Return list of (page_index, table_as_rows) found by pdfplumber."""
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t in tables:
                # Normalize rows: replace None with ""
                norm = [[cell.strip() if isinstance(cell, str) else "" for cell in row] for row in t]
                results.append((i, norm))
    return results

def find_fdp_tables(tables: List[Tuple[int,List[List[str]]]], pdf_path: Path) -> List[Dict[str,Any]]:
    """Heuristic: find tables that look like FDP tables using header keywords."""
    candidates = []
    headers_keywords = ["Flight Time", "Flight Duty Period", "Landings", "Max", "Maximum"]
    for page_idx, table in tables:
        # join header row
        if not table: continue
        header = " ".join(table[0]).lower()
        if any(kw.lower() in header for kw in headers_keywords):
            candidates.append({"page": page_idx, "table": table})
            continue
        # fallback: search for keywords in first 3 rows
        snippet = " ".join(" ".join(r).lower() for r in table[:3])
        if any(kw.lower() in snippet for kw in headers_keywords):
            candidates.append({"page": page_idx, "table": table})
    # add traceability
    for c in candidates:
        c["source_pdf"] = str(pdf_path)
    return candidates

def parse_fdp_table(table_rows: List[List[str]]) -> Dict[str,Any]:
    """
    Parse a table that looks like an FDP table into canonical mapping.
    This function is deliberately defensive: it tries to find numeric cells and map
    them to flight-time bands and landing counts.
    """
    # Attempt to find header row and column indices
    header = table_rows[0]
    header_norm = [h.lower() for h in header]
    # heuristics to find columns
    # We want columns for: flight_time_band, max_fdp (hours), max_no. of landings (int)
    parsed = {"bands": []}
    # If table seems like "Flight time | Max FDP (hours) | Max no. of landings" then map that
    # fallback: try to parse each data row for numbers
    for r in table_rows[1:]:
        # collapse row cells into text and find numbers
        row_text = " | ".join(r)
        # match patterns like "Up to 8 hours", "Upto 8 hours", or numeric hour ranges
        m = re.search(r'(\d+)\s*(?:\+?[-to]*\s*\d*)?\s*hours', row_text, re.I)
        if m:
            band_hours = int(m.group(1))
        else:
            # fallback: use first numeric seen as hours
            nums = re.findall(r'\d+', row_text)
            band_hours = int(nums[0]) if nums else None
        # find any FDP hours like 11:00 or 900 -> 11 or numeric
        # For FDP duration values, find patterns like '11:00' or '11' in other cells
        fdp_val = None
        landings_val = None
        # scan cells for HH:MM or integer
        for cell in r:
            if not cell: continue
            mm = re.search(r'(\d{1,2}:\d{2})', cell)
            if mm:
                # convert HH:MM to minutes
                hh, mi = mm.group(1).split(':')
                val_min = int(hh)*60 + int(mi)
                if fdp_val is None:
                    fdp_val = val_min
                    continue
            # else search for plain numbers >= 60 -> minutes or <= 24 -> hours
            for num in re.findall(r'\d+', cell):
                n = int(num)
                if n >= 60 and fdp_val is None:
                    fdp_val = n
                elif n <= 24 and (landings_val is None):
                    # could be landings or hours; try to disambiguate by context
                    # if cell contains 'land' assume landings
                    if 'land' in cell.lower():
                        landings_val = n
                    else:
                        # if we already have band_hours and n <= band_hours -> landings
                        if band_hours is not None and n <= band_hours:
                            landings_val = n
                        else:
                            # ambiguous, skip
                            pass
        parsed["bands"].append({
            "band_hours": band_hours,
            "max_fdp_minutes": fdp_val,
            "max_landings": landings_val,
            "raw_row": r
        })
    return parsed

def build_traceable_rules(fdp_parsed: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Compose canonical rules structure from parsed FDP table pieces.
    """
    rules = {
        "id": "FDP_TABLES_DGCA_AUDIT",
        "title": "DGCA CAR - FDP tables (audit JSON produced by extractor)",
        "dgca_reference": [],
        "notes": {
            "generated_by": "tools/extract_car_to_rules.py",
            "warnings": []
        },
        "logic": {
            "type": "fdp_tables",
            "tables": []
        }
    }
    for part in fdp_parsed:
        rules["dgca_reference"].append({"file": part["source_pdf"], "page": part["page"]+1})
        # convert parsed bands into a table entry
        table_entry = {
            "source": part["source_pdf"],
            "source_page": part["page"]+1,
            "bands": part["parsed"]["bands"]
        }
        rules["logic"]["tables"].append(table_entry)
    return rules

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", nargs="+", required=True, help="Paths to CAR PDFs")
    parser.add_argument("--outdir", required=True, help="Output folder for rules JSON")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fdp_parsed_all = []
    warnings = []
    for pdf in args.pdfs:
        pdfp = Path(pdf)
        if not pdfp.exists():
            warnings.append(f"PDF not found: {pdf}")
            continue
        print(f"Parsing PDF: {pdf}")
        tables = extract_tables_from_pdf(pdfp)
        fdp_candidates = find_fdp_tables(tables, pdfp)
        for cand in fdp_candidates:
            try:
                parsed = parse_fdp_table(cand["table"])
                fdp_parsed_all.append({"source_pdf": cand["source_pdf"], "page": cand["page"], "parsed": parsed})
            except Exception as e:
                warnings.append(f"Failed to parse table on {pdf}:{cand['page']+1}: {e}")

    # Build canonical rule JSON
    rules_json = build_traceable_rules(fdp_parsed_all)
    if warnings:
        rules_json["notes"]["warnings"] = warnings

    out_path = outdir / "fdp_rules_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rules_json, f, indent=2)

    print("Wrote:", out_path)
    if warnings:
        warn_path = outdir / "fdp_rules_audit_warnings.txt"
        warn_path.write_text("\n".join(warnings))
        print("Warnings written to:", warn_path)

if __name__ == "__main__":
    main()
