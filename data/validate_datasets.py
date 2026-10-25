"""Integrity checks for the three spare-parts datasets. Stdlib only."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def load(name: str) -> list[dict]:
    with (D / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    if x is None or x == "":
        return None
    return float(x)


def chk(rows_out, name, ok, info=""):
    rows_out.append((name, "PASS" if ok else "FAIL", str(info)))


def unique(rows, keys):
    seen = set()
    for r in rows:
        k = tuple(r[c] for c in keys)
        if k in seen:
            return False
        seen.add(k)
    return True


def main():
    d1 = load("01_demand_forecasting.csv")
    d2 = load("02_scheduling_inputs.csv")
    d3 = load("03_spare_parts_catalogue.csv")
    R = []
    d2_fields = list(csv.DictReader(open(D / "02_scheduling_inputs.csv", encoding="utf-8")).fieldnames)
    chk(R, "D2 no duplicate supply_mode column", d2_fields.count("supply_mode") == 1, d2_fields.count("supply_mode"))

    chk(R, "D1 key unique", unique(d1, ["part_id", "region", "week_start_date"]))
    chk(R, "D2 key unique", unique(d2, ["part_id"]))
    chk(R, "D3 key unique", unique(d3, ["part_id"]))
    chk(R, "D3 part_number unique", unique(d3, ["part_number"]))
    chk(R, "D3 part_name unique", unique(d3, ["part_name"]))
    chk(R, "D3 component-model-variant unique", unique(d3, ["component", "vehicle_model", "variant"]))

    p3 = {r["part_id"] for r in d3}
    p1 = {r["part_id"] for r in d1}
    p2 = {r["part_id"] for r in d2}
    chk(R, "D1 parts subset of D3", p1 <= p3)
    chk(R, "D2 parts subset of D3", p2 <= p3)
    miss = p3 - p2
    miss_ok = all(next(r["lifecycle_status"] for r in d3 if r["part_id"] == pid) == "Obsolete" for pid in miss)
    chk(R, "D3 missing from D2 only Obsolete", miss_ok, sorted(miss))

    d3m = {r["part_id"]: r for r in d3}
    d1_first = {}
    for r in d1:
        d1_first.setdefault(r["part_id"], r)
    same = True
    for pid, r in d1_first.items():
        c = d3m[pid]
        for col in ["part_name", "component", "vehicle_system", "part_category", "vehicle_model", "vehicle_type",
                    "criticality", "abc_class", "part_type", "lifecycle_status", "make_or_buy", "is_imported", "unit_cost"]:
            if str(r[col]) != str(c[col]):
                same = False
                break
    chk(R, "D1 vs D3 shared attributes identical", same)

    sl_ok = all(abs(float(r["service_level_target"]) - {"High": 0.99, "Medium": 0.95, "Low": 0.90}[r["criticality"]]) < 1e-9 for r in d3)
    chk(R, "D3 service level follows criticality", sl_ok)

    cat = defaultdict(set)
    mat = defaultdict(set)
    for r in d3:
        cat[r["component"]].add(r["part_category"])
        mat[r["component"]].add(r["material"])
    chk(R, "D3 one category/material per component", all(len(v) == 1 for v in cat.values()) and all(len(v) == 1 for v in mat.values()))

    chk(R, "D3 imported iff country not India", all((r["is_imported"] == "1") == (r["country_of_origin"] != "India") for r in d3))
    chk(R, "D3 new parts have launch_week", all((r["launch_week"] != "") == (r["lifecycle_status"] == "New") for r in d3))
    sets = {r["component"] for r in d3 if r["unit_of_measure"] == "SET"}
    chk(R, "D3 SET only for kits", sets <= {"Rear brake pad set", "Piston ring set"}, sets)
    perish = {r["part_category"] for r in d3 if r["shelf_life_days"]}
    chk(R, "D3 shelf life only sensible categories", perish <= {"Rubber", "Polymer", "Electrical", "Mechanical"}, perish)

    chk(R, "D2 available = on_hand - reserved - blocked",
        all(int(r["available_qty"]) == int(r["on_hand_qty"]) - int(r["reserved_qty"]) - int(r["blocked_qty"]) for r in d2))
    chk(R, "D2 open_po = due_this + due_later",
        all(int(r["open_po_qty"]) == int(r["open_po_due_this_month_qty"]) + int(r["open_po_due_later_qty"]) for r in d2))
    qty_cols = [c for c in d2[0] if c.endswith("_qty")]
    chk(R, "D2 no negative qty", all(all(int(r[c]) >= 0 for c in qty_cols if r[c] != "") for r in d2))
    sob_ok = True
    for r in d2:
        s = sum(float(r[f"sup{k}_sob_pct"]) for k in (1, 2, 3) if r[f"sup{k}_sob_pct"] != "")
        if abs(s - 100) > 0.05:
            sob_ok = False
    chk(R, "D2 SOB sums to 100", sob_ok)
    chk(R, "D2 primary supplier present, slots ordered",
        all(r["sup1_id"] and not (r["sup3_id"] and not r["sup2_id"]) for r in d2))
    chk(R, "D2 Make parts have no import supplier",
        not any(r["supply_mode"] == "Production" and any(r[f"sup{k}_type"] == "Import" for k in (1, 2, 3)) for r in d2))

    # D1 target
    neg = 0
    miss_match = True
    cons_ok = True
    for r in d1:
        if r["split_role"] == "future":
            if r["demand_qty"] != "":
                miss_match = False
            continue
        if r["is_missing_week"] == "1":
            if r["demand_qty"] != "":
                miss_match = False
        else:
            q = num(r["demand_qty"])
            if q is None or q < 0 or abs(q - round(q)) > 1e-9:
                neg += 1
            if r["out_consumption_qty"] != "" and float(r["out_consumption_qty"]) > q:
                cons_ok = False
    chk(R, "D1 target non-negative integers", neg == 0, neg)
    chk(R, "D1 target blank only missing/future", miss_match)
    chk(R, "D1 consumption <= demand", cons_ok)

    # contiguous weekly
    series = defaultdict(list)
    for r in d1:
        series[(r["part_id"], r["region"])].append(datetime.fromisoformat(r["week_start_date"]).date())
    contig = True
    for k, ds in series.items():
        ds = sorted(ds)
        for a, b in zip(ds, ds[1:]):
            if (b - a).days != 7:
                contig = False
                break
    chk(R, "D1 series contiguous weekly", contig)

    fut = [r for r in d1 if r["split_role"] == "future"]
    fut_weeks = {r["week_start_date"] for r in fut}
    chk(R, "D1 future is 14 weeks", len(fut_weeks) == 14, len(fut_weeks))
    obs = {r["part_id"] for r in d3 if r["lifecycle_status"] == "Obsolete"}
    chk(R, "D1 no obsolete in future", not any(r["part_id"] in obs for r in fut))

    mix = defaultdict(int)
    seen = set()
    for r in d1:
        k = (r["part_id"], r["region"])
        if k in seen:
            continue
        seen.add(k)
        mix[r["demand_class_trainval"] or "short"] += 1
    chk(R, "D1 has lumpy series", mix.get("Lumpy", 0) >= 40, dict(mix))
    chk(R, "D1 has all four SBC classes", all(mix.get(c, 0) >= 20 for c in ("Smooth", "Erratic", "Intermittent", "Lumpy")), dict(mix))
    chk(R, "D1 festival_name not literal None", all(r["festival_name"] != "None" for r in d1))

    print("\n".join(f"{s:6}  {n}  {i}" for n, s, i in R))
    fails = sum(1 for _, s, _ in R if s == "FAIL")
    print(f"\nFAILS: {fails} of {len(R)}")
    print("class mix (series):", dict(mix))
    print("rows:", {"D1": len(d1), "D2": len(d2), "D3": len(d3)})
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
