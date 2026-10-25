"""
Generate the three spare-parts datasets from the problem statement.

Uses the Python standard library only (no pandas/numpy). Seed 42.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

SEED = 42
OUT = Path(__file__).resolve().parent
HIST_START = date(2023, 1, 2)  # Monday — three full years of history
HIST_END = date(2026, 9, 21)
FUTURE_END = date(2026, 12, 28)
AS_OF = date(2026, 9, 26)
WARMUP_END = date(2024, 1, 1)
TRAINVAL_END = date(2026, 3, 23)
REGIONS = ["North", "West", "South", "East"]
REGION_SCALE = {"North": 1.18, "West": 1.08, "South": 1.00, "East": 0.82}

VEHICLES = [
    ("Model_A", "4W SUV"),
    ("Model_B", "4W Sedan"),
    ("Model_C", "4W Hatch"),
    ("Model_D", "4W SUV"),
    ("Model_E", "CV"),
    ("Model_F", "2W"),
    ("Model_G", "4W Sedan"),
    ("Model_H", "4W Hatch"),
]

# One category and one material per component.
# Tuple: name, system, category, material, uom, pack, shelf_days, (cost_lo, cost_hi), crit_bias, (wk_qty_lo, wk_qty_hi), default_sbc
COMPONENTS = [
    ("Oil filter cartridge", "Engine", "Mechanical", "Mild Steel", "EA", 20, 1095, (90, 280), "Medium", (45, 140), "Smooth"),
    ("Air filter element", "Engine", "Polymer", "Polypropylene", "EA", 10, 730, (120, 450), "Medium", (35, 110), "Smooth"),
    ("Fuel injector assy", "Engine", "Mechanical", "Brass", "EA", 4, None, (2800, 8500), "High", (2, 12), "Intermittent"),
    ("Piston ring set", "Engine", "Ferrous", "Cast Iron", "SET", 4, None, (900, 2800), "High", (3, 14), "Intermittent"),
    ("Cylinder head gasket", "Engine", "Ferrous", "Mild Steel", "EA", 10, None, (250, 900), "High", (4, 18), "Intermittent"),
    ("Turbocharger unit", "Engine", "Mechanical", "Cast Iron", "EA", 1, None, (18000, 42000), "High", (1, 6), "Lumpy"),
    ("Front brake disc RH", "Braking", "Ferrous", "Cast Iron", "EA", 2, None, (1200, 3800), "High", (8, 28), "Erratic"),
    ("Rear brake pad set", "Braking", "Mechanical", "Ceramic", "SET", 4, 1095, (800, 2600), "High", (20, 70), "Smooth"),
    ("Brake caliper LH", "Braking", "Ferrous", "Cast Iron", "EA", 1, None, (3500, 9800), "High", (2, 10), "Lumpy"),
    ("ABS wheel speed sensor rear", "Braking", "Electronics", "FR4", "EA", 10, None, (450, 1600), "High", (6, 22), "Erratic"),
    ("Brake master cylinder", "Braking", "Mechanical", "Aluminium", "EA", 1, None, (1800, 5200), "High", (2, 9), "Lumpy"),
    ("Front shock absorber", "Suspension", "Mechanical", "Mild Steel", "EA", 2, None, (1400, 4200), "High", (6, 24), "Erratic"),
    ("Suspension coil spring", "Suspension", "Ferrous", "Spring Steel", "EA", 2, None, (700, 2200), "Medium", (5, 20), "Intermittent"),
    ("Control arm LH", "Suspension", "Ferrous", "Mild Steel", "EA", 2, None, (1100, 3400), "High", (4, 16), "Intermittent"),
    ("Stabilizer link", "Suspension", "Ferrous", "Mild Steel", "EA", 10, None, (180, 650), "Medium", (12, 40), "Smooth"),
    ("Tie rod end LH", "Steering", "Ferrous", "Mild Steel", "EA", 10, None, (220, 780), "High", (10, 36), "Smooth"),
    ("Steering rack assy", "Steering", "Mechanical", "Mild Steel", "EA", 1, None, (6500, 18500), "High", (1, 7), "Lumpy"),
    ("Transmission mount", "Transmission", "Rubber", "NBR", "EA", 5, 1095, (250, 900), "Medium", (8, 28), "Erratic"),
    ("CV joint boot", "Transmission", "Rubber", "EPDM", "EA", 10, 1095, (90, 380), "Low", (15, 50), "Smooth"),
    ("Clutch plate assy", "Transmission", "Mechanical", "Mild Steel", "EA", 2, None, (1500, 4800), "Medium", (4, 16), "Intermittent"),
    ("Gearbox synchro ring", "Transmission", "Non-ferrous", "Brass", "EA", 10, None, (350, 1200), "Medium", (5, 18), "Intermittent"),
    ("Radiator assy", "Cooling", "Non-ferrous", "Aluminium", "EA", 1, None, (3200, 9800), "High", (2, 10), "Lumpy"),
    ("Coolant hose upper", "Cooling", "Rubber", "EPDM", "EA", 10, 1095, (80, 320), "Medium", (18, 55), "Smooth"),
    ("Thermostat housing", "Cooling", "Non-ferrous", "Aluminium", "EA", 5, None, (280, 950), "Medium", (6, 22), "Erratic"),
    ("Water pump", "Cooling", "Mechanical", "Cast Iron", "EA", 2, None, (1100, 3400), "High", (4, 16), "Intermittent"),
    ("Cabin air filter", "HVAC", "Polymer", "Polypropylene", "EA", 20, 730, (150, 480), "Low", (30, 90), "Smooth"),
    ("Blower motor assy", "HVAC", "Electrical", "Copper", "EA", 1, None, (1800, 5200), "Medium", (4, 14), "Intermittent"),
    ("AC compressor", "HVAC", "Mechanical", "Aluminium", "EA", 1, None, (8500, 22000), "High", (1, 6), "Lumpy"),
    ("Condenser core", "HVAC", "Non-ferrous", "Aluminium", "EA", 1, None, (2800, 7600), "Medium", (2, 9), "Lumpy"),
    ("Headlamp assy LH", "Lighting", "Electrical", "Polycarbonate", "EA", 1, None, (4500, 14000), "Medium", (2, 10), "Lumpy"),
    ("Tail lamp assy RH", "Lighting", "Electrical", "Polycarbonate", "EA", 1, None, (1800, 6200), "Low", (3, 12), "Intermittent"),
    ("Fog lamp bulb", "Lighting", "Electrical", "Glass", "EA", 20, None, (80, 280), "Low", (40, 120), "Smooth"),
    ("LED indicator module", "Lighting", "Electronics", "FR4", "EA", 10, None, (220, 780), "Low", (12, 40), "Erratic"),
    ("Front bumper assy", "Body", "Body", "Polypropylene", "EA", 1, None, (4500, 16000), "Medium", (2, 9), "Lumpy"),
    ("Bonnet/hood panel", "Body", "Body", "Mild Steel", "EA", 1, None, (5200, 15000), "Medium", (1, 6), "Lumpy"),
    ("Fender liner RH", "Body", "Polymer", "Polypropylene", "EA", 5, None, (350, 1100), "Low", (8, 28), "Erratic"),
    ("Door handle outer LH", "Body", "Body", "Polypropylene", "EA", 10, None, (180, 650), "Low", (10, 32), "Smooth"),
    ("Battery 12V", "Electrical", "Electrical", "Polypropylene", "EA", 1, 540, (3800, 9200), "High", (6, 22), "Erratic"),
    ("Starter motor", "Electrical", "Electrical", "Copper", "EA", 1, None, (3200, 8900), "High", (2, 10), "Lumpy"),
    ("Alternator assy", "Electrical", "Electrical", "Copper", "EA", 1, None, (4500, 12500), "High", (2, 9), "Lumpy"),
    ("Wiring harness front", "Electrical", "Electrical", "Copper", "EA", 2, None, (1600, 5400), "High", (3, 12), "Intermittent"),
    ("Voltage regulator", "Electrical", "Electronics", "FR4", "EA", 10, None, (280, 980), "Medium", (8, 26), "Erratic"),
]

SYS_CODE = {
    "Engine": "ENG", "Braking": "BRK", "Suspension": "SUS", "Steering": "STR",
    "Transmission": "TRN", "Cooling": "CLG", "HVAC": "HVC", "Lighting": "LGT",
    "Body": "BDY", "Electrical": "ELE",
}

CRIT_SL = {"High": 0.99, "Medium": 0.95, "Low": 0.90}


def mondays(start: date, end: date) -> list[date]:
    d = start
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def iso_week(d: date) -> int:
    return d.isocalendar()[1]


def fiscal(d: date) -> tuple[str, str]:
    if d.month >= 4:
        fy = f"FY{d.year}-{str(d.year + 1)[2:]}"
        q = {4: "Q1", 5: "Q1", 6: "Q1", 7: "Q2", 8: "Q2", 9: "Q2", 10: "Q3", 11: "Q3", 12: "Q3"}[d.month]
    else:
        fy = f"FY{d.year - 1}-{str(d.year)[2:]}"
        q = "Q4"
    return q, fy


def split_role(d: date) -> str:
    if d > HIST_END:
        return "future"
    if d <= WARMUP_END:
        return "warmup"
    if d <= TRAINVAL_END:
        return "train_val"
    return "final_test"


FESTIVAL_SCOPE = {
    "Durga Puja": {"East"},
    "Onam": {"South"},
    "Pongal": {"South"},
    "Ganesh Chaturthi": {"West"},
    "Holi": {"North", "West", "East"},
}


def build_festival_map() -> dict[date, tuple[str, str]]:
    raw = [
        (date(2023, 1, 15), "Pongal", "Medium"),
        (date(2023, 3, 8), "Holi", "Medium"),
        (date(2023, 4, 22), "Eid", "Medium"),
        (date(2023, 8, 15), "Independence Day", "Low"),
        (date(2023, 8, 31), "Onam", "Medium"),
        (date(2023, 9, 19), "Ganesh Chaturthi", "Medium"),
        (date(2023, 10, 21), "Durga Puja", "High"),
        (date(2023, 10, 24), "Dussehra", "High"),
        (date(2023, 11, 12), "Diwali", "High"),
        (date(2023, 12, 25), "Christmas", "Low"),
        (date(2024, 1, 15), "Pongal", "Medium"),
        (date(2024, 3, 25), "Holi", "Medium"),
        (date(2024, 4, 11), "Eid", "Medium"),
        (date(2024, 8, 15), "Independence Day", "Low"),
        (date(2024, 9, 7), "Ganesh Chaturthi", "Medium"),
        (date(2024, 9, 15), "Onam", "Medium"),
        (date(2024, 10, 12), "Dussehra", "High"),
        (date(2024, 10, 10), "Durga Puja", "High"),
        (date(2024, 10, 31), "Diwali", "High"),
        (date(2024, 12, 25), "Christmas", "Low"),
        (date(2025, 1, 14), "Pongal", "Medium"),
        (date(2025, 3, 14), "Holi", "Medium"),
        (date(2025, 3, 31), "Eid", "Medium"),
        (date(2025, 8, 15), "Independence Day", "Low"),
        (date(2025, 8, 27), "Ganesh Chaturthi", "Medium"),
        (date(2025, 9, 5), "Onam", "Medium"),
        (date(2025, 9, 29), "Durga Puja", "High"),
        (date(2025, 10, 2), "Dussehra", "High"),
        (date(2025, 10, 20), "Diwali", "High"),
        (date(2025, 12, 25), "Christmas", "Low"),
        (date(2026, 1, 15), "Pongal", "Medium"),
        (date(2026, 3, 3), "Holi", "Medium"),
        (date(2026, 3, 21), "Eid", "Medium"),
        (date(2026, 8, 15), "Independence Day", "Low"),
        (date(2026, 8, 26), "Onam", "Medium"),
        (date(2026, 9, 14), "Ganesh Chaturthi", "Medium"),
        (date(2026, 10, 17), "Durga Puja", "High"),
        (date(2026, 10, 20), "Dussehra", "High"),
        (date(2026, 11, 8), "Diwali", "High"),
        (date(2026, 12, 25), "Christmas", "Low"),
    ]
    m: dict[date, tuple[str, str]] = {}
    for d, name, inten in raw:
        w = monday_of(d)
        m[w] = (name, inten)
        if inten == "High":
            prev = w - timedelta(days=7)
            m.setdefault(prev, (name, "Medium"))
    return m


def public_holiday_weeks() -> set[date]:
    days = [
        date(2024, 1, 26), date(2025, 1, 26), date(2026, 1, 26),
        date(2023, 10, 2), date(2024, 10, 2), date(2025, 10, 2), date(2026, 10, 2),
        date(2024, 8, 15), date(2025, 8, 15), date(2026, 8, 15),
        date(2024, 1, 26),
    ]
    return {monday_of(d) for d in days}


def shutdown_weeks() -> set[date]:
    # Diwali week plus a May annual-maintenance week — not a 90% demand wipe-out.
    diwali = [
        date(2023, 11, 13), date(2024, 11, 4), date(2025, 10, 20), date(2026, 11, 9),
    ]
    may = [date(2024, 5, 20), date(2025, 5, 19), date(2026, 5, 18)]
    return {monday_of(d) for d in diwali + may}


def festival_intensity_num(s: str | None) -> float:
    return {"High": 0.85, "Medium": 0.55, "Low": 0.25}.get(s or "", 0.0)


def system_season(system: str, month: int) -> float:
    """Calendar / vehicle-service seasonality by system. Mean ~1.0."""
    monsoon = month in (6, 7, 8, 9)
    summer = month in (4, 5, 6)
    winter = month in (12, 1, 2)
    festive = month in (9, 10, 11)
    if system == "HVAC":
        return 1.45 if summer else 0.82
    if system == "Cooling":
        return 1.35 if summer else 0.88
    if system == "Braking":
        return 1.28 if monsoon else 0.94
    if system == "Suspension":
        return 1.22 if monsoon or winter else 0.95
    if system == "Body":
        return 1.25 if festive or monsoon else 0.92
    if system == "Lighting":
        return 1.30 if festive or winter else 0.93
    if system == "Electrical":
        return 1.18 if winter else 0.96
    if system == "Engine":
        return 1.16 if monsoon else 0.97
    if system == "Transmission":
        return 1.10 if festive else 0.98
    if system == "Steering":
        return 1.14 if monsoon else 0.97
    return 1.0


def sbc_class(values: list[float | None]) -> tuple[str | None, float | None, float | None]:
    xs = [v for v in values if v is not None]
    if len(xs) < 26:
        return None, None, None
    nz = [v for v in xs if v > 0]
    if len(nz) < 3:
        adi = len(xs) / max(len(nz), 1)
        return ("Lumpy" if adi >= 1.32 else "Smooth"), round(adi, 3), 1.0
    adi = len(xs) / len(nz)
    mu = sum(nz) / len(nz)
    var = sum((x - mu) ** 2 for x in nz) / (len(nz) - 1) if len(nz) > 1 else 0.0
    cv2 = (math.sqrt(var) / mu) ** 2 if mu > 0 else 0.0
    if adi < 1.32 and cv2 < 0.49:
        c = "Smooth"
    elif adi < 1.32 and cv2 >= 0.49:
        c = "Erratic"
    elif adi >= 1.32 and cv2 < 0.49:
        c = "Intermittent"
    else:
        c = "Lumpy"
    return c, round(adi, 3), round(cv2, 3)


def draw_size(rng: random.Random, mean: float, cv: float, lumpy: bool) -> float:
    cv = max(cv, 0.05)
    sigma = math.sqrt(math.log(1 + cv * cv))
    mu = math.log(max(mean, 1.0)) - 0.5 * sigma * sigma
    if lumpy and rng.random() < 0.28:
        return rng.lognormvariate(mu + math.log(rng.uniform(4.0, 12.0)), sigma * 0.7)
    return rng.lognormvariate(mu, sigma)


def clamp_nonneg_int(x: float) -> int:
    if x < 0:
        return 0
    return int(round(x))


def build_parts(rng: random.Random) -> list[dict]:
    parts = []
    used = set()
    n_target = 160
    vid = 0
    while len(parts) < n_target:
        comp = COMPONENTS[len(parts) % len(COMPONENTS)]
        model, vtype = VEHICLES[vid % len(VEHICLES)]
        vid += 1
        key = (comp[0], model)
        variant_n = 1 + sum(1 for p in parts if p["component"] == comp[0] and p["vehicle_model"] == model)
        variant = f"V{variant_n}"
        if (comp[0], model, variant) in used:
            continue
        used.add((comp[0], model, variant))
        pid = f"P{len(parts) + 1:04d}"
        lo, hi = comp[7]
        cost = round(rng.uniform(lo, hi), 2)
        crit_bias = comp[8]
        qty_lo, qty_hi = comp[9]
        r = rng.random()
        if crit_bias == "High":
            crit = "High" if r < 0.55 else ("Medium" if r < 0.9 else "Low")
        elif crit_bias == "Low":
            crit = "Low" if r < 0.55 else ("Medium" if r < 0.9 else "High")
        else:
            crit = "Medium" if r < 0.5 else ("High" if r < 0.75 else "Low")
        make = rng.random() < 0.16
        imported = (not make) and rng.random() < 0.22
        origin = "India" if not imported else rng.choice(["CN", "DE", "JP", "KR", "TH"])
        name = f"{comp[0]} - {model} {vtype}" + (f" ({variant})" if variant_n > 1 else "")
        src = "sourced in-house" if make else "sourced from suppliers"
        desc = (
            f"{comp[0]} for {model} ({vtype}); {comp[1].lower()} system, "
            f"{comp[2].lower()} part, {comp[3]}; {src}."
        )
        parts.append({
            "part_id": pid,
            "part_number": f"SP-{SYS_CODE[comp[1]]}-{int(pid[1:]):04d}",
            "part_name": name,
            "component": comp[0],
            "variant": variant,
            "part_description": desc,
            "vehicle_system": comp[1],
            "part_category": comp[2],
            "material": comp[3],
            "vehicle_model": model,
            "vehicle_type": vtype,
            "criticality": crit,
            "service_level_target": CRIT_SL[crit],
            "make_or_buy": "Make" if make else "Buy",
            "is_imported": 1 if imported else 0,
            "country_of_origin": origin,
            "unit_cost": cost,
            "unit_of_measure": comp[4],
            "standard_pack_qty": comp[5],
            "shelf_life_days": comp[6],
            "base_weekly_qty": (qty_lo, qty_hi),
            "default_sbc": comp[10],
        })
    # Unique names already by construction; fix any collision
    seen = {}
    for p in parts:
        if p["part_name"] in seen:
            p["part_name"] = f"{p['part_name']} ({p['part_id']})"
        seen[p["part_name"]] = 1
    return parts


def assign_lifecycle_and_patterns(parts: list[dict], rng: random.Random) -> None:
    n = len(parts)
    # Lifecycle statuses (catalogue)
    idx = list(range(n))
    rng.shuffle(idx)
    obsolete = set(idx[:2])
    new_ids = set(idx[2:14])
    phase = set(idx[14:30])
    classes = [p["default_sbc"] for p in parts]
    # Keep a usable count of every SBC class even after lifecycle overrides.
    counts = {c: classes.count(c) for c in ("Smooth", "Erratic", "Intermittent", "Lumpy")}
    need = {c: max(0, 28 - counts[c]) for c in counts}
    for i, p in enumerate(parts):
        if sum(need.values()) == 0:
            break
        for c, k in need.items():
            if k and classes[i] != c and p["default_sbc"] in ("Erratic", "Smooth"):
                classes[i] = c
                need[c] -= 1
                break
    life_pat = (
        ["maturity"] * 88
        + ["growth"] * 16
        + ["introduction"] * 12
        + ["decline"] * 18
        + ["fad"] * 8
        + ["dead"] * 6
        + ["project"] * 12
    )
    while len(life_pat) < n:
        life_pat.append("maturity")
    rng.shuffle(life_pat)

    launch_choices = mondays(date(2025, 11, 3), date(2026, 4, 27))
    for i, p in enumerate(parts):
        p["intended_sbc_class"] = classes[i]
        p["lifecycle_pattern"] = life_pat[i]
        p["launch_week"] = ""
        if i in obsolete:
            p["lifecycle_status"] = "Obsolete"
            p["lifecycle_pattern"] = "decline"
            p["intended_sbc_class"] = "Lumpy"
        elif i in new_ids:
            p["lifecycle_status"] = "New"
            p["lifecycle_pattern"] = "introduction"
            p["launch_week"] = launch_choices[rng.randrange(len(launch_choices))].isoformat()
        elif i in phase or p["lifecycle_pattern"] == "decline":
            p["lifecycle_status"] = "Phase-out"
            p["lifecycle_pattern"] = "decline"
        else:
            p["lifecycle_status"] = "Active"

        tags = []
        if p["intended_sbc_class"] in ("Intermittent", "Lumpy"):
            tags.append("slow_moving")
        if rng.random() < 0.18:
            tags.append("promo_sensitive")
        if p["lifecycle_pattern"] == "project":
            tags.append("project_based")
            p["intended_sbc_class"] = "Lumpy"
        if rng.random() < 0.12:
            tags.append("dealer_batching")
        if rng.random() < 0.08:
            tags.append("bullwhip")
        if rng.random() < 0.06:
            tags.append("panic_sensitive")
        if rng.random() < 0.10:
            tags.append("step_change")
        if p["make_or_buy"] == "Make":
            tags.append("dependent_demand")
        if p["unit_cost"] > 8000 and p["intended_sbc_class"] in ("Intermittent", "Lumpy"):
            tags.append("mto")
        else:
            tags.append("mts")
        if p["lifecycle_pattern"] == "dead":
            tags.append("dead_stock")
            p["intended_sbc_class"] = "Lumpy"
        p["demand_driver_tags"] = "|".join(sorted(set(tags)))
        p["fulfilment_mode"] = "MTO" if "mto" in tags else "MTS"
        p["regions"] = list(REGIONS)
        if "mto" in tags:
            p["regions"] = rng.sample(REGIONS, k=3)

    # Substitution pairs: same component, old vs new variant
    by_comp = defaultdict(list)
    for p in parts:
        if p["lifecycle_status"] == "Active":
            by_comp[p["component"]].append(p)
    pairs = 0
    for comp, group in by_comp.items():
        if pairs >= 5:
            break
        if len(group) < 2:
            continue
        a, b = group[0], group[1]
        a["substitution_group"] = f"SUB{pairs+1:02d}"
        b["substitution_group"] = f"SUB{pairs+1:02d}"
        a["demand_driver_tags"] += "|cannibalized"
        b["demand_driver_tags"] += "|substitute_new"
        b["lifecycle_pattern"] = "growth"
        pairs += 1
    for p in parts:
        p.setdefault("substitution_group", "")

    # Multi-modal: extra tag on some erratic/lumpy
    for p in parts:
        if p["intended_sbc_class"] in ("Erratic", "Lumpy") and rng.random() < 0.35:
            if "multi_modal" not in p["demand_driver_tags"]:
                p["demand_driver_tags"] += "|multi_modal"


def lifecycle_curve(pattern: str, t: float, status: str) -> float:
    """t in [0,1] over the part's observed history."""
    if pattern == "introduction" or status == "New":
        return 0.15 + 0.95 * min(1.0, t / 0.7)
    if pattern == "growth":
        return 0.55 + 0.9 * t
    if pattern == "decline" or status in ("Phase-out", "Obsolete"):
        return max(0.08, 1.15 - 1.05 * t)
    if pattern == "fad":
        peak = math.exp(-((t - 0.35) ** 2) / (2 * 0.08 ** 2))
        return 0.15 + 1.6 * peak
    if pattern == "dead":
        return 0.12
    if pattern == "project":
        return 0.7
    return 1.0 + 0.08 * t  # mild trend


def build_calendar(weeks: list[date]) -> list[dict]:
    fest = build_festival_map()
    hol = public_holiday_weeks()
    shut = shutdown_weeks()
    fest_weeks = sorted(fest)
    rows = []
    last_shut = None
    for w in weeks:
        name, inten = fest.get(w, ("", ""))
        # nearest festival distances
        nxt = [f for f in fest_weeks if f >= w]
        prv = [f for f in fest_weeks if f <= w]
        to_next = (nxt[0] - w).days / 7 if nxt else None
        since = (w - prv[-1]).days / 7 if prv else None
        is_shut = 1 if w in shut else 0
        if is_shut:
            last_shut = w
            work = 0
        else:
            work = 5 if w in hol else 6
        since_shut = None if last_shut is None else (w - last_shut).days / 7
        month = w.month
        q, fy = fiscal(w)
        rows.append({
            "week_start_date": w,
            "year": w.year,
            "month": month,
            "quarter_fy": q,
            "fiscal_year": fy,
            "week_of_year": iso_week(w),
            "week_of_month": ((w.day - 1) // 7) + 1,
            "festival_flag": 1 if name else 0,
            "festival_name": name,
            "festival_intensity": inten,
            "weeks_to_next_festival": to_next,
            "weeks_since_last_festival": since,
            "public_holiday_flag": 1 if w in hol else 0,
            "plant_shutdown_flag": is_shut,
            "plant_working_days": work,
            "weeks_since_last_shutdown": since_shut,
            "monsoon_flag": 1 if month in (6, 7, 8, 9) else 0,
            "summer_peak_flag": 1 if month in (4, 5, 6) else 0,
            "fiscal_year_end_flag": 1 if month == 3 else 0,
            "dealer_order_cycle_flag": 1 if ((w.day - 1) // 7) == 0 else 0,
            "split_role": split_role(w),
        })
    return rows


def market_index(w: date, rng_event: random.Random, events: set[date]) -> tuple[float, int]:
    # Slow cycle + monsoon dip + festive lift
    t = (w - HIST_START).days / 7
    cycle = 0.08 * math.sin(2 * math.pi * t / 78.0)
    month = w.month
    season = 0.06 if month in (9, 10, 11) else (-0.05 if month in (7, 8) else 0.0)
    event = 1 if w in events else 0
    shock = 0.12 if event else 0.0
    idx = 1.0 + cycle + season + shock
    return round(max(0.75, min(1.25, idx)), 4), event


def generate_series_demand(
    rng: random.Random,
    cal: list[dict],
    part: dict,
    region: str,
    start_week: date,
    end_week: date,
    promo_weeks: set[date],
    campaign_span: list[tuple[date, date]],
    events: set[date],
    parc0: float,
    age0: float,
    cannibal_from: date | None,
) -> list[dict]:
    intended = part["intended_sbc_class"]
    tags = part["demand_driver_tags"]
    # Occurrence probability from SBC class
    if intended == "Smooth":
        p0, cv = 0.96, 0.32
    elif intended == "Erratic":
        p0, cv = 0.94, 1.05
    elif intended == "Intermittent":
        p0, cv = 0.38, 0.32
    else:
        p0, cv = 0.28, 1.15
    if "dealer_batching" in tags:
        p0 *= 0.55
    if "dead_stock" in tags:
        p0 = 0.08
        cv = 1.2
    qlo, qhi = part["base_weekly_qty"]
    mean_size = rng.uniform(qlo, qhi)
    if part["vehicle_type"] == "2W":
        mean_size *= 1.55
    if part["vehicle_type"] == "CV":
        mean_size *= 0.85
    mean_size *= REGION_SCALE[region]
    if "mto" in tags:
        mean_size *= 0.5
        p0 = min(p0, 0.22)

    step_week = None
    step_mult = 1.0
    if "step_change" in tags:
        cands = [c["week_start_date"] for c in cal if WARMUP_END < c["week_start_date"] <= date(2025, 12, 1)]
        if cands:
            step_week = rng.choice(cands)
            step_mult = rng.choice([0.55, 0.7, 1.45, 1.8])

    rows = []
    relevant = [c for c in cal if start_week <= c["week_start_date"] <= end_week]
    n = max(len(relevant), 1)
    # Pre-draw occurrence then sizes, then retune to hit class on train_val
    for attempt in range(10):
        demand = []
        for i, c in enumerate(relevant):
            w = c["week_start_date"]
            t = i / n
            p = p0
            if "dealer_batching" in tags and c["dealer_order_cycle_flag"] == 0:
                p *= 0.08
            if "project_based" in tags:
                p = 0.12
            occ = rng.random() < p
            if not occ:
                demand.append(0)
                continue
            size = draw_size(rng, mean_size, cv, lumpy=("multi_modal" in tags or intended == "Lumpy"))
            size *= lifecycle_curve(part["lifecycle_pattern"], t, part["lifecycle_status"])
            size *= system_season(part["vehicle_system"], c["month"])
            size *= REGION_SCALE[region] ** 0  # already in mean
            # festival
            fi = festival_intensity_num(c["festival_intensity"])
            fname = c["festival_name"]
            scope = FESTIVAL_SCOPE.get(fname)
            if scope and region not in scope:
                fi *= 0.15
            elif fname == "Diwali" and region in ("North", "West"):
                fi *= 1.15
            fest_boost = 1.0 + fi * (0.55 if part["vehicle_system"] in ("Lighting", "Body", "Braking") or part["vehicle_type"] == "2W" else 0.25)
            size *= fest_boost
            # shutdown: dependent/Make hit harder; aftermarket continues
            if c["plant_shutdown_flag"]:
                size *= 0.40 if "dependent_demand" in tags else 0.82
            else:
                size *= 0.85 + 0.15 * (c["plant_working_days"] / 6)
            # promo / campaign
            if w in promo_weeks and "promo_sensitive" in tags:
                size *= rng.uniform(1.6, 2.4)
            # post-promo dip
            if (w - timedelta(days=7)) in promo_weeks and "promo_sensitive" in tags:
                size *= 0.72
            for a, b in campaign_span:
                if a <= w <= b:
                    size *= 1.28
            mi, ev = market_index(w, rng, events)
            size *= mi
            if ev and "panic_sensitive" in tags:
                size *= rng.uniform(1.8, 3.2)
            if ev and "bullwhip" in tags:
                size *= rng.uniform(1.3, 2.1)
            if step_week and w >= step_week:
                size *= step_mult
            if cannibal_from and w >= cannibal_from and "cannibalized" in tags:
                size *= 0.55
            if cannibal_from and w >= cannibal_from and "substitute_new" in tags:
                size *= 1.65
            if "project_based" in tags and rng.random() < 0.15:
                size *= rng.uniform(6, 18)
            if "dead_stock" in tags:
                size = rng.choice([1, 2, 3, rng.randint(8, 40)])
            demand.append(clamp_nonneg_int(size))

        train_vals = []
        for c, q in zip(relevant, demand):
            if c["split_role"] in ("warmup", "train_val") and c["week_start_date"] <= TRAINVAL_END:
                train_vals.append(q)
        got, adi, cv2 = sbc_class(train_vals)
        if part["lifecycle_status"] == "New" or len(train_vals) < 26:
            break
        if got == intended:
            break
        # retune
        if intended in ("Intermittent", "Lumpy") and (got in ("Smooth", "Erratic")):
            p0 *= 0.72
        elif intended in ("Smooth", "Erratic") and got in ("Intermittent", "Lumpy"):
            p0 = min(0.99, p0 * 1.25)
        if intended in ("Erratic", "Lumpy") and got in ("Smooth", "Intermittent"):
            cv *= 1.35
        elif intended in ("Smooth", "Intermittent") and got in ("Erratic", "Lumpy"):
            cv *= 0.7

    # attach rows
    hist_map = {c["week_start_date"]: q for c, q in zip(relevant, demand)}
    for i, c in enumerate(relevant):
        w = c["week_start_date"]
        q = hist_map[w]
        mi, ev = market_index(w, rng, events)
        parc = parc0 * (1 + 0.00035 * ((w - HIST_START).days / 7))
        age = age0 + 0.019 * ((w - HIST_START).days / 7 / 52)
        promo = 1 if w in promo_weeks else 0
        camp = 1 if any(a <= w <= b for a, b in campaign_span) else 0
        future = c["split_role"] == "future"
        rows.append({
            **c,
            "part_id": part["part_id"],
            "region": region,
            "demand_qty": None if future else q,
            "promotion_flag": promo,
            "promotion_intensity": round(rng.uniform(0.18, 0.55), 3) if promo else 0.0,
            "campaign_flag": camp,
            "campaign_intensity": round(0.32, 3) if camp else 0.0,
            "market_event_flag": None if future else ev,
            "market_condition_index": mi,
            "vehicle_parc": round(parc, 0),
            "avg_vehicle_age_years": round(age, 2),
            "raw_demand": None if future else q,
        })
    return rows


def simulate_outcomes(rng: random.Random, series_rows: list[dict], service_level: float) -> None:
    hist = [r for r in series_rows if r["split_role"] != "future" and r["demand_qty"] is not None]
    avg = sum(r["demand_qty"] for r in hist[-26:]) / max(len(hist[-26:]), 1)
    stock = max(avg * 6, 2)
    for r in series_rows:
        if r["split_role"] == "future" or r["demand_qty"] is None:
            r["out_consumption_qty"] = None
            r["out_backorder_qty"] = None
            r["out_lost_sales_qty"] = None
            r["out_stockout_flag"] = None
            r["out_spike_flag"] = None
            r["out_unusual_demand_flag"] = None
            continue
        d = r["demand_qty"]
        # replenishment noise
        stock += rng.uniform(0.6, 1.1) * avg
        cons = min(d, int(stock))
        stock -= cons
        short = d - cons
        lost = int(round(short * (0.35 if service_level >= 0.99 else 0.55)))
        bo = short - lost
        r["out_consumption_qty"] = cons
        r["out_backorder_qty"] = bo
        r["out_lost_sales_qty"] = lost
        r["out_stockout_flag"] = 1 if cons < d else 0
        r["out_spike_flag"] = 0
        r["out_unusual_demand_flag"] = 0


def cap_and_flags(series_rows: list[dict]) -> None:
    tv = [r["demand_qty"] for r in series_rows if r["split_role"] in ("warmup", "train_val") and r["demand_qty"] is not None]
    pos = [x for x in tv if x > 0]
    if len(pos) >= 8:
        sp = sorted(pos)
        cap = sp[int(0.985 * (len(sp) - 1))]
        med = sp[len(sp) // 2]
    else:
        cap, med = 10 ** 9, 0
    for r in series_rows:
        d = r["demand_qty"]
        if d is None:
            r["demand_qty_capped"] = None
            continue
        r["demand_qty_capped"] = min(d, cap)
        r["out_spike_flag"] = 1 if d > cap else 0
        r["out_unusual_demand_flag"] = 1 if (med and d > 3 * med) else 0


def main() -> None:
    rng = random.Random(SEED)
    weeks = mondays(HIST_START, FUTURE_END)
    cal = build_calendar(weeks)
    parts = build_parts(rng)
    assign_lifecycle_and_patterns(parts, rng)

    # Shared promo / campaign / market-event calendar (known drivers)
    hist_weeks = [c["week_start_date"] for c in cal if c["split_role"] != "future"]
    promo_weeks = set(rng.sample(hist_weeks, k=22))
    promo_weeks.update({date(2026, 11, 2), date(2026, 11, 9)})  # planned Diwali campaign
    # planned campaign windows (including one in future — known plan)
    campaign_span = [
        (date(2024, 2, 5), date(2024, 2, 26)),
        (date(2024, 10, 7), date(2024, 10, 28)),
        (date(2025, 2, 3), date(2025, 2, 24)),
        (date(2025, 9, 1), date(2025, 9, 22)),
        (date(2026, 2, 2), date(2026, 2, 23)),
        (date(2026, 10, 5), date(2026, 10, 26)),  # future planned campaign
    ]
    events = set(rng.sample(hist_weeks, k=12))

    # Vehicle parc by model-region
    parc = {}
    age = {}
    for model, vtype in VEHICLES:
        base = {"2W": 900000, "CV": 180000, "4W SUV": 420000, "4W Sedan": 360000, "4W Hatch": 400000}[vtype]
        for reg in REGIONS:
            parc[(model, reg)] = base * REGION_SCALE[reg] * rng.uniform(0.85, 1.15)
            age[(model, reg)] = rng.uniform(3.8, 7.2)

    cannibal_from = {}
    for p in parts:
        if p.get("substitution_group") and "substitute_new" in p["demand_driver_tags"]:
            cannibal_from[p["substitution_group"]] = date(2025, 4, 7)

    all_rows: list[dict] = []
    series_index: dict[tuple, list[dict]] = {}
    for p in parts:
        start = HIST_START
        end = FUTURE_END
        if p["lifecycle_status"] == "New" and p["launch_week"]:
            start = date.fromisoformat(p["launch_week"])
        if p["lifecycle_status"] == "Obsolete":
            end = date(2026, 3, 23)  # no recent/future demand
        cf = cannibal_from.get(p.get("substitution_group") or "", None)
        for reg in p["regions"]:
            srows = generate_series_demand(
                rng, cal, p, reg, start, end, promo_weeks, campaign_span, events,
                parc[(p["vehicle_model"], reg)], age[(p["vehicle_model"], reg)], cf,
            )
            simulate_outcomes(rng, srows, p["service_level_target"])
            cap_and_flags(srows)
            series_index[(p["part_id"], reg)] = srows
            all_rows.extend(srows)

    # Missing weeks (data quality) — blank target, not in future
    hist_idx = [i for i, r in enumerate(all_rows) if r["split_role"] != "future"]
    miss_n = 72
    for i in rng.sample(hist_idx, k=miss_n):
        r = all_rows[i]
        r["is_missing_week"] = 1
        r["demand_qty"] = None
        r["demand_qty_capped"] = None
        r["out_consumption_qty"] = None
        r["out_backorder_qty"] = None
        r["out_lost_sales_qty"] = None
        r["out_stockout_flag"] = None
        r["out_spike_flag"] = None
        r["out_unusual_demand_flag"] = None
    for r in all_rows:
        r.setdefault("is_missing_week", 0)

    # lag52
    key_map = {(r["part_id"], r["region"], r["week_start_date"]): r for r in all_rows}
    for r in all_rows:
        prev = r["week_start_date"] - timedelta(days=364)
        src = key_map.get((r["part_id"], r["region"], prev))
        r["demand_lag52"] = src["demand_qty"] if src and src["demand_qty"] is not None else None

    # Classify on train_val only
    class_map = {}
    volume = defaultdict(float)
    revenue = defaultdict(float)
    for (pid, reg), srows in series_index.items():
        tv = [
            r["demand_qty"]
            for r in srows
            if r["week_start_date"] <= TRAINVAL_END and r["is_missing_week"] == 0 and r["demand_qty"] is not None
        ]
        c, adi, cv2 = sbc_class(tv)
        class_map[(pid, reg)] = (c, adi, cv2)
        volume[pid] += sum(x for x in tv if x)
    part_by_id = {p["part_id"]: p for p in parts}
    for pid, p in part_by_id.items():
        revenue[pid] = volume[pid] * p["unit_cost"]
    ranked = sorted(revenue, key=lambda k: revenue[k], reverse=True)
    n = len(ranked)
    abc = {}
    for i, pid in enumerate(ranked):
        if i < 0.2 * n:
            abc[pid] = "A"
        elif i < 0.5 * n:
            abc[pid] = "B"
        else:
            abc[pid] = "C"

    # part_type from train_val volume / class
    vols = [volume[p["part_id"]] for p in parts]
    vs = sorted(vols)
    hi = vs[int(0.67 * (len(vs) - 1))] if vs else 0
    lo = vs[int(0.33 * (len(vs) - 1))] if vs else 0
    for p in parts:
        classes = [class_map[k][0] for k in class_map if k[0] == p["part_id"] and class_map[k][0]]
        if p["lifecycle_status"] == "New" or not classes:
            p["part_type"] = "Unclassified (new)"
        elif all(c in ("Intermittent", "Lumpy") for c in classes):
            p["part_type"] = "Intermittent" if volume[p["part_id"]] <= lo else "Slow Moving"
        elif volume[p["part_id"]] >= hi:
            p["part_type"] = "Fast Moving"
        elif volume[p["part_id"]] <= lo:
            p["part_type"] = "Slow Moving"
        else:
            p["part_type"] = "Medium"
        p["abc_class"] = abc[p["part_id"]]
        # planning priority follows criticality, A-class can lift it
        if p["criticality"] == "High" or p["abc_class"] == "A":
            p["planning_priority"] = "High"
        elif p["criticality"] == "Low" and p["abc_class"] == "C":
            p["planning_priority"] = "Low"
        else:
            p["planning_priority"] = "Medium"

    # Write D1
    d1_cols = [
        "week_start_date", "part_id", "region", "split_role", "demand_qty", "demand_qty_capped",
        "is_missing_week", "year", "month", "quarter_fy", "fiscal_year", "week_of_year", "week_of_month",
        "festival_flag", "festival_name", "festival_intensity", "weeks_to_next_festival",
        "weeks_since_last_festival", "public_holiday_flag", "plant_shutdown_flag", "plant_working_days",
        "weeks_since_last_shutdown", "monsoon_flag", "summer_peak_flag", "fiscal_year_end_flag",
        "dealer_order_cycle_flag", "promotion_flag", "promotion_intensity", "campaign_flag",
        "campaign_intensity", "market_event_flag", "market_condition_index", "vehicle_parc",
        "avg_vehicle_age_years", "demand_lag52", "part_name", "component", "vehicle_system", "part_category",
        "vehicle_model", "vehicle_type", "criticality", "abc_class", "part_type", "lifecycle_status",
        "make_or_buy", "is_imported", "unit_cost", "fulfilment_mode", "demand_driver_tags",
        "lifecycle_pattern", "substitution_group", "intended_sbc_class", "demand_class_trainval",
        "adi_trainval", "cv2_trainval", "out_consumption_qty", "out_backorder_qty", "out_lost_sales_qty",
        "out_stockout_flag", "out_spike_flag", "out_unusual_demand_flag",
    ]
    d1_path = OUT / "01_demand_forecasting.csv"
    with d1_path.open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=d1_cols, extrasaction="ignore")
        wri.writeheader()
        for r in all_rows:
            p = part_by_id[r["part_id"]]
            c, adi, cv2 = class_map[(r["part_id"], r["region"])]
            out = {
                **r,
                "week_start_date": r["week_start_date"].isoformat(),
                "part_name": p["part_name"],
                "component": p["component"],
                "vehicle_system": p["vehicle_system"],
                "part_category": p["part_category"],
                "vehicle_model": p["vehicle_model"],
                "vehicle_type": p["vehicle_type"],
                "criticality": p["criticality"],
                "abc_class": p["abc_class"],
                "part_type": p["part_type"],
                "lifecycle_status": p["lifecycle_status"],
                "make_or_buy": p["make_or_buy"],
                "is_imported": p["is_imported"],
                "unit_cost": p["unit_cost"],
                "fulfilment_mode": p["fulfilment_mode"],
                "demand_driver_tags": p["demand_driver_tags"],
                "lifecycle_pattern": p["lifecycle_pattern"],
                "substitution_group": p["substitution_group"],
                "intended_sbc_class": p["intended_sbc_class"],
                "demand_class_trainval": c or "",
                "adi_trainval": adi if adi is not None else "",
                "cv2_trainval": cv2 if cv2 is not None else "",
            }
            wri.writerow(out)

    # ---- D3 catalogue ----
    d3_cols = [
        "part_id", "part_number", "part_name", "component", "variant", "part_description",
        "vehicle_system", "part_category", "material", "part_type", "vehicle_model", "vehicle_type",
        "criticality", "service_level_target", "planning_priority", "abc_class", "make_or_buy",
        "is_imported", "country_of_origin", "lifecycle_status", "launch_week", "unit_cost",
        "unit_of_measure", "standard_pack_qty", "shelf_life_days", "fulfilment_mode",
        "demand_driver_tags", "lifecycle_pattern", "substitution_group",
    ]
    d3_path = OUT / "03_spare_parts_catalogue.csv"
    with d3_path.open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=d3_cols, extrasaction="ignore")  # drop generator-only fields
        wri.writeheader()
        for p in parts:
            wri.writerow(p)

    # ---- D2 scheduling snapshot ----
    suppliers = []
    for i in range(1, 16):
        suppliers.append({
            "id": f"S{i:03d}", "type": "Domestic", "country": "India",
            "lead": rng.randint(7, 28),
            "cap": 0,  # fill later
            "commit": round(rng.uniform(55, 82), 1),
            "otd": round(rng.uniform(78, 97), 1),
            "delay": round(rng.uniform(0.4, 9.5), 1),
            "reject": round(rng.uniform(0.2, 6.5), 2),
        })
    for i in range(16, 23):
        ctry = rng.choice(["CN", "DE", "JP", "KR", "TH"])
        suppliers.append({
            "id": f"S{i:03d}", "type": "Import", "country": ctry,
            "lead": rng.randint(50, 110),
            "cap": 0,
            "commit": round(rng.uniform(58, 80), 1),
            "otd": round(rng.uniform(80, 96), 1),
            "delay": round(rng.uniform(0.3, 8.0), 1),
            "reject": round(rng.uniform(0.3, 3.5), 2),
        })
    for s in suppliers:
        s["rel"] = round(max(60.0, s["otd"] - s["reject"] - rng.uniform(0, 6)), 1)

    # monthly demand last 12 months for capacity scaling
    last12_start = date(2025, 10, 6)
    monthly = defaultdict(float)
    weekly_avg = defaultdict(list)
    for r in all_rows:
        if r["split_role"] == "future" or r["demand_qty"] is None:
            continue
        if r["week_start_date"] >= last12_start:
            mon = r["week_start_date"].replace(day=1)
            monthly[(r["part_id"], mon)] += r["demand_qty"]
        if r["week_start_date"] >= date(2026, 6, 29):
            weekly_avg[r["part_id"]].append(r["demand_qty"])
    part_month_demand = defaultdict(float)
    for (pid, mon), q in monthly.items():
        part_month_demand[pid] += q
    months_n = 12
    part_mavg = {pid: part_month_demand[pid] / months_n for pid in part_by_id}

    d2_rows = []
    slot_load = defaultdict(float)  # supplier -> load
    assignments = {}
    for p in parts:
        if p["lifecycle_status"] == "Obsolete":
            continue
        nsup = 1 if rng.random() < 0.35 else (2 if rng.random() < 0.7 else 3)
        if p["make_or_buy"] == "Make":
            pool = [s for s in suppliers if s["type"] == "Domestic"]
        elif p["is_imported"]:
            pool = [s for s in suppliers if s["type"] == "Import"]
        else:
            pool = [s for s in suppliers if s["type"] == "Domestic"]
        chosen = rng.sample(pool, k=min(nsup, len(pool)))
        if nsup == 1:
            sobs = [100.0]
        elif nsup == 2:
            a = rng.choice([60.0, 70.0, 80.0])
            sobs = [a, 100.0 - a]
        else:
            sobs = [50.0, 30.0, 20.0]
        chosen = [c for _, c in sorted(zip(sobs, chosen), key=lambda x: -x[0])]
        sobs = sorted(sobs, reverse=True)
        assignments[p["part_id"]] = list(zip(chosen, sobs))
        mavg = max(part_mavg.get(p["part_id"], 20), 1)
        for s, sob in zip(chosen, sobs):
            slot_load[s["id"]] += mavg * sob / 100

    # set capacity so median load/available ~ 0.55-0.8
    for s in suppliers:
        load = slot_load.get(s["id"], 50)
        avail_target = load / rng.uniform(0.45, 0.95)
        s["cap"] = max(200, int(avail_target / max(0.15, 1 - s["commit"] / 100)))

    for p in parts:
        if p["lifecycle_status"] == "Obsolete":
            continue
        mavg = max(part_mavg.get(p["part_id"], 8), 1)
        wavg = (sum(weekly_avg.get(p["part_id"], [mavg / 4])) / max(len(weekly_avg.get(p["part_id"], [1])), 1))
        wavg = max(wavg, 0.2)
        cover_w = rng.choice([0.4, 0.8, 1.2, 2.0, 3.5, 5.0, 8.0, 12.0])
        on_hand = clamp_nonneg_int(wavg * cover_w * rng.uniform(0.8, 1.2))
        reserved = clamp_nonneg_int(on_hand * rng.uniform(0.0, 0.12))
        blocked = clamp_nonneg_int(on_hand * rng.uniform(0.0, 0.04))
        if reserved + blocked > on_hand:
            reserved, blocked = 0, 0
        available = on_hand - reserved - blocked
        backorder = 0 if rng.random() < 0.82 else clamp_nonneg_int(wavg * rng.uniform(0.2, 2.0))
        in_transit = 0 if rng.random() < 0.45 else clamp_nonneg_int(wavg * rng.uniform(0.5, 3.0))
        open_po = 0 if rng.random() < 0.4 else clamp_nonneg_int(wavg * rng.uniform(0.8, 4.0))
        due_this = int(round(open_po * rng.uniform(0.3, 1.0))) if open_po else 0
        due_later = open_po - due_this
        eta = "" if open_po == 0 else (AS_OF + timedelta(days=rng.randint(5, 75))).isoformat()
        emergency = 1 if (p["criticality"] == "High" and available < wavg * 2 and rng.random() < 0.45) else (1 if rng.random() < 0.08 else 0)
        ss = clamp_nonneg_int(wavg * rng.uniform(1.5, 4.5) * (1.4 if p["criticality"] == "High" else 1.0))
        rop = ss + clamp_nonneg_int(wavg * rng.uniform(2.0, 6.0))
        row = {
            "as_of_date": AS_OF.isoformat(),
            "part_id": p["part_id"],
            "supply_mode": "Production" if p["make_or_buy"] == "Make" else "Procurement",
            "lifecycle_status": p["lifecycle_status"],
            "criticality": p["criticality"],
            "service_level_target": p["service_level_target"],
            "planning_priority": p["planning_priority"],
            "unit_cost": p["unit_cost"],
            "carrying_cost_rate": round(rng.uniform(0.18, 0.28), 3),
            "on_hand_qty": on_hand,
            "reserved_qty": reserved,
            "blocked_qty": blocked,
            "available_qty": available,
            "backorder_qty": backorder,
            "in_transit_qty": in_transit,
            "open_po_qty": open_po,
            "open_po_due_this_month_qty": due_this,
            "open_po_due_later_qty": due_later,
            "earliest_open_po_eta": eta,
            "emergency_po_open_flag": emergency,
            "current_safety_stock_qty": ss,
            "current_reorder_point_qty": rop,
        }
        slots = assignments[p["part_id"]]
        for k in (1, 2, 3):
            prefix = f"sup{k}_"
            if k <= len(slots):
                s, sob = slots[k - 1]
                moq = max(1, int(rng.choice([1, 5, 10, 20, 50, 100])))
                # cap MOQ at ~6 weeks demand
                moq = min(moq, max(1, int(wavg * 6)))
                mult = rng.choice([1, 1, 1, 5, 10])
                row.update({
                    prefix + "id": s["id"],
                    prefix + "type": s["type"],
                    prefix + "country": s["country"],
                    prefix + "sob_pct": sob,
                    prefix + "lead_time_days": s["lead"],
                    prefix + "moq": moq,
                    prefix + "order_multiple": mult,
                    prefix + "monthly_capacity": s["cap"],
                    prefix + "capacity_committed_pct": s["commit"],
                    prefix + "otd_pct": s["otd"],
                    prefix + "avg_delay_days": s["delay"],
                    prefix + "reject_pct": s["reject"],
                    prefix + "reliability_pct": s["rel"],
                })
            else:
                for fld in ["id", "type", "country", "sob_pct", "lead_time_days", "moq", "order_multiple",
                            "monthly_capacity", "capacity_committed_pct", "otd_pct", "avg_delay_days",
                            "reject_pct", "reliability_pct"]:
                    row[prefix + fld] = ""
        d2_rows.append(row)

    d2_cols = [
        "as_of_date", "part_id", "supply_mode", "lifecycle_status", "criticality", "service_level_target",
        "planning_priority", "unit_cost", "carrying_cost_rate", "on_hand_qty", "reserved_qty", "blocked_qty",
        "available_qty", "backorder_qty", "in_transit_qty", "open_po_qty", "open_po_due_this_month_qty",
        "open_po_due_later_qty", "earliest_open_po_eta", "emergency_po_open_flag", "current_safety_stock_qty",
        "current_reorder_point_qty",
    ]
    for k in (1, 2, 3):
        d2_cols += [
            f"sup{k}_id", f"sup{k}_type", f"sup{k}_country", f"sup{k}_sob_pct", f"sup{k}_lead_time_days",
            f"sup{k}_moq", f"sup{k}_order_multiple", f"sup{k}_monthly_capacity", f"sup{k}_capacity_committed_pct",
            f"sup{k}_otd_pct", f"sup{k}_avg_delay_days", f"sup{k}_reject_pct", f"sup{k}_reliability_pct",
        ]
    d2_path = OUT / "02_scheduling_inputs.csv"
    with d2_path.open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=d2_cols)
        wri.writeheader()
        for row in d2_rows:
            wri.writerow(row)

    # Summary
    mix = defaultdict(int)
    for (pid, reg), (c, adi, cv2) in class_map.items():
        mix[c or "short"] += 1
    print("Wrote", d1_path, "rows", len(all_rows))
    print("Wrote", d2_path, "rows", len(d2_rows))
    print("Wrote", d3_path, "rows", len(parts))
    print("SBC mix (series):", dict(mix))
    print("parts", len(parts), "series", len(series_index))
    print("weeks", len(weeks), "hist", sum(1 for c in cal if c["split_role"] != "future"), "future", sum(1 for c in cal if c["split_role"] == "future"))


if __name__ == "__main__":
    main()
