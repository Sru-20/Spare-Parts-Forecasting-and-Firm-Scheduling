import csv
import hashlib
import json
import math
import re
import uuid
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from backend.planning.models import KPIResult, Part, ResultReference, Run, Warning
from backend.planning.services.results import file_sha256


RUN_ID = uuid.uuid5(
    uuid.NAMESPACE_URL, "spare-parts-demand-scheduling:phase9-phase6-snapshot-v1"
)
CANDIDATE_ID = "lightgbm_quantile::random_1"
SCHEDULER_ID = "risk_adjusted_mrp"
INITIALIZATION = "origin_visible"
SCHEDULE_RELATIVE = Path("results/phase6_provisional_firm_schedule_v1.csv")
KPI_RELATIVE = Path("results/phase6_kpi_comparison_v1.csv")
REPORT_RELATIVE = Path("reports/phase6_comparison_report.md")
FORECAST_RELATIVE = Path("results/phase9_ui_forecasts_phase6_v1.csv")
FORECAST_FIELDS = (
    "part_id",
    "part_name",
    "region",
    "week",
    "forecast_qty",
    "criticality",
    "part_category",
    "candidate_id",
    "model",
    "forecast_role",
    "forecast_status",
    "source_schedule_run_id",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise CommandError(f"Required saved Phase 6 evidence is missing: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(reader.fieldnames) > 256:
                raise CommandError(f"Invalid CSV schema in {path}")
            return list(reader.fieldnames), list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CommandError(f"Unable to read saved Phase 6 evidence: {path}: {error}") from error


def _number(value: str | None, *, field: str, allow_missing: bool = False) -> float | None:
    if value is None or value.strip() == "":
        if allow_missing:
            return None
        raise CommandError(f"Required numeric field {field} is blank")
    try:
        result = float(value)
    except ValueError:
        if allow_missing and value.strip().casefold() in {"na", "n/a", "not_measurable", "null"}:
            return None
        raise CommandError(f"Invalid numeric field {field}: {value!r}")
    if not math.isfinite(result):
        raise CommandError(f"Non-finite numeric field {field}: {value!r}")
    return result


def _forecast_rows(schedule_rows: list[dict[str, str]], parts: dict[str, Part]) -> list[dict[str, str]]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in schedule_rows:
        if row.get("candidate_id") != CANDIDATE_ID:
            raise CommandError("Saved schedule contains an unexpected forecast candidate")
        if row.get("scheduler_id") != SCHEDULER_ID:
            raise CommandError("Saved schedule contains an unexpected scheduling policy")
        part_id = (row.get("part_id") or "").strip()
        week = (row.get("week") or "").strip()
        if not part_id or not week:
            raise CommandError("Saved schedule is missing part_id or week")
        try:
            forecast = float(row.get("forecast_qty") or "")
        except ValueError as error:
            raise CommandError(f"Invalid forecast_qty for {part_id}, week {week}") from error
        if not math.isfinite(forecast) or forecast < 0:
            raise CommandError(f"Invalid forecast_qty for {part_id}, week {week}")
        part = parts.get(part_id)
        if part is None:
            raise CommandError(
                f"Part {part_id} is not in the imported catalogue; run import_catalogue first"
            )
        key = (part_id, week)
        derived = {
            "part_id": part_id,
            "part_name": row.get("part_name") or part.part_name,
            "region": row.get("region_plant") or "ALL",
            "week": week,
            "forecast_qty": format(forecast, ".12g"),
            "criticality": row.get("criticality") or part.criticality,
            "part_category": part.category,
            "candidate_id": CANDIDATE_ID,
            "model": CANDIDATE_ID,
            "forecast_role": "phase6_end_to_end_champion",
            "forecast_status": "PROVISIONAL",
            "source_schedule_run_id": row.get("run_id") or "",
        }
        existing = selected.get(key)
        if existing and existing["forecast_qty"] != derived["forecast_qty"]:
            raise CommandError(
                f"Conflicting saved forecast quantities for {part_id}, week {week}"
            )
        selected.setdefault(key, derived)
    if not selected:
        raise CommandError("The saved schedule contains no forecast rows")
    return sorted(selected.values(), key=lambda row: (row["part_id"], int(row["week"])))


def _write_forecast_once(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FORECAST_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
    except FileExistsError:
        existing_headers, existing_rows = _read_csv(path)
        if tuple(existing_headers) != FORECAST_FIELDS or existing_rows != rows:
            raise CommandError(
                f"Refusing to overwrite a different derived forecast artifact: {path}"
            )
    except OSError as error:
        raise CommandError(f"Unable to create derived forecast artifact: {path}: {error}") from error
    return file_sha256(path)


class Command(BaseCommand):
    help = (
        "Register the saved Phase 6 validation champion and provisional schedule "
        "for the Phase 9 UI. Reads only saved Phase 6 outputs and the imported catalogue."
    )

    def handle(self, *args, **options):
        base = settings.BASE_DIR
        schedule_path = base / SCHEDULE_RELATIVE
        kpi_path = base / KPI_RELATIVE
        report_path = base / REPORT_RELATIVE
        schedule_headers, schedule_rows = _read_csv(schedule_path)
        required_schedule_fields = {
            "run_id",
            "candidate_id",
            "scheduler_id",
            "part_id",
            "part_name",
            "week",
            "forecast_qty",
            "order_qty",
            "status",
            "failed_checks",
        }
        if required_schedule_fields - set(schedule_headers):
            raise CommandError("Saved Phase 6 schedule is missing required columns")
        if not report_path.is_file():
            raise CommandError(f"Phase 6 report is missing: {report_path}")
        report = report_path.read_text(encoding="utf-8")
        config_match = re.search(
            r"Config SHA-256:\s*`([a-fA-F0-9]{64})`", report
        )
        if not config_match:
            raise CommandError("Phase 6 report does not contain its configuration hash")
        config_hash = config_match.group(1).lower()

        all_part_ids = {row["part_id"] for row in schedule_rows if row.get("part_id")}
        parts = {
            part.part_id: part
            for part in Part.objects.filter(part_id__in=all_part_ids)
        }
        if set(parts) != all_part_ids:
            missing = sorted(all_part_ids - set(parts))
            raise CommandError(
                f"Catalogue is missing schedule parts: {', '.join(missing[:8])}; run import_catalogue first"
            )
        forecasts = _forecast_rows(schedule_rows, parts)

        kpi_headers, kpi_rows = _read_csv(kpi_path)
        required_kpi_fields = {
            "candidate_id",
            "scheduler_id",
            "initialization",
            "kpi",
            "formula",
            "unit",
            "basis",
            "denominator",
            "value",
            "baseline_value",
            "absolute_difference",
            "percentage_difference",
            "percentage_difference_status",
            "worse_than_baseline",
        }
        if required_kpi_fields - set(kpi_headers):
            raise CommandError("Saved Phase 6 KPI evidence is missing required columns")
        selected_kpis = [
            row
            for row in kpi_rows
            if row.get("candidate_id") == CANDIDATE_ID
            and row.get("scheduler_id") == SCHEDULER_ID
            and row.get("initialization") == INITIALIZATION
        ]
        if not selected_kpis:
            raise CommandError("No matching origin-visible Phase 6 KPI rows were found")
        if len({row.get("kpi") for row in selected_kpis}) != len(selected_kpis):
            raise CommandError("Duplicate KPI names exist for the selected Phase 6 pair")
        for row in selected_kpis:
            for field in ("value", "baseline_value", "absolute_difference", "percentage_difference"):
                _number(row[field], field=f"{row['kpi']}.{field}", allow_missing=True)

        forecast_path = base / FORECAST_RELATIVE
        forecast_hash = _write_forecast_once(forecast_path, forecasts)

        schedule_hash = file_sha256(schedule_path)
        kpi_hash = file_sha256(kpi_path)
        data_digest = hashlib.sha256(
            f"{schedule_hash}\n{kpi_hash}\n".encode("ascii")
        ).hexdigest()
        checker_path = base / "src" / "scheduling" / "checker.py"
        if not checker_path.is_file():
            raise CommandError(f"Scheduling checker source is missing: {checker_path}")
        checker_hash = file_sha256(checker_path)

        existing = Run.objects.filter(run_id=RUN_ID).first()
        if existing:
            references = {
                item.kind: item
                for item in existing.results.all()
            }
            if (
                existing.data_hash != data_digest
                or existing.config_hash != config_hash
                or references.get(ResultReference.Kind.FORECAST) is None
                or references[ResultReference.Kind.FORECAST].file_hash != forecast_hash
                or references.get(ResultReference.Kind.SCHEDULE) is None
                or references[ResultReference.Kind.SCHEDULE].file_hash != schedule_hash
            ):
                raise CommandError("Registered Phase 6 snapshot differs; refusing overwrite")
            self.stdout.write(self.style.SUCCESS(f"Phase 6 snapshot already registered: {RUN_ID}"))
            return

        with transaction.atomic():
            run = Run.objects.create(
                run_id=RUN_ID,
                data_version="phase6_validation_and_provisional_future_v1",
                data_hash=data_digest,
                config_version="phase6_config_v1",
                config_hash=config_hash,
                model_version=CANDIDATE_ID,
                horizon_weeks=14,
                scheduling_policy=SCHEDULER_ID,
                checker_version=checker_hash,
                status=Run.Status.COMPLETED,
                output_location=SCHEDULE_RELATIVE.as_posix(),
            )
            ResultReference.objects.create(
                run=run,
                kind=ResultReference.Kind.FORECAST,
                relative_path=FORECAST_RELATIVE.as_posix(),
                file_hash=forecast_hash,
                version=1,
            )
            ResultReference.objects.create(
                run=run,
                kind=ResultReference.Kind.SCHEDULE,
                relative_path=SCHEDULE_RELATIVE.as_posix(),
                file_hash=schedule_hash,
                version=1,
            )
            for row in selected_kpis:
                value = _number(row["value"], field=f"{row['kpi']}.value", allow_missing=True)
                details = {
                    "formula": row["formula"],
                    "denominator_basis": row["denominator"],
                    "candidate_id": CANDIDATE_ID,
                    "scheduler_id": SCHEDULER_ID,
                    "initialization": INITIALIZATION,
                    "comparison_basis": "Phase 6 origin-visible validation; simulated data",
                    "source_file": KPI_RELATIVE.as_posix(),
                    "source_file_hash": kpi_hash,
                    "baseline_value": _number(
                        row["baseline_value"],
                        field=f"{row['kpi']}.baseline_value",
                        allow_missing=True,
                    ),
                    "absolute_difference": _number(
                        row["absolute_difference"],
                        field=f"{row['kpi']}.absolute_difference",
                        allow_missing=True,
                    ),
                    "percentage_difference": _number(
                        row["percentage_difference"],
                        field=f"{row['kpi']}.percentage_difference",
                        allow_missing=True,
                    ),
                    "percentage_difference_status": row["percentage_difference_status"],
                    "worse_than_baseline": row["worse_than_baseline"].strip().casefold() == "true",
                }
                KPIResult.objects.create(
                    run=run,
                    name=row["kpi"],
                    value=value,
                    unit=row["unit"],
                    basis=row["basis"],
                    denominator=None,
                    details=details,
                )
            for row in schedule_rows:
                if (row.get("status") or "").casefold() != "unresolved":
                    continue
                part = parts[row["part_id"]]
                failed_checks = [
                    item.strip()
                    for item in re.split(r"[|;,]", row.get("failed_checks") or "")
                    if item.strip()
                ]
                Warning.objects.create(
                    run=run,
                    part=part,
                    code="phase6_unresolved_order",
                    severity="high" if (row.get("criticality") or "").casefold() == "high" else "medium",
                    evidence={
                        "status": "unresolved",
                        "failed_checks": failed_checks,
                        "week": row.get("week"),
                        "supplier_id": row.get("supplier_id"),
                        "order_qty": row.get("order_qty"),
                        "candidate_id": CANDIDATE_ID,
                        "schedule_source": SCHEDULE_RELATIVE.as_posix(),
                        "schedule_hash": schedule_hash,
                    },
                    recommendation=(
                        "Review the recorded hard-constraint failures before any release; "
                        "this unresolved row is not executable."
                    ),
                )

        self.stdout.write(
            self.style.SUCCESS(
                json.dumps(
                    {
                        "run_id": str(RUN_ID),
                        "forecast_rows": len(forecasts),
                        "schedule_rows": len(schedule_rows),
                        "kpi_records": len(selected_kpis),
                        "unresolved_warning_rows": sum(
                            (row.get("status") or "").casefold() == "unresolved"
                            for row in schedule_rows
                        ),
                        "forecast_artifact": FORECAST_RELATIVE.as_posix(),
                        "final_test_accessed": False,
                        "schedule_status": "provisional",
                    },
                    sort_keys=True,
                )
            )
        )
