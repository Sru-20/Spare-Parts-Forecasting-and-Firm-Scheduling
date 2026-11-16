from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone
from rest_framework.exceptions import APIException, PermissionDenied, ValidationError

from backend.accounts.models import User
from backend.planning.models import (
    AuditChainHead,
    AuditRecord,
    KPIResult,
    Part,
    Proposal,
    ProposalVersion,
    ResultReference,
    Run,
)
from backend.planning.services.results import (
    file_sha256,
    read_csv_records,
    resolve_result_path,
)
from src.data.loader import ROOT, load_scheduling
from src.scheduling.checker import check_schedule
from src.simulation.weekly import simulate_weekly


class ProposalConflict(APIException):
    status_code = 409
    default_code = "proposal_conflict"
    default_detail = "Proposal can no longer be applied"


class InfeasibleProposal(ValidationError):
    pass


def _snapshot(proposal: Proposal) -> dict[str, Any]:
    return {
        "version": proposal.version,
        "status": proposal.status,
        "source_run_version": proposal.source_run_version,
        "source_schedule_hash": proposal.source_schedule_hash,
        "intent": proposal.intent,
        "before_values": proposal.before_values,
        "after_values": proposal.after_values,
        "validation": proposal.validation,
        "impact": proposal.impact,
        "warnings": proposal.warnings,
        "expires_at": proposal.expires_at.isoformat(),
    }


def _record_version(proposal: Proposal) -> None:
    ProposalVersion.objects.create(
        proposal=proposal,
        version=proposal.version,
        status=proposal.status,
        snapshot=_snapshot(proposal),
    )


def append_audit(
    *,
    actor: User,
    action: str,
    proposal: Proposal | None = None,
    run: Run | None = None,
    before_values: Any = None,
    after_values: Any = None,
    validation: Any = None,
    details: Any = None,
) -> AuditRecord:
    head = AuditChainHead.objects.select_for_update().get(head_id="global")
    previous_hash = head.last_hash
    audit_id = uuid.uuid4()
    body = {
        "audit_id": str(audit_id),
        "action": action,
        "actor_id": actor.pk,
        "proposal_id": str(proposal.proposal_id) if proposal else None,
        "run_id": str(run.run_id) if run else None,
        "before_values": before_values or [],
        "after_values": after_values or [],
        "validation": validation or {},
        "details": details or {},
        "previous_hash": previous_hash,
    }
    record_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    record = AuditRecord.objects.create(
        audit_id=audit_id,
        action=action,
        actor=actor,
        proposal=proposal,
        run=run,
        before_values=body["before_values"],
        after_values=body["after_values"],
        validation=body["validation"],
        details=body["details"],
        previous_hash=previous_hash,
        record_hash=record_hash,
    )
    head.last_hash = record_hash
    head.version += 1
    head.save(update_fields=("last_hash", "version"))
    return record


def _source_schedule(run_id: Any) -> tuple[Run, ResultReference, list[str], list[dict[str, str]]]:
    run = Run.objects.select_for_update().filter(run_id=run_id).first()
    if run is None or run.status != Run.Status.COMPLETED:
        raise ValidationError({"run_id": "A completed planning run is required"})
    reference = run.results.filter(kind=ResultReference.Kind.SCHEDULE).order_by(
        "-version"
    ).first()
    if reference is None:
        raise ValidationError({"run_id": "The run has no saved schedule"})
    headers, rows = read_csv_records(reference)
    required = {"part_id", "supplier_id", "release_week", "order_qty", "status"}
    if required - set(headers):
        raise ValidationError({"schedule": "Saved schedule is missing required fields"})
    if not rows:
        raise ValidationError({"schedule": "Saved schedule contains no rows"})
    return run, reference, headers, rows


def _supplier_for_part(part: dict[str, Any], supplier_id: str) -> tuple[int, dict[str, Any]]:
    for slot in (1, 2, 3):
        if str(part.get(f"sup{slot}_id") or "") == supplier_id:
            try:
                supplier = {
                    "lead_time_days": float(part[f"sup{slot}_lead_time_days"]),
                    "moq": float(part[f"sup{slot}_moq"]),
                    "order_multiple": float(part[f"sup{slot}_order_multiple"]),
                }
            except (TypeError, ValueError, KeyError) as error:
                raise ValidationError(
                    {"supplier_id": "Supplier constraints are incomplete"}
                ) from error
            if not all(math.isfinite(value) for value in supplier.values()):
                raise ValidationError(
                    {"supplier_id": "Supplier constraints must be finite"}
                )
            return slot, supplier
    raise ValidationError(
        {"supplier_id": f"Supplier {supplier_id!r} is not valid for this part"}
    )


def _d2_parts() -> dict[str, dict[str, Any]]:
    d2 = load_scheduling(ROOT / "data" / "02_scheduling_inputs.csv")
    return {
        str(row["part_id"]): {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
        }
        for row in d2.to_dict(orient="records")
    }


def _changed_quantity(current: float, percentage: float, minimum: float, multiple: float) -> int:
    if not math.isfinite(current) or current < 0:
        raise ValidationError("Schedule has an invalid order quantity")
    if multiple <= 0 or minimum < 0:
        raise ValidationError("Supplier MOQ or order multiple is invalid")
    raw = current * (1.0 + percentage / 100.0)
    if raw <= 0:
        return 0
    return int(max(minimum, math.ceil((raw - 1e-12) / multiple) * multiple))


def _validate_candidate(
    headers: list[str], rows: list[dict[str, str]], d2_parts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    order_rows = []
    for row in rows:
        try:
            quantity = float(row.get("order_qty") or 0)
        except ValueError as error:
            raise ValidationError("Schedule contains a non-numeric order quantity") from error
        if quantity <= 0:
            continue
        part_id = str(row.get("part_id") or "")
        part = d2_parts.get(part_id)
        if part is None:
            raise ValidationError(f"No validated scheduling input exists for {part_id}")
        supplier_id = str(row.get("supplier_id") or "")
        _, supplier = _supplier_for_part(part, supplier_id)
        release_week = int(float(row.get("release_week") or 0))
        order_rows.append(
            {
                "part_id": part_id,
                "supplier_id": supplier_id,
                "release_week": release_week,
                "lead_time_days": supplier["lead_time_days"],
                "arrival_week": release_week
                + max(1, math.ceil(supplier["lead_time_days"] / 7.0)),
                "order_qty": quantity,
            }
        )
    if not order_rows:
        return {
            "feasible": True,
            "failed_checks": [],
            "checked_orders": 0,
            "status": "firm",
        }
    check = check_schedule(
        pd.DataFrame(order_rows),
        d2_parts,
        horizon_weeks=14,
        check_requirements=False,
    )
    return check


def _simulate_part(
    candidate: list[dict[str, str]],
    part_id: str,
    changed_orders: list[dict[str, Any]],
    d2_part: dict[str, Any],
    original_rows: list[dict[str, str]],
) -> dict[str, Any]:
    indices = [
        index for index, row in enumerate(candidate) if str(row.get("part_id")) == part_id
    ]
    weekly: dict[int, dict[str, float]] = {}
    for index in indices:
        row = candidate[index]
        try:
            week = int(float(row["week"]))
            current = {
                key: float(row[key])
                for key in (
                    "forecast_qty",
                    "opening_usable_stock",
                    "expected_receipts",
                )
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError(
                {"schedule": f"Cannot recalculate simulated inventory for {part_id}"}
            ) from error
        if week in weekly and any(
            not math.isclose(weekly[week][key], current[key], abs_tol=1e-8)
            for key in current
        ):
            raise ValidationError(
                {"schedule": f"Conflicting weekly values for {part_id}, week {week}"}
            )
        weekly[week] = current
    if set(weekly) != set(range(1, 15)):
        raise ValidationError(
            {"schedule": f"Fourteen complete forecast weeks are required for {part_id}"}
        )

    original_by_week = {
        week: int(round(values["expected_receipts"]))
        for week, values in weekly.items()
    }
    after_by_week = dict(original_by_week)
    for change in changed_orders:
        row_index = int(change["row_index"])
        original_order = original_rows[row_index]
        part = d2_part
        _, supplier = _supplier_for_part(
            part, str(original_order.get("supplier_id") or "")
        )
        release_week = int(float(original_order["release_week"]))
        arrival_week = release_week + max(
            1, math.ceil(supplier["lead_time_days"] / 7.0)
        )
        if 1 <= arrival_week <= 14:
            delta = int(change["order_qty"]) - int(original_order["order_qty"])
            after_by_week[arrival_week] += delta
            if after_by_week[arrival_week] < 0:
                raise ValidationError(
                    {"schedule": "Adjusted receipts cannot become negative"}
                )

    initial_stock = int(round(weekly[1]["opening_usable_stock"]))
    demand = [weekly[week]["forecast_qty"] for week in range(1, 15)]
    if initial_stock < 0 or any(
        not math.isfinite(value) or value < 0 for value in demand
    ):
        raise ValidationError(
            {"schedule": "Initial inventory and demand must be finite and non-negative"}
        )
    criticality = str(d2_part.get("criticality") or "")
    common = {
        "initial_stock": initial_stock,
        "demand": demand,
        "criticality": criticality,
        "initial_backlog": 0,
        "unit_cost": float(d2_part.get("unit_cost") or 0),
        "carrying_cost_rate": float(d2_part.get("carrying_cost_rate") or 0.25),
    }
    before_simulation = simulate_weekly(
        **common,
        receipts=[
            {"week": week, "quantity": quantity}
            for week, quantity in original_by_week.items()
            if quantity > 0
        ],
    )
    after_simulation = simulate_weekly(
        **common,
        receipts=[
            {"week": week, "quantity": quantity}
            for week, quantity in after_by_week.items()
            if quantity > 0
        ],
    )
    for simulation_row in after_simulation.to_dict(orient="records"):
        week = int(simulation_row["week"])
        for index in indices:
            if int(float(candidate[index]["week"])) == week:
                candidate[index]["opening_usable_stock"] = str(
                    int(simulation_row["opening_stock"])
                )
                candidate[index]["expected_receipts"] = str(
                    int(simulation_row["receipts_qty"])
                )
                candidate[index]["projected_inventory"] = str(
                    int(simulation_row["closing_stock"])
                )
                candidate[index]["backlog_qty"] = str(
                    int(simulation_row["backlog_qty"])
                )

    def summary(frame: pd.DataFrame) -> dict[str, float]:
        total_demand = float(frame["demand_qty"].sum())
        fulfilled = float(frame["fulfilled_qty"].sum())
        return {
            "fill_rate": fulfilled / total_demand if total_demand else 0.0,
            "stockout_qty": float(frame["stockout_qty"].sum()),
            "emergency_order_qty": float(frame["emergency_order_qty"].sum()),
            "carrying_cost_proxy": float(frame["carrying_cost_proxy"].sum()),
            "ending_inventory": float(frame["closing_stock"].iloc[-1]),
            "demand_qty": total_demand,
        }

    return {
        "part_id": part_id,
        "before": summary(before_simulation),
        "after": summary(after_simulation),
        "basis": "simulated_forecast_demand_path",
    }


def _simulate_affected_parts(
    original_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    changes: list[dict[str, Any]],
    d2_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changes_by_part: dict[str, list[dict[str, Any]]] = {}
    for change in changes:
        changes_by_part.setdefault(str(change["part_id"]), []).append(change)
    return [
        _simulate_part(
            candidate_rows,
            part_id,
            part_changes,
            d2_parts[part_id],
            original_rows,
        )
        for part_id, part_changes in changes_by_part.items()
    ]


def create_proposal(
    *,
    actor: User,
    payload: dict[str, Any],
    idempotency_key: str,
) -> tuple[Proposal, bool]:
    if not actor.is_planner:
        raise PermissionDenied("Planner role is required to propose a change")
    if not idempotency_key or len(idempotency_key) > 128:
        raise ValidationError({"Idempotency-Key": "A valid idempotency key is required"})

    existing = Proposal.objects.filter(
        proposer=actor, idempotency_key=idempotency_key
    ).first()
    if existing:
        if (
            existing.request_text != payload["request_text"]
            or existing.intent.get("percent_change")
            != str(payload["percent_change"])
            or existing.intent.get("category") != payload.get("category")
            or existing.intent.get("part_ids") != sorted(payload.get("part_ids", []))
        ):
            raise ProposalConflict("Idempotency key was already used for another request")
        return existing, False

    percent = float(payload["percent_change"])
    with transaction.atomic():
        run, reference, headers, rows = _source_schedule(payload["run_id"])
        source_hash = reference.file_hash
        path = resolve_result_path(reference)
        if file_sha256(path) != source_hash:
            raise ProposalConflict("Source schedule changed; proposal must be recalculated")
        if payload.get("category"):
            part_ids = set(
                Part.objects.filter(category__iexact=payload["category"]).values_list(
                    "part_id", flat=True
                )
            )
            if not part_ids:
                raise ValidationError({"category": "No catalogue parts match this category"})
            category = payload["category"]
        else:
            part_ids = set(payload["part_ids"])
            found = set(Part.objects.filter(part_id__in=part_ids).values_list("part_id", flat=True))
            if found != part_ids:
                raise ValidationError({"part_ids": "One or more parts are unknown"})
            category = None

        d2_parts = _d2_parts()
        before_values = []
        after_values = []
        total_before = 0
        total_after = 0
        selected_parts = set()
        for row_index, row in enumerate(rows):
            part_id = str(row.get("part_id") or "")
            if part_id not in part_ids:
                continue
            try:
                current = float(row.get("order_qty") or 0)
            except ValueError as error:
                raise ValidationError("Schedule contains a non-numeric order quantity") from error
            if not math.isfinite(current) or current < 0:
                raise ValidationError("Schedule has an invalid order quantity")
            if current <= 0:
                continue
            if not current.is_integer():
                raise ValidationError("Schedule order quantities must be whole units")
            if str(row.get("status", "")).casefold() != "firm":
                raise InfeasibleProposal(
                    {"part_id": f"Part {part_id} has a non-firm order in the saved schedule"}
                )
            part = d2_parts.get(part_id)
            if part is None:
                raise ValidationError(f"No validated scheduling input exists for {part_id}")
            _, supplier = _supplier_for_part(part, str(row.get("supplier_id") or ""))
            revised = _changed_quantity(
                current,
                percent,
                supplier["moq"],
                supplier["order_multiple"],
            )
            if revised != int(current):
                selected_parts.add(part_id)
                before_values.append(
                    {
                        "row_index": row_index,
                        "part_id": part_id,
                        "supplier_id": row["supplier_id"],
                        "week": row.get("week"),
                        "order_qty": int(current),
                    }
                )
                after_values.append(
                    {
                        "row_index": row_index,
                        "part_id": part_id,
                        "supplier_id": row["supplier_id"],
                        "week": row.get("week"),
                        "order_qty": revised,
                    }
                )
                total_before += int(current)
                total_after += revised
        if not after_values:
            raise ValidationError(
                {"proposal": "No firm executable order matches the requested change"}
            )

        candidate = [dict(row) for row in rows]
        for change in after_values:
            candidate[change["row_index"]]["order_qty"] = str(change["order_qty"])
        validation = _validate_candidate(headers, candidate, d2_parts)
        if not validation["feasible"]:
            raise InfeasibleProposal(
                {
                    "feasible": False,
                    "failed_checks": validation["failed_checks"],
                    "status": "unresolved",
                }
            )
        simulated_impacts = _simulate_affected_parts(
            rows, candidate, after_values, d2_parts
        )
        expires_at = timezone.now() + timedelta(
            minutes=int(getattr(settings, "PROPOSAL_EXPIRY_MINUTES", 60))
        )
        proposal = Proposal(
            source_run=run,
            source_result=reference,
            source_schedule_hash=source_hash,
            source_run_version=run.version,
            request_text=payload["request_text"],
            intent={
                "action": "adjust_firm_order_quantities",
                "part_ids": sorted(selected_parts),
                "category": category,
                "percent_change": str(payload["percent_change"]),
            },
            before_values=before_values,
            after_values=after_values,
            validation=validation,
            impact={
                "basis": "simulated_forecast_demand_path_affected_parts",
                "orders_changed": len(after_values),
                "parts_changed": len(selected_parts),
                "order_qty_before": total_before,
                "order_qty_after": total_after,
                "order_qty_delta": total_after - total_before,
                "simulated_part_impacts": simulated_impacts,
            },
            warnings=[
                "Proposal affects a simulated schedule only; no ERP or purchase-order system is connected."
            ],
            proposer=actor,
            expires_at=expires_at,
            idempotency_key=idempotency_key,
        )
        try:
            with transaction.atomic():
                proposal.save()
        except IntegrityError:
            existing = Proposal.objects.get(proposer=actor, idempotency_key=idempotency_key)
            return existing, False
        _record_version(proposal)
        append_audit(
            actor=actor,
            action="proposal_created",
            proposal=proposal,
            run=run,
            before_values=before_values,
            after_values=after_values,
            validation=validation,
            details={"idempotency_key": idempotency_key},
        )
    return proposal, True


def _write_versioned_schedule(
    run: Run,
    source: ResultReference,
    headers: list[str],
    rows: list[dict[str, str]],
    changes: list[dict[str, Any]],
    expected_values: list[dict[str, Any]],
    expected_impacts: list[dict[str, Any]] | None = None,
) -> tuple[ResultReference, list[dict[str, Any]]]:
    path = resolve_result_path(source)
    if file_sha256(path) != source.file_hash:
        raise ProposalConflict("Schedule changed after proposal validation")
    original_rows = [dict(row) for row in rows]
    candidate_rows = [dict(row) for row in rows]
    expected_by_index = {
        int(item["row_index"]): int(item["order_qty"]) for item in expected_values
    }
    for change in changes:
        index = int(change["row_index"])
        if not 0 <= index < len(candidate_rows):
            raise ProposalConflict("Proposal row no longer exists in the source schedule")
        if str(candidate_rows[index].get("part_id")) != str(change["part_id"]):
            raise ProposalConflict("Proposal part does not match the source row")
        current_qty = int(float(candidate_rows[index].get("order_qty") or 0))
        if expected_by_index.get(index) != current_qty:
            raise ProposalConflict("Source order quantity changed after proposal validation")
        candidate_rows[index]["order_qty"] = str(change["order_qty"])

    d2_parts = _d2_parts()
    impacts = _simulate_affected_parts(
        original_rows, candidate_rows, changes, d2_parts
    )
    if expected_impacts is not None and impacts != expected_impacts:
        raise ProposalConflict(
            "Recomputed simulation differs from the validated proposal"
        )
    candidate_check = _validate_candidate(headers, candidate_rows, d2_parts)
    if not candidate_check["feasible"]:
        raise InfeasibleProposal(
            {"failed_checks": candidate_check["failed_checks"], "status": "unresolved"}
        )

    current_version = run.results.filter(
        kind=ResultReference.Kind.SCHEDULE
    ).order_by("-version").values_list("version", flat=True).first()
    next_version = int(current_version or source.version) + 1
    relative = (
        Path("results")
        / "api_simulated_proposals"
        / str(run.run_id)
        / f"schedule_v{next_version}_{uuid.uuid4().hex}.csv"
    )
    destination = settings.BASE_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        writer.writerows(candidate_rows)
        stream.flush()
        os.fsync(stream.fileno())
    return (
        ResultReference.objects.create(
            run=run,
            kind=ResultReference.Kind.SCHEDULE,
            relative_path=relative.as_posix(),
            file_hash=file_sha256(destination),
            version=next_version,
        ),
        impacts,
    )


def _write_schedule_impact_kpis(
    run: Run,
    before_values: list[dict[str, Any]],
    after_values: list[dict[str, Any]],
    impacts: list[dict[str, Any]],
) -> None:
    before = sum(int(item["order_qty"]) for item in before_values)
    after = sum(int(item["order_qty"]) for item in after_values)
    KPIResult.objects.create(
        run=run,
        name="proposal_order_quantity",
        value=float(after),
        unit="units",
        basis="simulated",
        denominator=None,
        details={
            "calculation": "sum of validated firm order_qty values changed by this proposal",
            "before": before,
            "after": after,
            "delta": after - before,
            "full_service_kpi_recalculation": False,
            "kpi_scope": "affected_parts_only",
        },
    )
    kpi_fields = {
        "fill_rate": ("ratio", "simulated"),
        "stockout_qty": ("units", "simulated"),
        "emergency_order_qty": ("units", "simulated"),
        "carrying_cost_proxy": ("INR", "proxy"),
        "ending_inventory": ("units", "simulated"),
    }
    for impact in impacts:
        part_id = impact["part_id"]
        for field, (unit, basis) in kpi_fields.items():
            old_value = float(impact["before"][field])
            new_value = float(impact["after"][field])
            KPIResult.objects.create(
                run=run,
                name=f"proposal_{part_id}_{field}",
                value=new_value,
                unit=unit,
                basis=basis,
                denominator=(
                    float(impact["after"]["demand_qty"])
                    if field == "fill_rate"
                    else None
                ),
                details={
                    "part_id": part_id,
                    "before": old_value,
                    "after": new_value,
                    "delta": new_value - old_value,
                    "calculation": "src.simulation.weekly.simulate_weekly",
                    "demand_basis": "saved forecast path",
                    "currency": "INR" if unit == "INR" else None,
                },
            )


def confirm_proposal(
    *,
    actor: User,
    proposal_id: Any,
    proposal_version: int,
    run_version: int,
    idempotency_key: str,
) -> Proposal:
    if not actor.is_approver:
        raise PermissionDenied("Approver role is required to confirm a proposal")
    if not idempotency_key or len(idempotency_key) > 128:
        raise ValidationError({"Idempotency-Key": "A valid idempotency key is required"})
    result: Proposal | None = None
    conflict: str | None = None
    with transaction.atomic():
        proposal = (
            Proposal.objects.select_for_update()
            .select_related("source_run", "source_result", "proposer")
            .filter(proposal_id=proposal_id)
            .first()
        )
        if proposal is None:
            raise ProposalConflict("Proposal not found")
        if proposal.status != Proposal.Status.VALIDATED:
            raise ProposalConflict(f"Proposal is already {proposal.status}")
        if proposal.proposer_id == actor.pk and not getattr(
            settings, "ALLOW_SELF_APPROVAL", False
        ):
            raise PermissionDenied("A proposer cannot approve their own proposal")
        if proposal.version != proposal_version:
            raise ProposalConflict("Proposal version changed; reload before confirming")
        run = Run.objects.select_for_update().get(run_id=proposal.source_run_id)
        latest = run.results.filter(kind=ResultReference.Kind.SCHEDULE).order_by(
            "-version"
        ).first()
        if run.version != run_version or run.version != proposal.source_run_version:
            conflict = "Schedule run version changed; proposal is stale"
        elif latest is None or latest.pk != proposal.source_result_id:
            conflict = "Source schedule is no longer the latest version"
        elif latest.file_hash != proposal.source_schedule_hash:
            conflict = "Source schedule hash changed; proposal is stale"
        elif proposal.expires_at <= timezone.now():
            proposal.status = Proposal.Status.EXPIRED
            proposal.version += 1
            proposal.save(update_fields=("status", "version", "updated_at"))
            _record_version(proposal)
            append_audit(
                actor=actor, action="proposal_expired", proposal=proposal, run=run
            )
            conflict = "Proposal expired"
        else:
            headers, rows = read_csv_records(latest)
            if file_sha256(resolve_result_path(latest)) != proposal.source_schedule_hash:
                conflict = "Source schedule file changed; proposal is stale"
            else:
                candidate = [dict(row) for row in rows]
                for change in proposal.after_values:
                    candidate[int(change["row_index"])]["order_qty"] = str(
                        change["order_qty"]
                    )
                candidate_check = _validate_candidate(
                    headers, candidate, _d2_parts()
                )
                if not candidate_check["feasible"]:
                    proposal.status = Proposal.Status.REJECTED
                    proposal.version += 1
                    proposal.save(
                        update_fields=("status", "version", "updated_at")
                    )
                    _record_version(proposal)
                    append_audit(
                        actor=actor,
                        action="proposal_rejected",
                        proposal=proposal,
                        run=run,
                        validation=candidate_check,
                    )
                    conflict = "Updated schedule violates hard constraints"
                else:
                    claimed = Run.objects.filter(
                        run_id=run.run_id, version=run_version
                    ).update(version=run_version + 1)
                    if claimed != 1:
                        conflict = "Schedule run version changed; proposal is stale"
                    else:
                        run.version = run_version + 1
                        try:
                            reference, impacts = _write_versioned_schedule(
                                run,
                                latest,
                                headers,
                                rows,
                                proposal.after_values,
                                proposal.before_values,
                                proposal.impact["simulated_part_impacts"],
                            )
                        except InfeasibleProposal as error:
                            proposal.status = Proposal.Status.REJECTED
                            proposal.version += 1
                            proposal.save(
                                update_fields=("status", "version", "updated_at")
                            )
                            _record_version(proposal)
                            append_audit(
                                actor=actor,
                                action="proposal_rejected",
                                proposal=proposal,
                                run=run,
                                validation={"detail": str(error.detail)},
                            )
                            conflict = "Updated schedule violates hard constraints"
                        else:
                            before = proposal.before_values
                            proposal.status = Proposal.Status.APPLIED
                            proposal.approver = actor
                            proposal.applied_result = reference
                            proposal.version += 1
                            proposal.save(
                                update_fields=(
                                    "status",
                                    "approver",
                                    "applied_result",
                                    "version",
                                    "updated_at",
                                )
                            )
                            _record_version(proposal)
                            _write_schedule_impact_kpis(
                                run, before, proposal.after_values, impacts
                            )
                            append_audit(
                                actor=actor,
                                action="proposal_applied",
                                proposal=proposal,
                                run=run,
                                before_values=before,
                                after_values=proposal.after_values,
                                validation=proposal.validation,
                                details={
                                    "idempotency_key": idempotency_key,
                                    "result_hash": reference.file_hash,
                                },
                            )
                            result = proposal
        if conflict and proposal.status == Proposal.Status.VALIDATED:
            proposal.status = Proposal.Status.STALE
            proposal.version += 1
            proposal.save(update_fields=("status", "version", "updated_at"))
            _record_version(proposal)
            append_audit(
                actor=actor,
                action="proposal_stale",
                proposal=proposal,
                run=proposal.source_run,
                details={"reason": conflict},
            )
    if conflict:
        raise ProposalConflict(conflict)
    if result is None:
        raise ProposalConflict("Proposal was not applied")
    return result


def cancel_proposal(*, actor: User, proposal_id: Any) -> Proposal:
    with transaction.atomic():
        proposal = Proposal.objects.select_for_update().filter(
            proposal_id=proposal_id
        ).first()
        if proposal is None:
            raise ProposalConflict("Proposal not found")
        if proposal.proposer_id != actor.pk and not actor.is_project_admin:
            raise PermissionDenied("Only the proposer or an admin can cancel")
        if proposal.status != Proposal.Status.VALIDATED:
            raise ProposalConflict(f"Proposal is already {proposal.status}")
        proposal.status = Proposal.Status.CANCELED
        proposal.version += 1
        proposal.save(update_fields=("status", "version", "updated_at"))
        _record_version(proposal)
        append_audit(
            actor=actor,
            action="proposal_canceled",
            proposal=proposal,
            run=proposal.source_run,
            details={"schedule_mutated": False},
        )
    return proposal


def revert_proposal(
    *, actor: User, proposal_id: Any, idempotency_key: str
) -> Proposal:
    if not actor.is_approver:
        raise PermissionDenied("Approver role is required to revert a proposal")
    if not idempotency_key or len(idempotency_key) > 120:
        raise ValidationError({"Idempotency-Key": "A valid idempotency key is required"})
    with transaction.atomic():
        original = (
            Proposal.objects.select_for_update()
            .select_related("source_run", "applied_result", "proposer")
            .filter(proposal_id=proposal_id)
            .first()
        )
        if original is None or original.status != Proposal.Status.APPLIED:
            raise ProposalConflict("Only an applied proposal can be reverted")
        revert_key = f"revert:{idempotency_key}"
        existing_revert = Proposal.objects.filter(
            proposer=actor, idempotency_key=revert_key
        ).first()
        if existing_revert:
            if existing_revert.reverts_proposal_id != original.pk:
                raise ProposalConflict(
                    "Idempotency key was already used to revert another proposal"
                )
            return existing_revert
        if original.proposer_id == actor.pk and not getattr(
            settings, "ALLOW_SELF_APPROVAL", False
        ):
            raise PermissionDenied("A proposer cannot approve their own revert")
        run = Run.objects.select_for_update().get(run_id=original.source_run_id)
        latest = run.results.filter(kind=ResultReference.Kind.SCHEDULE).order_by(
            "-version"
        ).first()
        if (
            latest is None
            or original.applied_result_id != latest.pk
            or run.version != original.source_run_version + 1
        ):
            raise ProposalConflict("Schedule changed since application; revert is stale")
        headers, rows = read_csv_records(latest)
        current_values = [
            {
                **{key: value for key, value in change.items() if key != "order_qty"},
                "order_qty": int(float(rows[change["row_index"]]["order_qty"])),
            }
            for change in original.after_values
        ]
        reverted_rows = [dict(row) for row in rows]
        for change in original.before_values:
            reverted_rows[int(change["row_index"])]["order_qty"] = str(
                change["order_qty"]
            )
        validation = _validate_candidate(headers, reverted_rows, _d2_parts())
        if not validation["feasible"]:
            raise InfeasibleProposal(
                {"failed_checks": validation["failed_checks"], "status": "unresolved"}
            )
        reference, impacts = _write_versioned_schedule(
            run,
            latest,
            headers,
            rows,
            original.before_values,
            original.after_values,
        )
        reverted = Proposal.objects.create(
            source_run=run,
            source_result=latest,
            source_schedule_hash=latest.file_hash,
            source_run_version=run.version,
            reverts_proposal=original,
            request_text=f"Authorized revert of proposal {original.proposal_id}",
            intent={"action": "revert", "source_proposal_id": str(original.proposal_id)},
            before_values=current_values,
            after_values=original.before_values,
            validation=validation,
            impact={
                "basis": "simulated_forecast_demand_path_affected_parts",
                "simulated_part_impacts": impacts,
            },
            warnings=["Revert changes the simulated schedule only."],
            status=Proposal.Status.APPLIED,
            proposer=actor,
            approver=actor,
            expires_at=timezone.now(),
            idempotency_key=revert_key,
            applied_result=reference,
        )
        _record_version(reverted)
        run.version += 1
        run.save(update_fields=("version",))
        _write_schedule_impact_kpis(
            run, current_values, original.before_values, impacts
        )
        append_audit(
            actor=actor,
            action="proposal_reverted",
            proposal=reverted,
            run=run,
            before_values=current_values,
            after_values=original.before_values,
            validation=reverted.validation,
            details={
                "original_proposal_id": str(original.proposal_id),
                "original_audit_ids": [
                    str(item.audit_id)
                    for item in original.audit.filter(action="proposal_applied")
                ],
                "result_hash": reference.file_hash,
                "idempotency_key": idempotency_key,
            },
        )
    return reverted
