from __future__ import annotations

from typing import Any

from rest_framework.exceptions import ValidationError

from backend.planning.models import (
    AuditRecord,
    KPIResult,
    Part,
    Proposal,
    ResultReference,
    Run,
    Warning,
)
from backend.planning.services.proposals import (
    cancel_proposal,
    confirm_proposal,
    create_proposal,
    revert_proposal,
)
from backend.planning.services.results import latest_result, paged_csv_rows


def get_schedule(run_id: str, *, part_id: str | None = None, page_size: int = 20) -> dict[str, Any]:
    reference = latest_result(run_id, ResultReference.Kind.SCHEDULE)
    filters = {"part_id": part_id} if part_id else {}
    return paged_csv_rows(reference, page=1, page_size=page_size, filters=filters)


def get_forecast(run_id: str, *, part_id: str | None = None, page_size: int = 20) -> dict[str, Any]:
    reference = latest_result(run_id, ResultReference.Kind.FORECAST)
    filters = {"part_id": part_id} if part_id else {}
    return paged_csv_rows(reference, page=1, page_size=page_size, filters=filters)


def get_inventory(run_id: str, *, part_id: str | None = None) -> dict[str, Any]:
    schedule = get_schedule(run_id, part_id=part_id, page_size=100)
    return {
        "basis": "simulated_schedule_snapshot",
        "run_id": schedule["run_id"],
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "part_id",
                    "week",
                    "opening_usable_stock",
                    "expected_receipts",
                    "projected_inventory",
                    "backlog_qty",
                    "status",
                )
                if key in row
            }
            for row in schedule["results"]
        ],
    }


def get_warnings(run_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise ValidationError({"limit": "Limit must be 1..100"})
    Run.objects.get(run_id=run_id)
    return list(
        Warning.objects.filter(run_id=run_id)
        .select_related("part")
        .order_by("-created_at")
        .values(
            "code",
            "severity",
            "week",
            "recommendation",
            "part__part_id",
            "evidence",
        )[:limit]
    )


def get_kpis(run_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise ValidationError({"limit": "Limit must be 1..100"})
    Run.objects.get(run_id=run_id)
    return list(
        KPIResult.objects.filter(run_id=run_id)
        .order_by("name", "pk")
        .values("name", "value", "unit", "basis", "denominator", "details")[:limit]
    )


def find_parts_by_category(category: str, *, limit: int = 20) -> list[dict[str, Any]]:
    if not category or len(category) > 128 or not 1 <= limit <= 100:
        raise ValidationError("Category or result limit is invalid")
    return list(
        Part.objects.filter(category__iexact=category)
        .order_by("part_id")
        .values(
            "part_id",
            "part_name",
            "category",
            "component",
            "criticality",
            "vehicle_type",
        )[:limit]
    )


def propose_change(
    actor, payload: dict[str, Any], *, idempotency_key: str
) -> Proposal:
    proposal, _ = create_proposal(
        actor=actor, payload=payload, idempotency_key=idempotency_key
    )
    return proposal


def validate_change(proposal_id: str, *, actor) -> dict[str, Any]:
    proposal = Proposal.objects.filter(proposal_id=proposal_id).first()
    if proposal is None:
        raise ValidationError({"proposal_id": "Proposal not found"})
    if proposal.proposer_id != actor.pk and not actor.is_approver and not actor.is_project_admin:
        raise ValidationError({"proposal_id": "Proposal is not visible to this user"})
    return proposal.validation


def apply_change(
    *,
    actor,
    proposal_id: str,
    proposal_version: int,
    run_version: int,
    idempotency_key: str,
    explicit_confirmation: bool,
) -> Proposal:
    if explicit_confirmation is not True:
        raise ValidationError("An explicit confirmation action is required")
    return confirm_proposal(
        actor=actor,
        proposal_id=proposal_id,
        proposal_version=proposal_version,
        run_version=run_version,
        idempotency_key=idempotency_key,
    )


def revert_change(*, actor, proposal_id: str, idempotency_key: str) -> Proposal:
    return revert_proposal(
        actor=actor, proposal_id=proposal_id, idempotency_key=idempotency_key
    )


def recalc_kpis(run_id: str) -> dict[str, Any]:
    """Return stored KPIs; live recomputation requires a validated schedule run."""
    Run.objects.get(run_id=run_id)
    return {"run_id": run_id, "kpis": get_kpis(run_id)}


def get_audit_log(*, limit: int = 50) -> list[dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise ValidationError({"limit": "Limit must be 1..100"})
    return list(
        AuditRecord.objects.select_related("actor")
        .order_by("-created_at")
        .values(
            "audit_id",
            "action",
            "actor__username",
            "proposal_id",
            "run_id",
            "created_at",
            "previous_hash",
            "record_hash",
        )[:limit]
    )


APPROVED_TOOLS = {
    "get_schedule": get_schedule,
    "find_parts_by_category": find_parts_by_category,
    "propose_change": propose_change,
    "validate_change": validate_change,
    "apply_change": apply_change,
    "revert_change": revert_change,
    "recalc_kpis": recalc_kpis,
    "get_audit_log": get_audit_log,
}
