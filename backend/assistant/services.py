from __future__ import annotations

import time
from typing import Any

from django.db import transaction
from rest_framework.exceptions import PermissionDenied, ValidationError

from backend.assistant.intents import Intent, extract_intent, localized
from backend.assistant.models import AssistantMessage, Conversation
from backend.assistant.tools import (
    find_parts_by_category,
    get_forecast,
    get_inventory,
    get_kpis,
    get_schedule,
    get_warnings,
    propose_change,
)
from backend.planning.models import Run
from backend.planning.serializers import ProposalSerializer


def _read_only_response(intent: Intent, run_id: str | None) -> tuple[str, dict[str, Any]]:
    if intent.name in {"prompt_injection", "authorization_bypass"}:
        return localized(intent.language, "injection"), {"blocked": True}
    if intent.name == "constraint_violation":
        return localized(intent.language, "constraint"), {"blocked": True}
    if intent.name == "invalid_percentage":
        return localized(intent.language, "invalid"), {"blocked": True}
    if intent.name == "unknown_part":
        return localized(intent.language, "unknown"), {"part_id": intent.part_id}
    if intent.name == "cancel":
        return localized(intent.language, "cancel"), {"write_performed": False}
    if intent.name == "revert":
        return localized(intent.language, "revert"), {"write_performed": False}
    if intent.name == "find_category":
        parts = find_parts_by_category(intent.category or "")
        return (
            localized(
                intent.language,
                "find",
                count=len(parts),
                category=intent.category or "",
            ),
            {"tool": "find_parts_by_category", "results": parts},
        )
    if intent.name.startswith("read_"):
        if not run_id:
            return localized(intent.language, "missing_run"), {"clarification_required": True}
        run = Run.objects.filter(run_id=run_id, status=Run.Status.COMPLETED).first()
        if run is None:
            return localized(intent.language, "no_results"), {"clarification_required": True}
        if intent.name == "read_schedule":
            result = get_schedule(run_id, page_size=20)
            kind = "schedule"
        elif intent.name == "read_forecast":
            result = get_forecast(run_id, page_size=20)
            kind = "forecast"
        elif intent.name == "read_inventory":
            result = get_inventory(run_id, part_id=intent.part_id)
            kind = "simulated inventory"
        elif intent.name == "read_warning":
            result = get_warnings(run_id)
            kind = "warning"
        else:
            result = get_kpis(run_id)
            kind = "KPI"
        count = (
            result.get("total", len(result.get("rows", [])))
            if isinstance(result, dict)
            else len(result)
        )
        return (
            localized(intent.language, "saved", count=count, kind=kind),
            {"tool": intent.name, "results": result},
        )
    return localized(intent.language, "clarify"), {"clarification_required": True}


def _proposal_response(
    *,
    actor,
    intent: Intent,
    request_text: str,
    run_id: str | None,
    idempotency_key: str | None,
) -> tuple[str, dict[str, Any]]:
    if not actor.is_planner:
        raise PermissionDenied(localized(intent.language, "permission"))
    if intent.clarification_required or not intent.category and not intent.part_id:
        return localized(intent.language, "clarify"), {"clarification_required": True}
    if not run_id:
        return localized(intent.language, "missing_run"), {"clarification_required": True}
    if not idempotency_key:
        raise ValidationError({"Idempotency-Key": "Required for proposal requests"})
    payload: dict[str, Any] = {
        "run_id": run_id,
        "request_text": request_text,
        "percent_change": intent.percent_change,
    }
    if intent.category:
        payload["category"] = intent.category
    else:
        payload["part_ids"] = [intent.part_id]
    proposal = propose_change(actor, payload, idempotency_key=idempotency_key)
    return localized(intent.language, "proposal"), {
        "proposal": ProposalSerializer(proposal).data,
        "explicit_confirmation_required": True,
    }


@transaction.atomic
def handle_message(
    *,
    actor,
    message: str,
    conversation_id: str | None,
    run_id: str | None,
    idempotency_key: str | None,
) -> dict[str, Any]:
    if len(message) > 2000 or not message.strip():
        raise ValidationError({"message": "Message must be 1..2000 characters"})
    start = time.perf_counter()
    intent = extract_intent(message)
    if conversation_id:
        conversation = Conversation.objects.select_for_update().filter(
            conversation_id=conversation_id, user=actor
        ).first()
        if conversation is None:
            raise ValidationError({"conversation_id": "Conversation not found"})
        conversation.language = intent.language
        conversation.save(update_fields=("language", "updated_at"))
    else:
        conversation = Conversation.objects.create(user=actor, language=intent.language)
    AssistantMessage.objects.create(
        conversation=conversation,
        direction=AssistantMessage.Direction.USER,
        content=message,
        intent=intent.name,
        language=intent.language,
        metadata={"run_id": run_id},
    )
    if intent.name in {"propose_change", "electrical_example"}:
        answer, metadata = _proposal_response(
            actor=actor,
            intent=intent,
            request_text=message,
            run_id=run_id,
            idempotency_key=idempotency_key,
        )
    else:
        answer, metadata = _read_only_response(intent, run_id)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metadata.update(
        {
            "language": intent.language,
            "intent": intent.name,
            "latency_ms": round(elapsed_ms, 3),
            "external_model_calls": 0,
            "external_model_cost": 0,
            "auto_applied": False,
            "quantity_source": (
                "deterministic_scheduling_service"
                if intent.name in {"propose_change", "electrical_example"}
                else "none"
            ),
        }
    )
    response = AssistantMessage.objects.create(
        conversation=conversation,
        direction=AssistantMessage.Direction.ASSISTANT,
        content=answer,
        intent=intent.name,
        language=intent.language,
        metadata=metadata,
    )
    return {
        "conversation_id": str(conversation.conversation_id),
        "message_id": response.pk,
        "response": answer,
        "intent": intent.name,
        "language": intent.language,
        "metadata": metadata,
    }
