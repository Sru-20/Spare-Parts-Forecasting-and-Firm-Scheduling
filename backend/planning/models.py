from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class Part(models.Model):
    part_id = models.CharField(max_length=32, primary_key=True)
    part_number = models.CharField(max_length=64, blank=True)
    part_name = models.CharField(max_length=255)
    component = models.CharField(max_length=128, blank=True)
    category = models.CharField(max_length=128, blank=True, db_index=True)
    material = models.CharField(max_length=128, blank=True)
    vehicle_type = models.CharField(max_length=128, blank=True)
    criticality = models.CharField(max_length=32, blank=True)
    lifecycle_status = models.CharField(max_length=32, blank=True)
    source_hash = models.CharField(max_length=64)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("part_id",)
        indexes = [models.Index(fields=("category", "part_id"))]


class Run(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    run_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    data_version = models.CharField(max_length=128)
    data_hash = models.CharField(max_length=64)
    config_version = models.CharField(max_length=64)
    config_hash = models.CharField(max_length=64)
    model_version = models.CharField(max_length=128)
    horizon_weeks = models.PositiveSmallIntegerField()
    scheduling_policy = models.CharField(max_length=64)
    checker_version = models.CharField(max_length=64)
    status = models.CharField(max_length=16, choices=Status.choices, db_index=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    error = models.TextField(blank=True)
    output_location = models.CharField(max_length=512, blank=True)
    version = models.PositiveIntegerField(default=1)

    class Meta:
        ordering = ("-created_at",)


class ResultReference(models.Model):
    class Kind(models.TextChoices):
        FORECAST = "forecast", "Forecast"
        SCHEDULE = "schedule", "Schedule"
        KPI = "kpi", "KPI"
        WARNING = "warning", "Warning"

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="results")
    kind = models.CharField(max_length=16, choices=Kind.choices, db_index=True)
    relative_path = models.CharField(max_length=512)
    file_hash = models.CharField(max_length=64)
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("run", "kind", "version"), name="unique_run_result_version"
            )
        ]
        indexes = [models.Index(fields=("run", "kind"))]


class Warning(models.Model):
    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="warnings")
    part = models.ForeignKey(
        Part, null=True, blank=True, on_delete=models.SET_NULL, related_name="warnings"
    )
    code = models.CharField(max_length=64, db_index=True)
    severity = models.CharField(max_length=16, db_index=True)
    week = models.DateField(null=True, blank=True)
    evidence = models.JSONField(default=dict)
    recommendation = models.CharField(max_length=512, blank=True)
    created_at = models.DateTimeField(default=timezone.now)


class KPIResult(models.Model):
    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="kpis")
    name = models.CharField(max_length=80, db_index=True)
    value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=32)
    basis = models.CharField(max_length=32)
    denominator = models.FloatField(null=True, blank=True)
    details = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now)


class Proposal(models.Model):
    class Status(models.TextChoices):
        VALIDATED = "validated", "Validated"
        STALE = "stale", "Stale"
        CANCELED = "canceled", "Canceled"
        EXPIRED = "expired", "Expired"
        APPLIED = "applied", "Applied"
        REJECTED = "rejected", "Rejected"

    proposal_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.PositiveIntegerField(default=1)
    source_run = models.ForeignKey(
        Run, on_delete=models.PROTECT, related_name="proposals"
    )
    source_result = models.ForeignKey(
        ResultReference, on_delete=models.PROTECT, related_name="proposals"
    )
    source_schedule_hash = models.CharField(max_length=64)
    source_run_version = models.PositiveIntegerField()
    reverts_proposal = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="revert_actions",
    )
    request_text = models.CharField(max_length=2000)
    intent = models.JSONField(default=dict)
    before_values = models.JSONField(default=list)
    after_values = models.JSONField(default=list)
    validation = models.JSONField(default=dict)
    impact = models.JSONField(default=dict)
    warnings = models.JSONField(default=list)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.VALIDATED, db_index=True
    )
    proposer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="proposals_created",
    )
    approver = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="proposals_approved",
    )
    expires_at = models.DateTimeField(db_index=True)
    idempotency_key = models.CharField(max_length=128)
    applied_result = models.ForeignKey(
        ResultReference,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="applied_proposals",
    )
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("proposer", "idempotency_key"),
                name="unique_proposal_idempotency_per_user",
            )
        ]


class ProposalVersion(models.Model):
    proposal = models.ForeignKey(
        Proposal, on_delete=models.PROTECT, related_name="versions"
    )
    version = models.PositiveIntegerField()
    status = models.CharField(max_length=16)
    snapshot = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("proposal", "version"), name="unique_proposal_version"
            )
        ]

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise RuntimeError("Proposal versions are immutable")
        return super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise RuntimeError("Proposal versions are immutable")


class AuditChainHead(models.Model):
    head_id = models.CharField(max_length=16, primary_key=True, default="global")
    last_hash = models.CharField(max_length=64, blank=True)
    version = models.PositiveIntegerField(default=0)


class AuditRecord(models.Model):
    audit_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    action = models.CharField(max_length=32, db_index=True)
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="audit_records"
    )
    proposal = models.ForeignKey(
        Proposal, null=True, blank=True, on_delete=models.PROTECT, related_name="audit"
    )
    run = models.ForeignKey(Run, null=True, blank=True, on_delete=models.PROTECT)
    before_values = models.JSONField(default=list)
    after_values = models.JSONField(default=list)
    validation = models.JSONField(default=dict)
    details = models.JSONField(default=dict)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    previous_hash = models.CharField(max_length=64, blank=True)
    record_hash = models.CharField(max_length=64, unique=True)

    class Meta:
        ordering = ("created_at",)

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise RuntimeError("Audit records are append-only")
        return super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise RuntimeError("Audit records are append-only")
