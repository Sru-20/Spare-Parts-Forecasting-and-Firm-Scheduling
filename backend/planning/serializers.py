from rest_framework import serializers
from django.conf import settings
from django.db import transaction
from backend.accounts.serializers import UserSerializer
from backend.planning.models import (
    AuditRecord,
    KPIResult,
    Part,
    Proposal,
    ProposalVersion,
    ResultReference,
    Run,
    Warning,
)
from backend.planning.services.results import file_sha256
from pathlib import Path
import re


class PartSerializer(serializers.ModelSerializer):
    class Meta:
        model = Part
        fields = (
            "part_id",
            "part_number",
            "part_name",
            "component",
            "category",
            "material",
            "vehicle_type",
            "criticality",
            "lifecycle_status",
        )


class RunSerializer(serializers.ModelSerializer):
    results = serializers.SerializerMethodField()

    class Meta:
        model = Run
        fields = (
            "run_id",
            "created_at",
            "data_version",
            "data_hash",
            "config_version",
            "config_hash",
            "model_version",
            "horizon_weeks",
            "scheduling_policy",
            "checker_version",
            "status",
            "duration_seconds",
            "version",
            "results",
        )
        read_only_fields = ("run_id", "created_at", "version", "results")

    def get_results(self, instance):
        return ResultReferenceSerializer(instance.results.all(), many=True).data

    def validate(self, attrs):
        if attrs.get("duration_seconds") is not None and attrs["duration_seconds"] < 0:
            raise serializers.ValidationError(
                {"duration_seconds": "Duration cannot be negative"}
            )
        if attrs.get("status") == Run.Status.COMPLETED and attrs.get("error"):
            raise serializers.ValidationError(
                {"error": "Completed runs cannot contain an error"}
            )
        return attrs


class RunCreateSerializer(serializers.ModelSerializer):
    result_references = serializers.ListField(
        child=serializers.DictField(), min_length=1, max_length=4, write_only=True
    )

    class Meta:
        model = Run
        fields = (
            "data_version",
            "data_hash",
            "config_version",
            "config_hash",
            "model_version",
            "horizon_weeks",
            "scheduling_policy",
            "checker_version",
            "status",
            "duration_seconds",
            "error",
            "output_location",
            "result_references",
        )

    def validate(self, attrs):
        if attrs.get("status") != Run.Status.COMPLETED:
            raise serializers.ValidationError(
                {"status": "Only completed runs with saved outputs can be registered"}
            )
        if not 1 <= int(attrs.get("horizon_weeks", 0)) <= 104:
            raise serializers.ValidationError({"horizon_weeks": "Must be 1..104"})
        for field in ("data_hash", "config_hash"):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", attrs.get(field, "")):
                raise serializers.ValidationError({field: "Expected a SHA-256 hex digest"})
        output = Path(attrs.get("output_location", ""))
        if output.is_absolute() or ".." in output.parts or not output.parts or output.parts[0] != "results":
            raise serializers.ValidationError(
                {"output_location": "Must be a relative path under results/"}
            )
        seen = set()
        kinds = set()
        for item in attrs["result_references"]:
            if set(item) != {"kind", "relative_path", "file_hash", "version"}:
                raise serializers.ValidationError(
                    {"result_references": "Each reference must specify kind, relative_path, file_hash, version"}
                )
            if item["kind"] not in ResultReference.Kind.values:
                raise serializers.ValidationError({"result_references": "Unknown result kind"})
            if not re.fullmatch(r"[0-9a-fA-F]{64}", item["file_hash"]) or int(item["version"]) < 1:
                raise serializers.ValidationError({"result_references": "Invalid hash or version"})
            key = (item["kind"], int(item["version"]))
            if key in seen:
                raise serializers.ValidationError({"result_references": "Duplicate kind/version"})
            seen.add(key)
            kinds.add(item["kind"])
            relative = Path(item["relative_path"])
            root = (settings.BASE_DIR / "results").resolve()
            candidate = (settings.BASE_DIR / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not candidate.is_relative_to(root):
                raise serializers.ValidationError({"result_references": "Unsafe result path"})
            if not candidate.is_file() or file_sha256(candidate) != item["file_hash"]:
                raise serializers.ValidationError({"result_references": "Result file is missing or hash mismatched"})
        if not {
            ResultReference.Kind.FORECAST,
            ResultReference.Kind.SCHEDULE,
        }.issubset(kinds):
            raise serializers.ValidationError(
                {"result_references": "Forecast and schedule references must belong to the same run"}
            )
        return attrs

    def create(self, validated_data):
        references = validated_data.pop("result_references")
        with transaction.atomic():
            run = Run.objects.create(**validated_data)
            for item in references:
                ResultReference.objects.create(run=run, **item)
        return run


class ResultReferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResultReference
        fields = ("kind", "relative_path", "file_hash", "version", "created_at")
        read_only_fields = fields


class WarningSerializer(serializers.ModelSerializer):
    part_id = serializers.CharField(source="part.part_id", allow_null=True)

    class Meta:
        model = Warning
        fields = (
            "id",
            "run_id",
            "part_id",
            "code",
            "severity",
            "week",
            "evidence",
            "recommendation",
            "created_at",
        )
        read_only_fields = fields


class KPIResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = KPIResult
        fields = (
            "id",
            "run_id",
            "name",
            "value",
            "unit",
            "basis",
            "denominator",
            "details",
            "created_at",
        )
        read_only_fields = fields


class ProposalCreateSerializer(serializers.Serializer):
    run_id = serializers.UUIDField()
    request_text = serializers.CharField(max_length=2000, trim_whitespace=True)
    percent_change = serializers.DecimalField(
        max_digits=6, decimal_places=2, min_value=-50, max_value=100
    )
    part_ids = serializers.ListField(
        child=serializers.CharField(max_length=32), min_length=1, max_length=160,
        required=False,
    )
    category = serializers.CharField(max_length=128, required=False, allow_blank=False)

    def validate(self, attrs):
        if bool(attrs.get("part_ids")) == bool(attrs.get("category")):
            raise serializers.ValidationError(
                "Provide exactly one of part_ids or category"
            )
        if attrs.get("part_ids") and len(set(attrs["part_ids"])) != len(attrs["part_ids"]):
            raise serializers.ValidationError({"part_ids": "Duplicate part IDs are not allowed"})
        return attrs


class ConfirmProposalSerializer(serializers.Serializer):
    proposal_version = serializers.IntegerField(min_value=1)
    run_version = serializers.IntegerField(min_value=1)


class ProposalSerializer(serializers.ModelSerializer):
    proposer = UserSerializer(read_only=True)
    approver = UserSerializer(read_only=True)
    source_run_id = serializers.UUIDField(read_only=True)
    applied_result = ResultReferenceSerializer(read_only=True)

    class Meta:
        model = Proposal
        fields = (
            "proposal_id",
            "version",
            "source_run_id",
            "reverts_proposal_id",
            "source_run_version",
            "source_schedule_hash",
            "request_text",
            "intent",
            "before_values",
            "after_values",
            "validation",
            "impact",
            "warnings",
            "status",
            "proposer",
            "approver",
            "expires_at",
            "created_at",
            "updated_at",
            "applied_result",
        )
        read_only_fields = fields


class ProposalVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProposalVersion
        fields = ("version", "status", "snapshot", "created_at")
        read_only_fields = fields


class AuditRecordSerializer(serializers.ModelSerializer):
    actor = UserSerializer(read_only=True)

    class Meta:
        model = AuditRecord
        fields = (
            "audit_id",
            "action",
            "actor",
            "proposal_id",
            "run_id",
            "before_values",
            "after_values",
            "validation",
            "details",
            "created_at",
            "previous_hash",
            "record_hash",
        )
        read_only_fields = fields


