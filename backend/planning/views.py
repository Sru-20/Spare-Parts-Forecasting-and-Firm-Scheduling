from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.accounts.models import User
from backend.accounts.permissions import HasProjectRole
from backend.accounts.serializers import UserCreateSerializer, UserSerializer
from backend.accounts.throttles import ProposalThrottle
from backend.planning.models import (
    AuditRecord,
    KPIResult,
    Part,
    Proposal,
    ResultReference,
    Run,
    Warning,
)
from backend.planning.serializers import (
    AuditRecordSerializer,
    ConfirmProposalSerializer,
    KPIResultSerializer,
    PartSerializer,
    ProposalCreateSerializer,
    ProposalSerializer,
    RunCreateSerializer,
    RunSerializer,
    WarningSerializer,
)
from backend.planning.services.proposals import (
    cancel_proposal,
    confirm_proposal,
    create_proposal,
    revert_proposal,
)
from backend.planning.services.results import latest_result, paged_csv_rows


class AdminOnly(HasProjectRole):
    required_role = "admin"


class PartListView(generics.ListAPIView):
    serializer_class = PartSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset = Part.objects.all()
        category = self.request.query_params.get("category")
        query = self.request.query_params.get("q")
        if category:
            if len(category) > 128:
                raise ValidationError({"category": "Maximum length is 128"})
            queryset = queryset.filter(category__iexact=category)
        if query:
            if len(query) > 128:
                raise ValidationError({"q": "Maximum length is 128"})
            queryset = queryset.filter(part_name__icontains=query)
        return queryset


class RunListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    queryset = Run.objects.prefetch_related("results")

    def get_serializer_class(self):
        if self.request.method == "POST":
            return RunCreateSerializer
        return RunSerializer

    def get_permissions(self):
        if self.request.method == "POST":
            return [AdminOnly()]
        return super().get_permissions()

    @transaction.atomic
    def perform_create(self, serializer):
        serializer.save()


class RunDetailView(generics.RetrieveAPIView):
    queryset = Run.objects.prefetch_related("results")
    serializer_class = RunSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = "run_id"


class SavedRowsView(APIView):
    permission_classes = [IsAuthenticated]
    result_kind = None

    def get(self, request):
        run_id = request.query_params.get("run_id")
        if not run_id:
            raise ValidationError({"run_id": "This query parameter is required"})
        reference = latest_result(run_id, self.result_kind)
        try:
            page = int(request.query_params.get("page", 1))
            page_size = int(request.query_params.get("page_size", 50))
        except ValueError as error:
            raise ValidationError("page and page_size must be integers") from error
        filters = {
            key: request.query_params[key]
            for key in ("part_id", "region", "week", "status", "criticality", "category")
            if key in request.query_params
        }
        return Response(
            paged_csv_rows(
                reference, page=page, page_size=page_size, filters=filters
            )
        )


class ForecastListView(SavedRowsView):
    result_kind = ResultReference.Kind.FORECAST


class ScheduleListView(SavedRowsView):
    result_kind = ResultReference.Kind.SCHEDULE


class KPIListView(generics.ListAPIView):
    serializer_class = KPIResultSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        run_id = self.request.query_params.get("run_id")
        if not run_id:
            raise ValidationError({"run_id": "This query parameter is required"})
        return KPIResult.objects.filter(run_id=run_id).order_by("name", "pk")


class WarningListView(generics.ListAPIView):
    serializer_class = WarningSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        run_id = self.request.query_params.get("run_id")
        if not run_id:
            raise ValidationError({"run_id": "This query parameter is required"})
        return Warning.objects.filter(run_id=run_id).select_related("part").order_by(
            "-created_at"
        )


class ProposalListCreateView(generics.ListCreateAPIView):
    serializer_class = ProposalSerializer
    permission_classes = [IsAuthenticated]
    throttle_classes = [ProposalThrottle]

    def get_queryset(self):
        queryset = Proposal.objects.select_related(
            "proposer", "approver", "source_run", "applied_result"
        )
        if self.request.user.is_project_admin or self.request.user.is_approver:
            return queryset.order_by("-created_at")
        return queryset.filter(proposer=self.request.user).order_by("-created_at")

    def create(self, request, *args, **kwargs):
        serializer = ProposalCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        proposal, created = create_proposal(
            actor=request.user,
            payload=serializer.validated_data,
            idempotency_key=request.headers.get("Idempotency-Key", ""),
        )
        response = ProposalSerializer(proposal).data
        return Response(
            response,
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class ProposalActionView(APIView):
    permission_classes = [IsAuthenticated]
    action = None
    throttle_classes = [ProposalThrottle]

    def post(self, request, proposal_id):
        if self.action == "confirm":
            payload = ConfirmProposalSerializer(data=request.data)
            payload.is_valid(raise_exception=True)
            proposal = confirm_proposal(
                actor=request.user,
                proposal_id=proposal_id,
                proposal_version=payload.validated_data["proposal_version"],
                run_version=payload.validated_data["run_version"],
                idempotency_key=request.headers.get("Idempotency-Key", ""),
            )
        elif self.action == "cancel":
            if request.data:
                raise ValidationError("Cancel does not accept a request body")
            proposal = cancel_proposal(actor=request.user, proposal_id=proposal_id)
        elif self.action == "revert":
            if request.data:
                raise ValidationError("Revert does not accept a request body")
            proposal = revert_proposal(
                actor=request.user,
                proposal_id=proposal_id,
                idempotency_key=request.headers.get("Idempotency-Key", ""),
            )
        else:
            raise NotFound()
        return Response(ProposalSerializer(proposal).data)


class AuditListView(generics.ListAPIView):
    serializer_class = AuditRecordSerializer
    permission_classes = [IsAuthenticated]
    queryset = AuditRecord.objects.select_related("actor").order_by("-created_at")


class UserListCreateView(generics.ListCreateAPIView):
    permission_classes = [AdminOnly]
    queryset = User.objects.all().order_by("username")

    def get_serializer_class(self):
        if self.request.method == "POST":
            return UserCreateSerializer
        return UserSerializer


class UserDetailView(generics.RetrieveUpdateAPIView):
    permission_classes = [AdminOnly]
    queryset = User.objects.all()
    serializer_class = UserSerializer
    lookup_field = "pk"

    def get_serializer(self, *args, **kwargs):
        if self.request.method in {"PUT", "PATCH"}:
            raise NotFound("User role changes must be performed through admin tooling")
        return super().get_serializer(*args, **kwargs)
