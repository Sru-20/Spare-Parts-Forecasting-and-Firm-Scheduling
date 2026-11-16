from django.urls import path

from backend.planning.views import (
    AuditListView,
    ForecastListView,
    KPIListView,
    PartListView,
    ProposalActionView,
    ProposalListCreateView,
    RunDetailView,
    RunListCreateView,
    ScheduleListView,
    UserDetailView,
    UserListCreateView,
    WarningListView,
)


urlpatterns = [
    path("parts/", PartListView.as_view(), name="parts"),
    path("forecasts/", ForecastListView.as_view(), name="forecasts"),
    path("schedules/", ScheduleListView.as_view(), name="schedules"),
    path("kpis/", KPIListView.as_view(), name="kpis"),
    path("warnings/", WarningListView.as_view(), name="warnings"),
    path("runs/", RunListCreateView.as_view(), name="runs"),
    path("runs/<uuid:run_id>/", RunDetailView.as_view(), name="run-detail"),
    path("users/", UserListCreateView.as_view(), name="users"),
    path("users/<int:pk>/", UserDetailView.as_view(), name="user-detail"),
    path("proposals/", ProposalListCreateView.as_view(), name="proposals"),
    path(
        "proposals/<uuid:proposal_id>/confirm/",
        ProposalActionView.as_view(action="confirm"),
        name="proposal-confirm",
    ),
    path(
        "proposals/<uuid:proposal_id>/cancel/",
        ProposalActionView.as_view(action="cancel"),
        name="proposal-cancel",
    ),
    path(
        "proposals/<uuid:proposal_id>/revert/",
        ProposalActionView.as_view(action="revert"),
        name="proposal-revert",
    ),
    path("audit/", AuditListView.as_view(), name="audit"),
]
