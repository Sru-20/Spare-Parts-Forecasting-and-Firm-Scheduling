from django.urls import path

from backend.assistant.views import (
    AssistantHistoryView,
    AssistantMessageView,
    ConversationDetailView,
    ConversationListView,
)


urlpatterns = [
    path("assistant/messages/", AssistantMessageView.as_view(), name="assistant-message"),
    path("assistant/history/", AssistantHistoryView.as_view(), name="assistant-history"),
    path("assistant/conversations/", ConversationListView.as_view(), name="conversations"),
    path(
        "assistant/conversations/<uuid:conversation_id>/",
        ConversationDetailView.as_view(),
        name="conversation-detail",
    ),
]
