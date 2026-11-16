from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.accounts.throttles import AssistantThrottle
from backend.assistant.models import AssistantMessage, Conversation
from backend.assistant.serializers import (
    AssistantMessageCreateSerializer,
    AssistantMessageSerializer,
    ConversationSerializer,
)
from backend.assistant.services import handle_message


class AssistantMessageView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [AssistantThrottle]

    def post(self, request):
        payload = AssistantMessageCreateSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        values = payload.validated_data
        result = handle_message(
            actor=request.user,
            message=values["message"],
            conversation_id=values.get("conversation_id"),
            run_id=str(values["run_id"]) if values.get("run_id") else None,
            idempotency_key=values.get("idempotency_key")
            or request.headers.get("Idempotency-Key"),
        )
        return Response(result, status=200)


class ConversationListView(generics.ListAPIView):
    serializer_class = ConversationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).prefetch_related(
            "messages"
        ).order_by("-updated_at")


class ConversationDetailView(generics.RetrieveAPIView):
    serializer_class = ConversationSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = "conversation_id"

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).prefetch_related(
            "messages"
        )


class AssistantHistoryView(generics.ListAPIView):
    serializer_class = AssistantMessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        conversation_id = self.request.query_params.get("conversation_id")
        if not conversation_id:
            return AssistantMessage.objects.none()
        return AssistantMessage.objects.filter(
            conversation__conversation_id=conversation_id,
            conversation__user=self.request.user,
        ).order_by("created_at")
