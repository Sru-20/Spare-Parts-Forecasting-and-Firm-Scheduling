from rest_framework import serializers

from backend.assistant.models import AssistantMessage, Conversation


class AssistantMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = AssistantMessage
        fields = (
            "id",
            "direction",
            "content",
            "intent",
            "language",
            "metadata",
            "created_at",
        )
        read_only_fields = fields


class ConversationSerializer(serializers.ModelSerializer):
    messages = AssistantMessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ("conversation_id", "language", "created_at", "updated_at", "messages")
        read_only_fields = fields


class AssistantMessageCreateSerializer(serializers.Serializer):
    conversation_id = serializers.UUIDField(required=False)
    message = serializers.CharField(max_length=2000, trim_whitespace=True)
    run_id = serializers.UUIDField(required=False)
    idempotency_key = serializers.CharField(max_length=128, required=False)

    def validate_message(self, value):
        if not value:
            raise serializers.ValidationError("Message cannot be blank")
        return value
