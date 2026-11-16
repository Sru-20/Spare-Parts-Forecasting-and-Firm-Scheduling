from django.contrib import admin

from backend.assistant.models import AssistantMessage, Conversation

admin.site.register(Conversation)
admin.site.register(AssistantMessage)
