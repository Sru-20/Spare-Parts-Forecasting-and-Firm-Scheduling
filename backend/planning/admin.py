from django.contrib import admin

from backend.planning.models import (
    AuditChainHead,
    AuditRecord,
    KPIResult,
    Part,
    Proposal,
    ProposalVersion,
    ResultReference,
    Run,
    Warning,
)


class ReadOnlyModelAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


for model in (
    Run,
    ResultReference,
    Warning,
    KPIResult,
    Proposal,
    ProposalVersion,
):
    admin.site.register(model, ReadOnlyModelAdmin)
admin.site.register(Part, ReadOnlyModelAdmin)
admin.site.register(AuditChainHead, ReadOnlyModelAdmin)


@admin.register(AuditRecord)
class AuditRecordAdmin(admin.ModelAdmin):
    readonly_fields = [field.name for field in AuditRecord._meta.fields]
    list_display = ("action", "actor", "created_at", "record_hash")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
