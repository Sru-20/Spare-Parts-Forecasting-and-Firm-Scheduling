from rest_framework.permissions import BasePermission


class HasProjectRole(BasePermission):
    required_role = None

    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        if user.is_superuser or user.role == "admin":
            return True
        role = self.required_role or getattr(view, "required_role", None)
        if role == "planner":
            return user.role in {"planner", "approver"}
        if role == "approver":
            return user.role == "approver"
        if role == "admin":
            return False
        return True
