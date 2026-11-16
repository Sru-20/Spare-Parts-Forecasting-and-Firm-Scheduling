from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class User(AbstractUser):
    class Role(models.TextChoices):
        VIEWER = "viewer", "Viewer"
        PLANNER = "planner", "Planner"
        APPROVER = "approver", "Approver"
        ADMIN = "admin", "Admin"

    role = models.CharField(max_length=16, choices=Role.choices, default=Role.VIEWER)

    @property
    def is_planner(self) -> bool:
        return self.is_superuser or self.role in {self.Role.PLANNER, self.Role.ADMIN}

    @property
    def is_approver(self) -> bool:
        return self.is_superuser or self.role in {self.Role.APPROVER, self.Role.ADMIN}

    @property
    def is_project_admin(self) -> bool:
        return self.is_superuser or self.role == self.Role.ADMIN


class RevokedAccessToken(models.Model):
    token_id = models.CharField(max_length=128, unique=True)
    expires_at = models.DateTimeField(db_index=True)
    revoked_at = models.DateTimeField(default=timezone.now)

    class Meta:
        indexes = [models.Index(fields=("token_id", "expires_at"))]
