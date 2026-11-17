import os

from django.core.management.base import BaseCommand, CommandError

from backend.accounts.models import User


class Command(BaseCommand):
    help = "Create the Render demo administrator from environment variables."

    def handle(self, *args, **options):
        username = os.environ.get("SPARECAST_ADMIN_USERNAME", "").strip()
        email = os.environ.get("SPARECAST_ADMIN_EMAIL", "").strip()
        password = os.environ.get("SPARECAST_ADMIN_PASSWORD", "")
        if not username or not email or not password:
            raise CommandError(
                "SPARECAST_ADMIN_USERNAME, SPARECAST_ADMIN_EMAIL, and "
                "SPARECAST_ADMIN_PASSWORD must be configured."
            )

        user = User.objects.filter(username=username).first()
        if user is None:
            user = User.objects.create_superuser(
                username=username,
                email=email,
                password=password,
            )
        elif not user.is_superuser:
            raise CommandError(
                f"Existing user {username!r} is not a superuser; refusing to change it."
            )

        if user.role != User.Role.ADMIN:
            user.role = User.Role.ADMIN
            user.save(update_fields=("role",))

        self.stdout.write(self.style.SUCCESS("Render demo administrator is ready."))
