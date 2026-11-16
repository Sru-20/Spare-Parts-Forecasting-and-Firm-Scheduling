from django.contrib import admin
from django.urls import include, path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from backend.accounts.views import LogoutView, MeView, RoleAwareTokenView


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/auth/token/", RoleAwareTokenView.as_view(), name="token_obtain_pair"),
    path("api/v1/auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/v1/auth/logout/", LogoutView.as_view(), name="token_logout"),
    path("api/v1/auth/me/", MeView.as_view(), name="auth_me"),
    path("api/v1/", include("backend.planning.urls")),
    path("api/v1/", include("backend.assistant.urls")),
]
