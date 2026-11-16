from datetime import datetime, timezone as datetime_timezone
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView

from backend.accounts.models import RevokedAccessToken
from backend.accounts.serializers import LogoutSerializer, UserSerializer
from backend.accounts.throttles import LoginThrottle


class ProjectTokenSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token["role"] = user.role
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data["user"] = UserSerializer(self.user).data
        return data


class RoleAwareTokenView(TokenObtainPairView):
    serializer_class = ProjectTokenSerializer
    throttle_classes = [LoginThrottle]


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = LogoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            refresh = RefreshToken(serializer.validated_data["refresh"])
            if str(refresh["user_id"]) != str(request.user.pk):
                return Response(
                    {"detail": "Refresh token does not belong to this user"},
                    status=status.HTTP_403_FORBIDDEN,
                )
            refresh.blacklist()
            authorization = request.headers.get("Authorization", "")
            if authorization.startswith("Bearer "):
                from rest_framework_simplejwt.tokens import UntypedToken

                presented = UntypedToken(authorization.removeprefix("Bearer ").strip())
                token_id = presented.get("jti")
                if token_id:
                    RevokedAccessToken.objects.get_or_create(
                        token_id=token_id,
                        defaults={
                            "expires_at": datetime.fromtimestamp(
                                int(presented["exp"]), tz=datetime_timezone.utc
                            )
                        },
                    )
        except TokenError:
            return Response(
                {"detail": "Invalid or expired refresh token"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(status=status.HTTP_204_NO_CONTENT)
