from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.authentication import JWTAuthentication

from backend.accounts.models import RevokedAccessToken


class RevocableJWTAuthentication(JWTAuthentication):
    def get_user(self, validated_token):
        token_id = validated_token.get("jti")
        if token_id and RevokedAccessToken.objects.filter(token_id=token_id).exists():
            raise AuthenticationFailed("Access token has been revoked")
        return super().get_user(validated_token)
