from rest_framework.throttling import AnonRateThrottle, UserRateThrottle


class LoginThrottle(AnonRateThrottle):
    scope = "login"


class AssistantThrottle(UserRateThrottle):
    scope = "assistant"


class ProposalThrottle(UserRateThrottle):
    scope = "proposal"
