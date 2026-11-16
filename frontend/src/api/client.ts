import type { ApiErrorPayload } from "./types";

export interface TokenPair {
  access: string;
  refresh: string;
}

interface ApiRequestOptions extends RequestInit {
  authenticate?: boolean;
  retryAfterRefresh?: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload: ApiErrorPayload | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const TOKEN_KEY = "nova-session";
const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");
let refreshInFlight: Promise<TokenPair | null> | null = null;

export function readTokens(): TokenPair | null {
  const serialized = window.sessionStorage.getItem(TOKEN_KEY);
  if (!serialized) return null;
  try {
    const parsed: unknown = JSON.parse(serialized);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "access" in parsed &&
      "refresh" in parsed &&
      typeof parsed.access === "string" &&
      typeof parsed.refresh === "string"
    ) {
      return { access: parsed.access, refresh: parsed.refresh };
    }
  } catch {
    window.sessionStorage.removeItem(TOKEN_KEY);
  }
  return null;
}

export function writeTokens(tokens: TokenPair) {
  window.sessionStorage.setItem(TOKEN_KEY, JSON.stringify(tokens));
}

export function clearTokens() {
  window.sessionStorage.removeItem(TOKEN_KEY);
}

function isAuthPath(path: string) {
  return path.includes("/auth/token/");
}

async function refreshSession(refresh: string): Promise<TokenPair | null> {
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = (async () => {
    try {
      const response = await fetch(`${BASE_URL}/auth/token/refresh/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ refresh }),
      });
      if (!response.ok) return null;
      const body = (await response.json()) as Partial<TokenPair>;
      if (!body.access) return null;
      const tokens = { access: body.access, refresh: body.refresh || refresh };
      writeTokens(tokens);
      return tokens;
    } finally {
      refreshInFlight = null;
    }
  })();
  return refreshInFlight;
}

async function parseResponse(response: Response) {
  const content = await response.text();
  if (!content) return null;
  try {
    return JSON.parse(content) as unknown;
  } catch {
    return { detail: content };
  }
}

function errorMessage(payload: unknown, status: number) {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = payload.detail;
    if (typeof detail === "string") return detail;
  }
  return `Request failed (${status})`;
}

export async function apiRequest<T>(
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  const {
    authenticate = true,
    retryAfterRefresh = true,
    headers: initialHeaders,
    ...requestOptions
  } = options;
  const headers = new Headers(initialHeaders);
  if (requestOptions.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const tokens = readTokens();
  if (authenticate && tokens?.access) {
    headers.set("Authorization", `Bearer ${tokens.access}`);
  }
  const response = await fetch(`${BASE_URL}${path.startsWith("/") ? path : `/${path}`}`, {
    ...requestOptions,
    headers,
    credentials: "same-origin",
  });
  if (
    response.status === 401 &&
    authenticate &&
    retryAfterRefresh &&
    tokens?.refresh &&
    !isAuthPath(path)
  ) {
    const refreshed = await refreshSession(tokens.refresh);
    if (refreshed) {
      return apiRequest<T>(path, { ...options, retryAfterRefresh: false });
    }
    clearTokens();
    window.dispatchEvent(new CustomEvent("nova:session-expired"));
    throw new ApiError("Your session expired. Please sign in again.", 401, null);
  }
  const payload = await parseResponse(response);
  if (!response.ok) {
    const errorPayload =
      typeof payload === "object" && payload !== null
        ? (payload as ApiErrorPayload)
        : null;
    throw new ApiError(errorMessage(payload, response.status), response.status, errorPayload);
  }
  return payload as T;
}

export function apiUrl(path: string) {
  return `${BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}
