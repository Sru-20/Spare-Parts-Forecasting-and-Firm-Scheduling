import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { PropsWithChildren } from "react";
import { apiRequest, clearTokens, readTokens, writeTokens } from "../../api/client";
import type { User, JwtResponse } from "../../api/types";

interface AuthContextValue {
  user: User | null;
  ready: boolean;
  sessionNotice: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: PropsWithChildren) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);

  const clearSession = useCallback((notice?: string) => {
    clearTokens();
    setUser(null);
    if (notice) setSessionNotice(notice);
  }, []);

  useEffect(() => {
    const onExpired = () => clearSession("Your session expired. Sign in to continue.");
    window.addEventListener("nova:session-expired", onExpired);
    return () => window.removeEventListener("nova:session-expired", onExpired);
  }, [clearSession]);

  useEffect(() => {
    if (!readTokens()) {
      setReady(true);
      return;
    }
    let active = true;
    apiRequest<User>("/auth/me/")
      .then((profile) => {
        if (active) setUser(profile);
      })
      .catch(() => {
        if (active) clearSession("Your saved session could not be restored.");
      })
      .finally(() => {
        if (active) setReady(true);
      });
    return () => {
      active = false;
    };
  }, [clearSession]);

  const login = useCallback(async (username: string, password: string) => {
    setSessionNotice(null);
    const result = await apiRequest<JwtResponse>("/auth/token/", {
      method: "POST",
      authenticate: false,
      body: JSON.stringify({ username, password }),
    });
    writeTokens({ access: result.access, refresh: result.refresh });
    setUser(result.user);
  }, []);

  const logout = useCallback(async () => {
    const tokens = readTokens();
    try {
      if (tokens) {
        await apiRequest<void>("/auth/logout/", {
          method: "POST",
          body: JSON.stringify({ refresh: tokens.refresh }),
          retryAfterRefresh: false,
        });
      }
    } finally {
      clearSession();
    }
  }, [clearSession]);

  const value = useMemo(
    () => ({ user, ready, sessionNotice, login, logout }),
    [user, ready, sessionNotice, login, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used within AuthProvider");
  return value;
}
