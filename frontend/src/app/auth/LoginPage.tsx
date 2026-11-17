import { ArrowRight, Boxes, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Navigate } from "react-router-dom";
import { ApiError } from "../../api/client";
import { Button, InlineNotice } from "../../design-system/ui";
import { useAuth } from "./AuthContext";

export function LoginPage() {
  const { user, ready, login, sessionNotice } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (sessionNotice) setError(sessionNotice);
  }, [sessionNotice]);

  if (user) return <Navigate to="/" replace />;
  if (!ready) {
    return (
      <main className="auth-loading">
        <span className="auth-spinner" aria-label="Loading sign-in" />
      </main>
    );
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await login(username.trim(), password);
    } catch (requestError) {
      setError(
        requestError instanceof ApiError
          ? requestError.message
          : requestError instanceof Error
            ? requestError.message
            : "Sign-in failed. Please try again.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-story" aria-label="Application overview">
        <div className="login-brand">
          <img className="brand-mark" src="/sparecast-mark.svg" alt="" aria-hidden="true" />
          <span>SpareCast <span className="login-brand-divider">/</span> SPARE PARTS PLANNING</span>
        </div>
        <div className="login-story-copy">
          <span className="login-overline">SUPPLY CHAIN DECISION SUPPORT</span>
          <h1>Clarity across the parts that keep you moving.</h1>
          <p>
            Demand signals, constrained planning, and accountable decisions in
            one place.
          </p>
          <div className="login-feature-list">
            <div><Boxes size={16} /><span>Part-level demand and inventory context</span></div>
            <div><ShieldCheck size={16} /><span>Firm orders stay behind validation and approval</span></div>
          </div>
        </div>
        <div className="login-disclaimer">
          <span className="login-disclaimer-mark" />
          Method validated on simulated data
        </div>
      </section>
      <section className="login-panel">
        <form className="login-form" onSubmit={(event) => void submit(event)}>
          <div className="login-form-heading">
            <span className="eyebrow">WELCOME BACK</span>
            <h2>Sign in to your workspace</h2>
            <p>Use your assigned SpareCast account to continue.</p>
          </div>
          {error && (
            <div className="login-error" role="alert">
              <InlineNotice tone="critical">{error}</InlineNotice>
            </div>
          )}
          <label className="form-field">
            <span className="form-label">Username</span>
            <input
              className="text-field"
              type="text"
              name="username"
              autoComplete="username"
              required
              maxLength={150}
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label className="form-field">
            <span className="form-label">Password</span>
            <input
              className="text-field"
              type="password"
              name="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          <Button type="submit" variant="primary" icon={ArrowRight} loading={submitting} className="login-submit">
            Sign in
          </Button>
          <p className="login-security-note">
            Protected by role-based access and expiring session credentials.
          </p>
        </form>
        <footer className="login-footer">
          A simulated planning environment · Not connected to purchasing systems
        </footer>
      </section>
    </main>
  );
}
