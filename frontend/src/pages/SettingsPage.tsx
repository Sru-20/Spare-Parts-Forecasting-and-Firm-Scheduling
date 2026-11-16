import { useState } from "react";
import { KeyRound, Moon, Shield, Sun, UserPlus, Users } from "lucide-react";
import type { FormEvent } from "react";
import type { Pagination, User as ProjectUser, UserRole } from "../api/types";
import { apiRequest } from "../api/client";
import { errorDetail, formatDate } from "../api/format";
import { useApiData } from "../api/hooks";
import { useAuth } from "../app/auth/AuthContext";
import { useRuns } from "../app/RunContext";
import { useToast } from "../app/ToastContext";
import { useTheme } from "../design-system/ThemeProvider";
import { Button, InlineNotice, PageHeader, Panel, StateCard, StatusBadge } from "../design-system/ui";

const roles: UserRole[] = ["viewer", "planner", "approver", "admin"];

export function SettingsPage() {
  const { user } = useAuth();
  const { currentRun } = useRuns();
  const { theme, toggleTheme } = useTheme();
  const { notify } = useToast();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("viewer");
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const usersRequest = useApiData<Pagination<ProjectUser>>(
    user?.role === "admin" ? "/users/?page=1&page_size=100" : null,
  );

  const createUser = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);
    setSubmitting(true);
    try {
      await apiRequest<ProjectUser>("/users/", {
        method: "POST",
        body: JSON.stringify({ username: username.trim(), email: email.trim(), password, role }),
      });
      setUsername("");
      setEmail("");
      setPassword("");
      setRole("viewer");
      usersRequest.reload();
      notify("Account created with the selected project role.", "success");
    } catch (requestError) {
      setFormError(errorDetail(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Workspace"
        title="Settings"
        description="Review your account, local display preferences, and the access controls enforced by the API."
      />
      <InlineNotice tone="info" icon={Shield}>
        Role enforcement is server-side on every endpoint. UI visibility is not an authorization boundary.
      </InlineNotice>
      <div className="settings-grid">
        <Panel title="Account & access" description="Current authenticated identity from the backend.">
          {user ? (
            <div className="account-card">
              <span className="account-avatar">{user.username.slice(0, 2).toUpperCase()}</span>
              <div className="account-identity">
                <strong>{user.username}</strong>
                <span>{user.email || "No email address recorded"}</span>
                <StatusBadge status={user.role} />
              </div>
              <div className="account-permissions">
                <PermissionLine label="Read planning results" allowed />
                <PermissionLine label="Create a proposal" allowed={["planner", "approver", "admin"].includes(user.role)} />
                <PermissionLine label="Confirm a proposal" allowed={["approver", "admin"].includes(user.role)} />
                <PermissionLine label="Administer accounts" allowed={user.role === "admin"} />
              </div>
            </div>
          ) : (
            <StateCard title="Account unavailable" description="The authenticated profile could not be found." />
          )}
        </Panel>

        <Panel title="Display preferences" description="Saved locally in this browser; no backend configuration changes are made.">
          <div className="preference-row">
            <span className="preference-icon">{theme === "light" ? <Sun size={16} /> : <Moon size={16} />}</span>
            <div><strong>{theme === "light" ? "Light theme" : "Dark theme"}</strong><small>Current appearance</small></div>
            <Button variant="secondary" icon={theme === "light" ? Moon : Sun} onClick={toggleTheme}>Switch theme</Button>
          </div>
          <div className="preference-row">
            <span className="preference-icon"><KeyRound size={16} /></span>
            <div><strong>Session security</strong><small>Access and refresh tokens are held in session storage.</small></div>
          </div>
        </Panel>

        <Panel title="Active planning context" description="The selected run is shared across planning pages.">
          {currentRun ? (
            <div className="settings-run-context">
              <div><small>Run ID</small><strong>{currentRun.run_id}</strong></div>
              <div><small>Model identity</small><strong>{currentRun.model_version}</strong></div>
              <div><small>Scheduling policy</small><strong>{currentRun.scheduling_policy}</strong></div>
              <div><small>Horizon</small><strong>{currentRun.horizon_weeks} weeks</strong></div>
              <div><small>Registered</small><strong>{formatDate(currentRun.created_at)}</strong></div>
            </div>
          ) : <StateCard title="No selected run" description="Planning data remains unavailable until a registered run is selected." />}
          <div className="settings-limitation">Model retraining and promotion are unavailable because the backend provides no such endpoint. No disconnected controls are shown.</div>
        </Panel>

        {user?.role === "admin" && (
          <Panel title="User administration" description="Create an account with an explicit role. Existing role changes are not exposed by the API." className="admin-users-panel">
            <form className="admin-user-form" onSubmit={(event) => void createUser(event)}>
              <label><span>Username</span><input required maxLength={150} value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="off" /></label>
              <label><span>Email</span><input type="email" maxLength={254} value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="off" /></label>
              <label><span>Temporary password</span><input type="password" required maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" /></label>
              <label><span>Role</span><select value={role} onChange={(event) => setRole(event.target.value as UserRole)}>{roles.map((item) => <option value={item} key={item}>{item}</option>)}</select></label>
              {formError && <p className="form-error" role="alert">{formError}</p>}
              <Button variant="primary" icon={UserPlus} loading={submitting}>Create account</Button>
            </form>
            {usersRequest.loading ? <div className="skeleton skeleton-block" /> : usersRequest.error ? (
              <StateCard kind="error" title="Account list unavailable" description={usersRequest.error} />
            ) : !usersRequest.data?.results.length ? (
              <div className="mini-empty"><Users size={16} /><span>No accounts returned.</span></div>
            ) : (
              <div className="admin-user-list">
                {usersRequest.data.results.map((account) => (
                  <div className="admin-user-row" key={account.id}>
                    <span className="profile-avatar">{account.username.slice(0, 2).toUpperCase()}</span>
                    <span><strong>{account.username}</strong><small>{account.email || "No email"}</small></span>
                    <StatusBadge status={account.role} />
                    <StatusBadge status={account.is_active ? "completed" : "failed"} />
                  </div>
                ))}
              </div>
            )}
          </Panel>
        )}
      </div>
    </>
  );
}

function PermissionLine({ label, allowed }: { label: string; allowed: boolean }) {
  return <div className="permission-line"><span className={allowed ? "permission-allowed" : "permission-denied"}>{allowed ? "Allowed" : "Not allowed"}</span><span>{label}</span></div>;
}
