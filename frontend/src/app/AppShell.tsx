import {
  Bell,
  ChevronDown,
  ChevronRight,
  Command,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Sun,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { apiRequest } from "../api/client";
import type { Pagination, Part, Warning } from "../api/types";
import { queryString, useApiData } from "../api/hooks";
import { useAuth } from "./auth/AuthContext";
import { navigation, pageTitles } from "./navigation";
import { useRuns } from "./RunContext";
import { useToast } from "./ToastContext";
import { useTheme } from "../design-system/ThemeProvider";

export function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { runs, currentRun, selectRun, error: runError } = useRuns();
  const warningPath = currentRun
    ? `/warnings/?${queryString({ run_id: currentRun.run_id, page: 1, page_size: 1 })}`
    : null;
  const warningPage = useApiData<Pagination<Warning>>(warningPath);
  const { theme, toggleTheme } = useTheme();
  const { notify } = useToast();
  const page = pageTitles[location.pathname] ?? {
    title: "Workspace",
    breadcrumb: "SpareCast / Planning",
  };
  const initials = user?.username.slice(0, 2) || "NA";

  useEffect(() => {
    setMobileOpen(false);
    setProfileOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen(true);
      }
      if (event.key === "Escape") setSearchOpen(false);
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, []);

  async function handleLogout() {
    try {
      await logout();
      notify("You have been signed out.", "success");
      navigate("/login", { replace: true });
    } catch (error) {
      notify(error instanceof Error ? error.message : "Sign-out could not be completed.", "error");
      navigate("/login", { replace: true });
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        collapsed={collapsed}
        mobileOpen={mobileOpen}
        onCollapse={() => setCollapsed((value) => !value)}
        onNavigate={() => setMobileOpen(false)}
      />
      <button
        type="button"
        className={`mobile-scrim ${mobileOpen ? "is-visible" : ""}`}
        aria-label="Close navigation"
        onClick={() => setMobileOpen(false)}
        tabIndex={mobileOpen ? 0 : -1}
      />
      <div className={`shell-main ${collapsed ? "is-collapsed" : ""}`}>
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="icon-button mobile-menu-button"
              aria-label="Open navigation"
              aria-expanded={mobileOpen}
              onClick={() => setMobileOpen(true)}
            >
              <Menu size={18} />
            </button>
            <div className="topbar-title-block">
              <div className="breadcrumb">{page.breadcrumb}</div>
              <h1 className="topbar-title">{page.title}</h1>
            </div>
          </div>
          <div className="topbar-actions">
            <label className="sr-only" htmlFor="active-run">Selected completed run</label>
            <select
              id="active-run"
              className="run-select"
              value={currentRun?.run_id ?? ""}
              onChange={(event) => selectRun(event.target.value)}
              aria-label="Select saved run"
            >
              {runs.length === 0 ? (
                <option value="">No completed runs</option>
              ) : (
                runs.map((run) => (
                  <option value={run.run_id} key={run.run_id}>
                    {run.model_version} · {run.run_id.slice(0, 8)}
                  </option>
                ))
              )}
            </select>
            <button
              type="button"
              className="icon-button search-trigger"
              aria-label="Search navigation and parts"
              onClick={() => setSearchOpen(true)}
            >
              <Search size={16} />
            </button>
            <button
              type="button"
              className="icon-button notification-trigger"
              aria-label="View warnings"
              title="View saved warnings"
              onClick={() => navigate("/inventory?view=warnings")}
            >
              <Bell size={16} />
              {typeof warningPage.data?.count === "number" && warningPage.data.count > 0 && (
                <span className="warning-count-badge" aria-label={`${warningPage.data.count} saved warnings`}>
                  {warningPage.data.count > 99 ? "99+" : warningPage.data.count}
                </span>
              )}
            </button>
            <button
              type="button"
              className="icon-button"
              aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
              onClick={toggleTheme}
            >
              {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
            </button>
            <div className="profile-menu">
              <button
                type="button"
                className="profile-trigger"
                aria-haspopup="menu"
                aria-expanded={profileOpen}
                onClick={() => setProfileOpen((value) => !value)}
              >
                <span className="profile-avatar">{initials}</span>
                <span className="profile-copy">
                  <span className="profile-name">{user?.username}</span>
                  <span className="profile-role">{user?.role}</span>
                </span>
                <ChevronDown className="profile-chevron" size={14} />
              </button>
              {profileOpen && (
                <div className="profile-popover" role="menu">
                  <div className="profile-popover-user">
                    <span className="profile-name">{user?.username}</span>
                    <span className="profile-role">{user?.role} access</span>
                  </div>
                  <button type="button" role="menuitem" onClick={() => navigate("/settings")}>
                    Account settings
                  </button>
                  <button type="button" role="menuitem" onClick={() => void handleLogout()}>
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
        {runError && (
          <div className="shell-message" role="status">
            Saved run metadata is unavailable: {runError}
          </div>
        )}
        <main className="page-content page-enter" key={location.pathname}>
          <Outlet />
        </main>
      </div>
      {searchOpen && <CommandSearch onClose={() => setSearchOpen(false)} />}
    </div>
  );
}

function Sidebar({
  collapsed,
  mobileOpen,
  onCollapse,
  onNavigate,
}: {
  collapsed: boolean;
  mobileOpen: boolean;
  onCollapse: () => void;
  onNavigate: () => void;
}) {
  return (
    <aside
      className={`shell-sidebar ${collapsed ? "is-collapsed" : ""} ${mobileOpen ? "is-mobile-open" : ""}`}
      aria-label="Main navigation"
    >
      <NavLink to="/" className="brand-lockup" aria-label="SpareCast planning overview" onClick={onNavigate}>
        <img className="brand-mark" src="/sparecast-mark.svg" alt="" aria-hidden="true" />
        <span className="brand-copy">
          <span className="brand-name">SpareCast</span>
          <span className="brand-caption">SPARE PARTS PLANNING</span>
        </span>
      </NavLink>
      <div className="workspace-switcher" title="Synthetic spare-parts planning workspace">
        <span className="workspace-dot" aria-hidden="true" />
        <span className="workspace-label">Spare parts · simulation</span>
      </div>
      <nav className="sidebar-nav">
        {navigation.map((section) => (
          <div className="nav-group" key={section.label}>
            <div className="nav-section">{section.label}</div>
            {section.items.map((item) => {
              const Icon = item.icon;
              if (!item.path) {
                return (
                  <div
                    className="nav-link nav-link-disabled"
                    aria-disabled="true"
                    title={item.disabledReason}
                    key={item.label}
                  >
                    <Icon className="nav-icon" size={17} aria-hidden="true" />
                    <span className="nav-link-label">{item.label}</span>
                  </div>
                );
              }
              return (
                <NavLink
                  to={item.path}
                  end={item.path === "/"}
                  className="nav-link"
                  aria-label={collapsed ? item.label : undefined}
                  title={collapsed ? item.label : undefined}
                  key={item.path}
                  onClick={onNavigate}
                >
                  <Icon className="nav-icon" size={17} aria-hidden="true" />
                  <span className="nav-link-label">{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <button
          type="button"
          className="sidebar-collapse"
          onClick={onCollapse}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          <span>Collapse menu</span>
        </button>
      </div>
    </aside>
  );
}

function CommandSearch({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [parts, setParts] = useState<Part[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const actions = useMemo(
    () => navigation.flatMap((section) => section.items.filter((item) => item.path)),
    [],
  );
  const matchedActions = actions.filter((item) =>
    item.label.toLowerCase().includes(query.trim().toLowerCase()),
  );

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setParts([]);
      setError(null);
      return;
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => {
      apiRequest<Pagination<Part>>(`/parts/?q=${encodeURIComponent(trimmed)}&page=1`, {
        signal: controller.signal,
      })
        .then((response) => {
          setParts(response.results);
          setError(null);
        })
        .catch((requestError: unknown) => {
          if (controller.signal.aborted) return;
          setError(requestError instanceof Error ? requestError.message : "Part search failed.");
        });
    }, 180);
    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [query]);

  return (
    <div className="command-backdrop" onMouseDown={onClose}>
      <section
        className="command-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="command-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <h2 id="command-title" className="sr-only">Search workspace</h2>
        <div className="command-search-row">
          <Search size={18} aria-hidden="true" />
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search pages or spare parts…"
            aria-label="Search pages or spare parts"
          />
          <kbd>ESC</kbd>
          <button type="button" className="icon-button command-close" onClick={onClose} aria-label="Close search">
            <X size={16} />
          </button>
        </div>
        <div className="command-results">
          {matchedActions.length > 0 && (
            <div className="command-group">
              <span className="command-group-label">Pages</span>
              {matchedActions.map((item) => (
                <button
                  type="button"
                  className="command-result"
                  key={item.path}
                  onClick={() => {
                    navigate(item.path!);
                    onClose();
                  }}
                >
                  <Command size={15} aria-hidden="true" />
                  <span>{item.label}</span>
                  <ChevronRight size={14} className="command-result-trailing" />
                </button>
              ))}
            </div>
          )}
          {parts.length > 0 && (
            <div className="command-group">
              <span className="command-group-label">Catalogue parts</span>
              {parts.map((part) => (
                <button
                  type="button"
                  className="command-result"
                  key={part.part_id}
                  onClick={() => {
                    navigate(`/inventory?part_id=${encodeURIComponent(part.part_id)}`);
                    onClose();
                  }}
                >
                  <span className="command-part-id">{part.part_id}</span>
                  <span>{part.part_name}</span>
                  <span className="command-result-trailing">{part.category || "Uncategorized"}</span>
                </button>
              ))}
            </div>
          )}
          {error && <p className="command-error" role="alert">{error}</p>}
          {query.trim().length >= 2 && !error && parts.length === 0 && matchedActions.length === 0 && (
            <p className="command-empty">No matching page or part in the current catalogue.</p>
          )}
          {query.trim().length < 2 && (
            <p className="command-hint">Type at least two characters to search the saved catalogue.</p>
          )}
        </div>
      </section>
    </div>
  );
}
