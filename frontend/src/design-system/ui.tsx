import type { ComponentProps, PropsWithChildren, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { AlertCircle, ArrowDownRight, ArrowUpRight, Check, LoaderCircle } from "lucide-react";

type ButtonVariant = "primary" | "secondary" | "quiet" | "danger";

interface ButtonProps extends ComponentProps<"button"> {
  variant?: ButtonVariant;
  icon?: LucideIcon;
  loading?: boolean;
}

export function Button({
  variant = "secondary",
  icon: Icon,
  loading = false,
  children,
  className = "",
  disabled,
  ...props
}: ButtonProps) {
  const LeadingIcon = loading ? LoaderCircle : Icon;
  return (
    <button
      className={`button button-${variant} ${className}`.trim()}
      disabled={disabled || loading}
      {...props}
    >
      {LeadingIcon && (
        <LeadingIcon className={loading ? "spin" : ""} size={15} aria-hidden="true" />
      )}
      {children}
    </button>
  );
}

interface PageHeaderProps {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}

export function PageHeader({ eyebrow, title, description, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-heading-copy">
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1 className="page-heading">{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </header>
  );
}

interface PanelProps extends PropsWithChildren {
  title?: string;
  description?: string;
  action?: ReactNode;
  className?: string;
  contentClassName?: string;
}

export function Panel({
  title,
  description,
  action,
  className = "",
  contentClassName = "",
  children,
}: PanelProps) {
  return (
    <section className={`panel ${className}`.trim()}>
      {(title || description || action) && (
        <div className="panel-heading">
          <div className="panel-heading-copy">
            {title && <h2 className="panel-title">{title}</h2>}
            {description && <p className="panel-description">{description}</p>}
          </div>
          {action && <div className="panel-action">{action}</div>}
        </div>
      )}
      <div className={`panel-content ${contentClassName}`.trim()}>{children}</div>
    </section>
  );
}

type MetricTrend = "up" | "down" | "neutral";

interface MetricCardProps {
  label: string;
  value: string;
  helper?: string;
  source?: string;
  icon: LucideIcon;
  trend?: MetricTrend;
  tone?: "brand" | "teal" | "amber" | "rose";
}

export function MetricCard({
  label,
  value,
  helper,
  source,
  icon: Icon,
  trend,
  tone = "brand",
}: MetricCardProps) {
  const TrendIcon = trend === "up" ? ArrowUpRight : ArrowDownRight;
  return (
    <article className={`metric-card tone-${tone}`}>
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        <span className="metric-icon">
          <Icon size={16} aria-hidden="true" />
        </span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-footer">
        {helper && (
          <span className={`metric-helper ${trend ? `trend-${trend}` : ""}`}>
            {trend && <TrendIcon size={13} aria-hidden="true" />}
            {helper}
          </span>
        )}
        {source && <span className="metric-source">{source}</span>}
      </div>
    </article>
  );
}

const statusLabels: Record<string, string> = {
  completed: "Completed",
  running: "Running",
  queued: "Queued",
  failed: "Failed",
  firm: "Firm",
  unresolved: "Unresolved",
  provisional: "Provisional",
  validated: "Awaiting approval",
  applied: "Applied",
  rejected: "Rejected",
  stale: "Stale",
  canceled: "Canceled",
  expired: "Expired",
  high: "High",
  medium: "Medium",
  low: "Low",
  healthy: "Within threshold",
  alert: "Alert",
  not_measurable: "Not measurable",
};

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.trim().toLowerCase().replaceAll(" ", "_");
  const tone =
    ["completed", "firm", "applied", "low", "healthy"].includes(normalized)
      ? "success"
      : ["failed", "rejected", "high", "alert"].includes(normalized)
        ? "critical"
        : ["running", "validated", "provisional", "medium"].includes(normalized)
          ? "warning"
          : ["unresolved", "stale", "not_measurable"].includes(normalized)
            ? "muted"
            : "neutral";
  const label = statusLabels[normalized] ?? status.replaceAll("_", " ");
  return (
    <span className={`status-badge status-${tone}`}>
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

interface StateCardProps {
  title: string;
  description: string;
  icon?: LucideIcon;
  action?: ReactNode;
  kind?: "empty" | "error";
}

export function StateCard({
  title,
  description,
  icon: Icon = AlertCircle,
  action,
  kind = "empty",
}: StateCardProps) {
  return (
    <div className={`state-card state-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span className="state-icon">
        <Icon size={19} aria-hidden="true" />
      </span>
      <h2>{title}</h2>
      <p>{description}</p>
      {action}
    </div>
  );
}

export function LoadingState({ label = "Loading saved data" }: { label?: string }) {
  return (
    <div className="loading-grid" role="status" aria-label={label}>
      <span className="sr-only">{label}</span>
      <div className="skeleton skeleton-title" />
      <div className="skeleton skeleton-block" />
      <div className="skeleton skeleton-block" />
      <div className="skeleton skeleton-block" />
    </div>
  );
}

export function InlineNotice({
  children,
  tone = "info",
  icon: Icon = Check,
}: PropsWithChildren<{ tone?: "info" | "warning" | "critical"; icon?: LucideIcon }>) {
  return (
    <div className={`inline-notice notice-${tone}`} role="note">
      <Icon size={16} aria-hidden="true" />
      <span>{children}</span>
    </div>
  );
}
