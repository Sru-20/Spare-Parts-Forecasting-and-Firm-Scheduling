import {
  Activity,
  Archive,
  Bot,
  ChartNoAxesCombined,
  ClipboardList,
  Gauge,
  LayoutDashboard,
  PackageSearch,
  Settings,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";

export interface NavigationItem {
  label: string;
  path?: string;
  icon: LucideIcon;
  disabledReason?: string;
}

export interface NavigationSection {
  label: string;
  items: NavigationItem[];
}

export const navigation: NavigationSection[] = [
  {
    label: "Workspace",
    items: [
      { label: "Overview", path: "/", icon: LayoutDashboard },
      { label: "Forecasts", path: "/forecasts", icon: ChartNoAxesCombined },
      { label: "Firm schedule", path: "/schedule", icon: ClipboardList },
      { label: "Inventory & supply", path: "/inventory", icon: PackageSearch },
      { label: "KPI analytics", path: "/kpis", icon: Gauge },
    ],
  },
  {
    label: "Assistant",
    items: [{ label: "Ask AI", path: "/assistant", icon: Bot }],
  },
  {
    label: "Governance",
    items: [
      { label: "Audit trail", path: "/audit", icon: ShieldCheck },
      { label: "Run history", path: "/runs", icon: Activity },
    ],
  },
  {
    label: "Manage",
    items: [
      { label: "Settings", path: "/settings", icon: Settings },
      {
        label: "Model management",
        icon: Archive,
        disabledReason: "No retraining or model-promotion API is available.",
      },
    ],
  },
];

export const pageTitles: Record<string, { title: string; breadcrumb: string }> = {
  "/": { title: "Overview", breadcrumb: "Planning / Overview" },
  "/forecasts": { title: "Forecasts", breadcrumb: "Planning / Demand" },
  "/schedule": { title: "Firm schedule", breadcrumb: "Planning / Execution" },
  "/inventory": { title: "Inventory & supply", breadcrumb: "Planning / Inventory" },
  "/kpis": { title: "KPI analytics", breadcrumb: "Performance / KPIs" },
  "/assistant": { title: "Ask AI", breadcrumb: "Workspace / Assistant" },
  "/audit": { title: "Audit trail", breadcrumb: "Governance / Audit" },
  "/runs": { title: "Run history", breadcrumb: "Governance / Runs" },
  "/settings": { title: "Settings", breadcrumb: "Workspace / Settings" },
};
