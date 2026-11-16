import type { KpiResult } from "../api/types";
import { formatNumber, formatPercent, numberFrom } from "../api/format";

export type KpiFormat = "ratio" | "percent" | "count" | "currency" | "units" | "number";

export interface KpiDefinition {
  key: string;
  aliases: string[];
  label: string;
  format: KpiFormat;
  basis: string;
  formula: string;
  limitation: string;
}

export const kpiDefinitions: KpiDefinition[] = [
  {
    key: "forecast_wape",
    aliases: ["forecast_wape"],
    label: "Forecast WAPE",
    format: "ratio",
    basis: "Matched actual demand units",
    formula: "Σ|forecast − actual| / Σ(actual)",
    limitation: "A zero-total actual denominator is not measurable; lead-time error and service outcomes also matter.",
  },
  {
    key: "forecast_accuracy",
    aliases: ["forecast_accuracy_pct"],
    label: "Forecast accuracy",
    format: "percent",
    basis: "Weekly WAPE-derived accuracy",
    formula: "100 × (1 − WAPE)",
    limitation: "Accuracy is WAPE-derived and does not replace lead-time or service metrics.",
  },
  {
    key: "forecast_variance_signed",
    aliases: ["forecast_variance_signed_units"],
    label: "Forecast variance (signed)",
    format: "units",
    basis: "Matched part-origin-week pairs",
    formula: "Σ(forecast − actual)",
    limitation: "Positive values indicate aggregate over-forecasting; negative values indicate under-forecasting.",
  },
  {
    key: "forecast_variance_absolute",
    aliases: ["forecast_variance_absolute_units"],
    label: "Forecast variance (absolute)",
    format: "units",
    basis: "Matched part-origin-week pairs",
    formula: "Σ|forecast − actual|",
    limitation: "This total depends on the matched sample and is not a normalized rate.",
  },
  {
    key: "forecast_bias",
    aliases: ["forecast_bias"],
    label: "Forecast bias",
    format: "ratio",
    basis: "Signed forecast error over actual demand",
    formula: "Σ(forecast − actual) / Σ(actual)",
    limitation: "Negative values indicate aggregate under-forecasting.",
  },
  {
    key: "tracking_signal",
    aliases: ["tracking_signal"],
    label: "Tracking signal",
    format: "number",
    basis: "Cumulative forecast error / mean absolute deviation",
    formula: "Σ(forecast − actual) / MAD",
    limitation: "Alert threshold is a provisional display assumption, not a frozen acceptance criterion.",
  },
  {
    key: "fill_rate",
    aliases: ["fill_rate"],
    label: "Fill rate",
    format: "ratio",
    basis: "Simulated fulfilled demand units",
    formula: "Fulfilled current demand / actual demand",
    limitation: "Simulation result, not an operational service measurement.",
  },
  {
    key: "stock_out",
    aliases: ["stock_out_pct"],
    label: "Stock-out rate",
    format: "ratio",
    basis: "Eligible simulated part-weeks",
    formula: "Part-weeks with shortage / eligible part-weeks",
    limitation: "Simulation result; eligibility follows the saved report.",
  },
  {
    key: "critical_availability",
    aliases: ["critical_part_availability"],
    label: "Critical-part availability",
    format: "ratio",
    basis: "High-criticality simulated part-weeks",
    formula: "Available high-criticality part-weeks / high-criticality part-weeks",
    limitation: "Simulation result, not historical operational evidence.",
  },
  {
    key: "schedule_adherence",
    aliases: ["schedule_adherence"],
    label: "Schedule adherence",
    format: "ratio",
    basis: "Simulated planned order units",
    formula: "On-time received order units / planned order units",
    limitation: "The saved one-run histories do not measure replanning stability.",
  },
  {
    key: "emergency_orders",
    aliases: ["emergency_order_count"],
    label: "Emergency procurement",
    format: "count",
    basis: "Simulated emergency order events",
    formula: "Count of simulated emergency order events",
    limitation: "Simulation count; not an observed purchasing record.",
  },
  {
    key: "emergency_cost",
    aliases: ["emergency_cost_proxy"],
    label: "Emergency cost proxy",
    format: "currency",
    basis: "Simulated cost proxy",
    formula: "Saved emergency-cost proxy",
    limitation: "A modelled proxy, not realized financial savings or spend.",
  },
  {
    key: "excess_inventory",
    aliases: ["excess_inventory_units"],
    label: "Excess inventory",
    format: "units",
    basis: "Simulated inventory units",
    formula: "Saved excess-inventory units",
    limitation: "Simulation result; no physical inventory feed is connected.",
  },
  {
    key: "carrying_cost",
    aliases: ["carrying_cost_proxy"],
    label: "Carrying cost proxy",
    format: "currency",
    basis: "Simulated carrying-cost proxy",
    formula: "Saved carrying-cost proxy",
    limitation: "A modelled proxy, not an accounting ledger amount.",
  },
  {
    key: "turnover",
    aliases: ["turnover_14_week"],
    label: "Inventory turnover",
    format: "number",
    basis: "14-week simulation window",
    formula: "Simulated COGS / average inventory value",
    limitation: "Not annualized; do not compare directly with annual turnover targets.",
  },
  {
    key: "emergency_reduction",
    aliases: ["emergency_procurement_reduction"],
    label: "Emergency procurement reduction",
    format: "ratio",
    basis: "Compared with saved traditional baseline",
    formula: "Baseline emergency events less candidate events, relative to baseline",
    limitation: "Simulated comparison only; not a realized cost or savings claim.",
  },
  {
    key: "excess_reduction",
    aliases: ["excess_inventory_reduction"],
    label: "Excess inventory reduction",
    format: "ratio",
    basis: "Compared with saved traditional baseline",
    formula: "Baseline excess inventory less candidate excess, relative to baseline",
    limitation: "Simulated comparison only; not a realized inventory reduction.",
  },
  {
    key: "carrying_reduction",
    aliases: ["carrying_cost_reduction"],
    label: "Carrying cost reduction",
    format: "ratio",
    basis: "Compared with saved traditional baseline",
    formula: "Baseline cost proxy less candidate proxy, relative to baseline",
    limitation: "Simulated comparison only; not realized financial savings.",
  },
];

export const forecastBiasAlertLimit = 0.1;
export const trackingSignalAlertLimit = 4;

export function findKpi(results: KpiResult[] | null | undefined, definition: KpiDefinition) {
  return results?.find((result) => definition.aliases.includes(result.name)) ?? null;
}

export function formatKpi(result: KpiResult | null, format: KpiFormat) {
  if (!result || result.value === null || !Number.isFinite(result.value)) return "Not measurable";
  switch (format) {
    case "ratio":
      return formatPercent(result.value, false);
    case "percent":
      return formatPercent(result.value, true);
    case "count":
    case "units":
      return formatNumber(result.value);
    case "currency":
      return `${formatNumber(result.value)} currency units (proxy)`;
    case "number":
      return formatNumber(result.value, 2);
  }
}

export function baselineValue(result: KpiResult | null) {
  if (!result) return null;
  const value = result.details.baseline_value;
  return numberFrom(value);
}
