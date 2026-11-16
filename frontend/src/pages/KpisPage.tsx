import { AlertCircle, ArrowDownUp, ChartNoAxesCombined, CircleDollarSign, Gauge, PackageCheck, ShieldAlert, TrendingDown, Warehouse } from "lucide-react";
import type { KpiResult } from "../api/types";
import { formatNumber, formatPercent } from "../api/format";
import { queryString, useApiData } from "../api/hooks";
import { useRuns } from "../app/RunContext";
import { InlineNotice, MetricCard, PageHeader, Panel, StateCard, StatusBadge } from "../design-system/ui";
import {
  baselineValue,
  findKpi,
  formatKpi,
  forecastBiasAlertLimit,
  kpiDefinitions,
  trackingSignalAlertLimit,
} from "./kpi-data";
import { RunRequired } from "./shared";

const icons = [
  ChartNoAxesCombined,
  ChartNoAxesCombined,
  ChartNoAxesCombined,
  ChartNoAxesCombined,
  ArrowDownUp,
  ArrowDownUp,
  PackageCheck,
  ShieldAlert,
  Gauge,
  PackageCheck,
  CircleDollarSign,
  Warehouse,
  TrendingDown,
  TrendingDown,
  CircleDollarSign,
  TrendingDown,
  TrendingDown,
];

function isSimulated(basis: string) {
  return basis.toLowerCase().includes("simulat") || basis.toLowerCase().includes("proxy");
}

function evidenceBasis(result: KpiResult) {
  const denominator = typeof result.details.denominator_basis === "string"
    ? ` · denominator: ${result.details.denominator_basis}`
    : result.denominator === null
      ? ""
      : ` · n=${formatNumber(result.denominator)}`;
  return `Basis: ${result.basis}${denominator}`;
}

export function KpisPage() {
  const { currentRun } = useRuns();
  const path = currentRun
    ? `/kpis/?${queryString({ run_id: currentRun.run_id, page: 1, page_size: 100 })}`
    : null;
  const { data, loading, error } = useApiData<{ results: KpiResult[]; count: number }>(path);
  const records = data?.results ?? [];
  const bias = records.find((record) => record.name === "forecast_bias") ?? null;
  const trackingSignal = records.find((record) => record.name === "tracking_signal") ?? null;
  const biasStatus = monitoringStatus(bias?.value ?? null, forecastBiasAlertLimit);
  const trackingStatus = monitoringStatus(trackingSignal?.value ?? null, trackingSignalAlertLimit);

  if (!currentRun) {
    return (
      <>
        <PageHeader eyebrow="Performance" title="KPI analytics" description="Review saved measures, their calculation basis, and any recorded baseline comparison." />
        <RunRequired />
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Performance"
        title="KPI analytics"
        description="Every value is linked to a saved KPI record, with its unit, denominator, and calculation basis."
      />
      <InlineNotice tone="warning" icon={AlertCircle}>
        Results are validated on simulated data only. The D2 inventory snapshot postdates the validation window; cost and inventory comparisons are scenario estimates, not historical operational evidence.
      </InlineNotice>
      <div className="kpi-context-strip">
        <span>Run <strong>{currentRun.run_id}</strong></span>
        <span>Candidate <strong>{currentRun.model_version}</strong></span>
        <span>Policy <strong>{currentRun.scheduling_policy}</strong></span>
        <span>Saved KPI records <strong>{loading ? "Loading" : error ? "Unavailable" : formatNumber(data?.count ?? records.length)}</strong></span>
      </div>
      {loading ? (
        <div className="metric-grid"><div className="skeleton skeleton-block" /><div className="skeleton skeleton-block" /><div className="skeleton skeleton-block" /><div className="skeleton skeleton-block" /></div>
      ) : error ? (
        <StateCard kind="error" title="KPI results unavailable" description={error} />
      ) : !records.length ? (
        <StateCard title="No KPI records saved for this run" description="The UI does not recompute or estimate KPI values. Register the corresponding versioned KPI evidence to make this page measurable." />
      ) : (
        <>
          <section className="metric-grid kpi-metric-grid" aria-label="Saved KPI values">
            {kpiDefinitions.map((definition, index) => {
              const result = findKpi(records, definition);
              const Icon = icons[index % icons.length] ?? Gauge;
              return (
                <MetricCard
                  key={definition.key}
                  label={definition.label}
                  value={formatKpi(result, definition.format)}
                  helper={result ? evidenceBasis(result) : "No saved record for this KPI"}
                  source={result ? `${result.unit} · ${result.basis}` : "Not measurable"}
                  icon={Icon}
                  tone={result && isSimulated(result.basis) ? "amber" : index % 3 === 0 ? "brand" : "teal"}
                />
              );
            })}
          </section>

          <div className="kpi-detail-grid">
            <Panel title="Metric definitions & evidence" description="Saved values are not recalculated by the browser. Where a value is absent, it remains not measurable." className="page-panel">
              <div className="table-scroll">
                <table className="data-table kpi-table">
                  <thead>
                    <tr><th>Measure</th><th>Saved value</th><th>Formula / basis</th><th>Comparison</th></tr>
                  </thead>
                  <tbody>
                    {kpiDefinitions.map((definition) => {
                      const result = findKpi(records, definition);
                      const baseline = baselineValue(result);
                      const formula = typeof result?.details.formula === "string"
                        ? result.details.formula
                        : definition.formula;
                      const baseText = baseline === null
                        ? "No saved baseline value"
                        : `${formatNumber(baseline, 2)} ${result?.unit ?? ""}`;
                      return (
                        <tr key={definition.key}>
                          <td><strong className="table-primary">{definition.label}</strong><span className="table-secondary">{definition.limitation}</span></td>
                          <td>
                            <strong>{formatKpi(result, definition.format)}</strong>
                            {result && <span className="table-secondary">{evidenceBasis(result)}</span>}
                          </td>
                          <td><span className="formula-cell">{formula}</span><span className="table-secondary">{result?.unit || definition.basis}</span></td>
                          <td>
                            {result ? (
                              <div className="comparison-cell">
                                <span>Traditional baseline</span>
                                <strong>{baseText}</strong>
                                {typeof result.details.percentage_difference === "number" && (
                                  <span className={result.details.worse_than_baseline === true ? "comparison-worse" : "comparison-better"}>
                                    {formatPercent(result.details.percentage_difference, true)} vs baseline
                                  </span>
                                )}
                              </div>
                            ) : "Not measurable"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>

            <div className="kpi-side-stack">
              <Panel title="Horizon coverage" description="The approved evaluation contract reports forecast horizons separately.">
                <div className="horizon-row">
                  <span><strong>Weeks 1–14</strong><small>Primary horizon</small></span>
                  <StatusBadge status={hasHorizonBand(records, "1_14") ? "completed" : "unresolved"} />
                </div>
                <div className="horizon-row">
                  <span><strong>Weeks 15–26</strong><small>Extended diagnostic</small></span>
                  <StatusBadge status={hasHorizonBand(records, "15_26") ? "completed" : "unresolved"} />
                </div>
                {!hasAnyHorizonBands(records) && (
                  <div className="panel-footnote">No separate horizon-band records are attached to this run. Aggregate values cannot be split without the underlying saved evidence.</div>
                )}
              </Panel>
              <Panel title="Segment coverage" description="Part, category, region, criticality, segment, and horizon breakdowns are required for an honest comparison.">
                {hasAnySegmentBreakdown(records) ? (
                  <StateCard title="Segment evidence is attached" description="The saved detail is available through the KPI record metadata." />
                ) : (
                  <div className="segment-unavailable">
                    <AlertCircle size={17} />
                    <p>Segment-level KPI records are not registered for this run. No aggregate is presented as a per-part or per-category result.</p>
                  </div>
                )}
              </Panel>
              <Panel
                title="Bias & tracking-signal alerts"
                description="Read-only status from saved KPI records; the browser does not recalculate either metric."
              >
                <div className="monitoring-list" role="group" aria-label="Forecast monitoring alerts">
                  <article className={`monitoring-row monitoring-${biasStatus}`}>
                    <div className="monitoring-copy">
                      <strong>Forecast bias</strong>
                      <small>Alert when absolute bias exceeds ±10%, from the configured forecast bias limit.</small>
                    </div>
                    <div className="monitoring-value">
                      <strong>{formatKpi(bias, "ratio")}</strong>
                      <StatusBadge status={biasStatus} />
                    </div>
                  </article>
                  <article className={`monitoring-row monitoring-${trackingStatus}`}>
                    <div className="monitoring-copy">
                      <strong>Tracking signal</strong>
                      <small>Alert at |signal| &gt; 4; provisional display threshold, not a frozen acceptance rule.</small>
                    </div>
                    <div className="monitoring-value">
                      <strong>{formatKpi(trackingSignal, "number")}</strong>
                      <StatusBadge status={trackingStatus} />
                    </div>
                  </article>
                </div>
                <div className="panel-footnote">Missing, null, or non-finite saved values remain not measurable and never appear as healthy.</div>
              </Panel>
              <Panel title="Additional forecast diagnostics" description="These metrics require dedicated saved evaluation outputs; they are not derived from aggregate dashboard values.">
                <div className="missing-metric-list">
                  {[
                    { label: "Lead-time WAPE", value: "Not measurable" },
                    { label: "MAE", value: "Not measurable" },
                    { label: "RMSE", value: "Not measurable" },
                    { label: "sMAPE", value: "Not measurable" },
                    { label: "MASE", value: "Not measurable" },
                    { label: "Tracking signal", value: formatKpi(trackingSignal, "number") },
                    { label: "Pinball loss", value: "Not measurable" },
                    { label: "P90 coverage and sample count", value: "Not measurable" },
                    { label: "P95 coverage and sample count", value: "Not measurable" },
                    { label: "P99 coverage and sample count", value: "Not measurable" },
                  ].map(({ label, value }) => (
                    <div className="missing-metric-row" key={label}>
                      <span>{label}</span>
                      <strong>{value}</strong>
                    </div>
                  ))}
                </div>
                <div className="panel-footnote">Missing saved diagnostics remain not measurable. Zero-demand and insufficient-sample cases are not inferred from aggregate dashboard values.</div>
              </Panel>
              <Panel title="Interpretation guardrails">
                <ul className="guardrail-list">
                  <li>Forecast accuracy is WAPE-derived and does not establish service quality.</li>
                  <li>Lead-time error and intermittent-series service outcomes take precedence when they disagree with weekly WAPE.</li>
                  <li>Cost and inventory comparisons are simulated proxies, not realized savings.</li>
                  <li>Zero-total actual demand and insufficient samples remain not measurable.</li>
                </ul>
              </Panel>
            </div>
          </div>
        </>
      )}
    </>
  );
}

function hasHorizonBand(records: KpiResult[], band: "1_14" | "15_26") {
  return records.some((record) => {
    const bands = record.details.horizon_bands;
    return typeof bands === "object" && bands !== null && band in bands;
  });
}

function hasAnyHorizonBands(records: KpiResult[]) {
  return records.some((record) => typeof record.details.horizon_bands === "object" && record.details.horizon_bands !== null);
}

function hasAnySegmentBreakdown(records: KpiResult[]) {
  return records.some((record) => Array.isArray(record.details.segments) && record.details.segments.length > 0);
}

function monitoringStatus(value: number | null, limit: number) {
  if (value === null || !Number.isFinite(value)) return "not_measurable";
  return Math.abs(value) > limit ? "alert" : "healthy";
}
