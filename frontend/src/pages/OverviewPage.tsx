import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  ChartNoAxesCombined,
  Gauge,
  PackageCheck,
} from "lucide-react";
import { Link } from "react-router-dom";
import type { AuditRecord, KpiResult, SavedRows, Warning } from "../api/types";
import { formatDate, formatNumber, numberFrom } from "../api/format";
import { queryString, useApiData } from "../api/hooks";
import { useRuns } from "../app/RunContext";
import { LineTrend, chartColor } from "../components/LineTrend";
import { InlineNotice, MetricCard, PageHeader, Panel, StateCard, StatusBadge } from "../design-system/ui";
import { findKpi, formatKpi, kpiDefinitions } from "./kpi-data";

function useRunPath(endpoint: string, runId: string | null, filters: Record<string, string | number> = {}) {
  return runId
    ? `/${endpoint}/?${queryString({ run_id: runId, page: 1, page_size: 100, ...filters })}`
    : null;
}

function smallPagePath(endpoint: string, runId: string | null, filters: Record<string, string | number> = {}) {
  return runId
    ? `/${endpoint}/?${queryString({ run_id: runId, page: 1, page_size: 1, ...filters })}`
    : null;
}

export function OverviewPage() {
  const { currentRun, loading: runsLoading, error: runsError } = useRuns();
  const runId = currentRun?.run_id ?? null;
  const { data: kpiPage, loading: kpisLoading, error: kpisError } =
    useApiData<{ results: KpiResult[] }>(runId ? `/kpis/?${queryString({ run_id: runId, page: 1 })}` : null);
  const { data: warningPage, loading: warningsLoading, error: warningsError } =
    useApiData<{ results: Warning[] }>(
      runId ? `/warnings/?${queryString({ run_id: runId, page: 1, page_size: 5 })}` : null,
    );
  const { data: auditPage, loading: auditLoading, error: auditError } =
    useApiData<{ results: AuditRecord[] }>("/audit/?page=1&page_size=5");
  const { data: forecastPage, loading: forecastLoading, error: forecastError } =
    useApiData<SavedRows>(useRunPath("forecasts", runId));
  const { data: firmTotal } = useApiData<SavedRows>(smallPagePath("schedules", runId, { status: "firm" }));
  const { data: unresolvedTotal } = useApiData<SavedRows>(smallPagePath("schedules", runId, { status: "unresolved" }));
  const [selectedPartId, setSelectedPartId] = useState("");

  useEffect(() => {
    if (!selectedPartId && forecastPage?.results[0]?.part_id) {
      setSelectedPartId(forecastPage.results[0].part_id);
    }
  }, [selectedPartId, forecastPage]);

  const selectedForecastPath = runId && selectedPartId
    ? `/${"forecasts"}/?${queryString({ run_id: runId, part_id: selectedPartId, page: 1, page_size: 100 })}`
    : null;
  const {
    data: selectedForecast,
    loading: selectedForecastLoading,
    error: selectedForecastError,
  } = useApiData<SavedRows>(selectedForecastPath);

  const kpis = kpiPage?.results ?? [];
  const kpiCards = [
    { key: "forecast_accuracy", icon: ChartNoAxesCombined, tone: "brand" as const },
    { key: "fill_rate", icon: PackageCheck, tone: "teal" as const },
    { key: "stock_out", icon: AlertTriangle, tone: "amber" as const },
    { key: "critical_availability", icon: Gauge, tone: "rose" as const },
  ];
  const trend = useMemo(() => {
    const rows = selectedForecast?.results ?? [];
    const regions = [...new Set(rows.map((row) => row.region || "Unspecified"))];
    const periods = [...new Set(rows.map((row) => row.week_start_date || row.week || ""))].filter(Boolean);
    const series = regions.map((region, index) => ({
      key: `region_${index}`,
      label: region,
      color: chartColor(index),
    }));
    const data = periods.map((period) => {
      const point: Record<string, string | number | null> = {
        label: period.startsWith("W") || /^\d+$/.test(period) ? `Week ${period.replace(/^W/i, "")}` : formatDate(period),
      };
      series.forEach((item, index) => {
        const row = rows.find((candidate) =>
          (candidate.week_start_date || candidate.week || "") === period &&
          (candidate.region || "Unspecified") === regions[index],
        );
        point[item.key] = numberFrom(row?.forecast_qty);
      });
      return point;
    });
    return { data, series };
  }, [selectedForecast]);

  const isReady = !runsLoading;
  if (runsError) {
    return (
      <>
        <PageHeader eyebrow="NOVA WORKSPACE" title="Overview" description="A decision-ready view of saved forecast, schedule, inventory, and warning evidence." />
        <StateCard kind="error" title="Run history unavailable" description={runsError} />
      </>
    );
  }
  if (!isReady || runsLoading) {
    return (
      <>
        <PageHeader eyebrow="NOVA WORKSPACE" title="Overview" description="A decision-ready view of saved forecast, schedule, inventory, and warning evidence." />
        <div className="loading-grid"><div className="skeleton skeleton-block" /></div>
      </>
    );
  }
  if (!currentRun) {
    return (
      <>
        <PageHeader eyebrow="NOVA WORKSPACE" title="Overview" description="A decision-ready view of saved forecast, schedule, inventory, and warning evidence." />
        <StateCard title="No registered planning run" description="An administrator must register a versioned result set. No dashboard metrics are mocked or inferred." action={<Link className="button button-secondary" to="/runs">Open run history</Link>} />
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Planning overview"
        title="Good morning, planner"
        description="A traceable snapshot of saved forecast, schedule, inventory, and warning evidence."
        actions={<Link className="button button-secondary" to="/runs">Run details <ArrowUpRight size={14} /></Link>}
      />
      <InlineNotice tone="warning" icon={AlertTriangle}>
        Method validated on simulated data. Phase 5 remains PARTIAL; this future schedule is provisional and is not an operational commitment.
      </InlineNotice>
      <div className="overview-runline">
        <span><span className="runline-dot" /> Selected run <strong>{currentRun.run_id.slice(0, 8)}</strong></span>
        <span>Model <strong>{currentRun.model_version}</strong></span>
        <span>Policy <strong>{currentRun.scheduling_policy}</strong></span>
        <StatusBadge status={currentRun.status} />
      </div>

      <section className="metric-grid overview-metrics" aria-label="Run KPI summary">
        {kpisLoading
          ? <div className="skeleton skeleton-block" />
          : kpisError
            ? <div className="metric-error" role="alert">KPI evidence unavailable: {kpisError}</div>
            : kpiCards.map(({ key, icon, tone }) => {
                const definition = kpiDefinitions.find((item) => item.key === key);
                if (!definition) return null;
                const result = findKpi(kpis, definition);
                return (
                  <MetricCard
                    key={key}
                    label={definition.label}
                    value={formatKpi(result, definition.format)}
                    helper={result ? result.basis : "No saved KPI record"}
                    source={result ? result.unit : "Not measurable"}
                    icon={icon}
                    tone={tone}
                  />
                );
              })}
      </section>

      <div className="overview-grid">
        <Panel
          title="Saved demand signal"
          description="Forecast quantities as stored for the selected part. No actuals are available on this future horizon."
          action={
            <label className="compact-select-label">
              <span className="sr-only">Forecast part</span>
              <select
                value={selectedPartId}
                onChange={(event) => setSelectedPartId(event.target.value)}
                disabled={!forecastPage?.results.length}
              >
                {[...new Set((forecastPage?.results ?? []).map((row) => row.part_id))].map((partId) => (
                  <option value={partId} key={partId}>{partId}</option>
                ))}
              </select>
            </label>
          }
          className="overview-forecast"
        >
          {forecastLoading || selectedForecastLoading
            ? <div className="skeleton skeleton-chart" />
            : forecastError || selectedForecastError
              ? <StateCard kind="error" title="Forecast series unavailable" description={forecastError ?? selectedForecastError ?? "The forecast result could not be loaded."} />
              : !forecastPage?.results.length || !trend.data.length
                ? <StateCard title="No forecast rows saved" description="This run does not include forecast rows for display." icon={ChartNoAxesCombined} />
                : <LineTrend data={trend.data} series={trend.series} valueLabel="units" />}
          {forecastPage && (
            <div className="panel-footnote">
              {forecastPage.total.toLocaleString("en-IN")} forecast rows · {selectedPartId || "Part not selected"} · saved result v{forecastPage.result_version}
            </div>
          )}
        </Panel>

        <Panel title="Schedule status" description="Counts from saved order rows; firm and unresolved are kept distinct.">
          {!firmTotal && !unresolvedTotal
            ? <div className="mini-stat-list"><div className="mini-stat"><span>Firm rows</span><strong>Not measurable</strong></div><div className="mini-stat"><span>Unresolved rows</span><strong>Not measurable</strong></div></div>
            : (() => {
                const firm = firmTotal?.total ?? 0;
                const unresolved = unresolvedTotal?.total ?? 0;
                const total = firm + unresolved;
                return (
                  <>
                    <div className="schedule-status-visual" aria-label={`${firm} firm rows and ${unresolved} unresolved rows`}>
                      <span className="schedule-status-firm" style={{ width: `${total ? firm / total * 100 : 0}%` }} />
                      <span className="schedule-status-unresolved" style={{ width: `${total ? unresolved / total * 100 : 0}%` }} />
                    </div>
                    <div className="mini-stat-list">
                      <div className="mini-stat"><span><i className="legend-dot status-firm-dot" />Firm orders</span><strong>{formatNumber(firm)}</strong></div>
                      <div className="mini-stat"><span><i className="legend-dot status-unresolved-dot" />Unresolved orders</span><strong>{formatNumber(unresolved)}</strong></div>
                    </div>
                    <Link className="panel-inline-link" to="/schedule">Inspect order paths <ArrowUpRight size={13} /></Link>
                  </>
                );
              })()}
          <div className="panel-footnote">Counts reflect saved status rows only; they are not a service-level or feasibility KPI.</div>
        </Panel>

        <Panel title="Warnings requiring attention" description="Latest saved warning records across registered runs." action={<Link className="panel-text-link" to="/inventory">View inventory</Link>}>
          {warningsLoading
            ? <div className="skeleton skeleton-block" />
            : warningsError
              ? <div className="section-error" role="alert">{warningsError}</div>
              : !warningPage?.results.length
                ? <div className="mini-empty"><AlertTriangle size={17} /><span>No warning records are registered.</span></div>
                : <div className="warning-list">
                    {warningPage.results.slice(0, 4).map((warning) => (
                      <article className="warning-row" key={warning.id}>
                        <StatusBadge status={warning.severity} />
                        <div className="warning-copy">
                          <strong>{warning.code.replaceAll("_", " ")}</strong>
                          <span>{warning.part_id ? `Part ${warning.part_id}` : "Run-level warning"}{warning.week ? ` · ${warning.week}` : ""}</span>
                        </div>
                        <span className="warning-date">{formatDate(warning.created_at)}</span>
                      </article>
                    ))}
                  </div>}
        </Panel>

        <Panel title="Recent governance activity" description="Append-only audit records for this workspace." action={<Link className="panel-text-link" to="/audit">Open audit trail</Link>}>
          {auditLoading
            ? <div className="skeleton skeleton-block" />
            : auditError
              ? <div className="section-error" role="alert">{auditError}</div>
              : !auditPage?.results.length
                ? <div className="mini-empty"><Activity size={17} /><span>No audit records are registered.</span></div>
                : <div className="activity-list">
                    {auditPage.results.slice(0, 4).map((record) => (
                      <article className="activity-row" key={record.audit_id}>
                        <span className="activity-icon"><Activity size={14} /></span>
                        <div className="activity-copy">
                          <strong>{record.action.replaceAll("_", " ")}</strong>
                          <span>{record.actor.username} · {formatDate(record.created_at)}</span>
                        </div>
                        <code>{record.record_hash.slice(0, 9)}</code>
                      </article>
                    ))}
                  </div>}
        </Panel>
      </div>
    </>
  );
}
