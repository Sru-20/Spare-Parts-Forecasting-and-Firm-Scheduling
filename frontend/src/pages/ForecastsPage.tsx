import { useMemo, useState } from "react";
import { ChartNoAxesCombined, Download, FilterX, Info } from "lucide-react";
import type { Part, SavedRow, SavedRows } from "../api/types";
import { formatNumber, numberFrom, safeString } from "../api/format";
import { queryString, useApiData } from "../api/hooks";
import { useRuns } from "../app/RunContext";
import { LineTrend, chartColor } from "../components/LineTrend";
import { Button, InlineNotice, PageHeader, Panel, StateCard, StatusBadge } from "../design-system/ui";
import { PartPicker, PaginationBar, RunRequired } from "./shared";

type SortKey = "part_id" | "region" | "week" | "forecast_qty" | "model";

function forecastPath(runId: string | null, part: Part | null, region: string, week: string, page: number) {
  if (!runId) return null;
  return `/${"forecasts"}/?${queryString({
    run_id: runId,
    page,
    page_size: 100,
    part_id: part?.part_id,
    region: region || undefined,
    week: week || undefined,
  })}`;
}

function sortRows(rows: SavedRow[], key: SortKey, direction: "asc" | "desc") {
  return [...rows].sort((a, b) => {
    const left = a[key] ?? "";
    const right = b[key] ?? "";
    const leftNumber = numberFrom(left);
    const rightNumber = numberFrom(right);
    const comparison =
      leftNumber !== null && rightNumber !== null
        ? leftNumber - rightNumber
        : String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
    return direction === "asc" ? comparison : -comparison;
  });
}

export function ForecastsPage() {
  const { currentRun } = useRuns();
  const [part, setPart] = useState<Part | null>(null);
  const [region, setRegion] = useState("");
  const [week, setWeek] = useState("");
  const [page, setPage] = useState(1);
  const [sortKey, setSortKey] = useState<SortKey>("part_id");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const rowsPath = forecastPath(currentRun?.run_id ?? null, part, region, week, page);
  const { data, loading, error, reload } = useApiData<SavedRows>(rowsPath);
  const chartPath = currentRun && part
    ? `/${"forecasts"}/?${queryString({
        run_id: currentRun.run_id,
        page: 1,
        page_size: 100,
        part_id: part.part_id,
        region: region || undefined,
      })}`
    : null;
  const chartRequest = useApiData<SavedRows>(chartPath);

  const regions = useMemo(
    () => [...new Set((data?.results ?? []).map((row) => row.region).filter((value): value is string => Boolean(value)))],
    [data],
  );
  const tableRows = useMemo(
    () => sortRows(data?.results ?? [], sortKey, sortDirection),
    [data, sortDirection, sortKey],
  );
  const trend = useMemo(() => {
    const rows = chartRequest.data?.results ?? [];
    const regionsInChart = [...new Set(rows.map((row) => row.region || "Unspecified"))];
    const periods = [...new Set(rows.map((row) => row.week_start_date || row.week || ""))].filter(Boolean);
    const series = regionsInChart.map((regionName, index) => ({
      key: `series_${index}`,
      label: regionName,
      color: chartColor(index),
    }));
    const chartData = periods.map((period) => {
      const point: Record<string, string | number | null> = {
        label: /^\d+$/.test(period) ? `Week ${period}` : safeString(period),
      };
      series.forEach((item, index) => {
        const row = rows.find((candidate) =>
          (candidate.week_start_date || candidate.week || "") === period &&
          (candidate.region || "Unspecified") === regionsInChart[index],
        );
        point[item.key] = numberFrom(row?.forecast_qty);
      });
      return point;
    });
    return { data: chartData, series };
  }, [chartRequest.data]);

  const changeSort = (key: SortKey) => {
    if (sortKey === key) setSortDirection((value) => value === "asc" ? "desc" : "asc");
    else {
      setSortKey(key);
      setSortDirection("asc");
    }
  };
  const resetFilters = () => {
    setPart(null);
    setRegion("");
    setWeek("");
    setPage(1);
  };

  return (
    <>
      <PageHeader
        eyebrow="Demand planning"
        title="Forecasts"
        description="Explore the saved demand path by part, region, and horizon. This page presents stored predictions; it does not calculate new forecasts."
        actions={<Button variant="secondary" icon={FilterX} onClick={resetFilters}>Clear filters</Button>}
      />
      {!currentRun ? <RunRequired /> : (
        <>
          <InlineNotice tone="warning" icon={Info}>
            Forecasts belong to the selected run and are provisional. The Track A point benchmark and Phase 6 end-to-end champion are separate identities; no model is promoted here.
          </InlineNotice>
          <Panel title="Forecast explorer" description="Filters are applied by the saved-results API. Sort changes apply to the current page only." className="page-panel">
            <div className="filter-toolbar">
              <PartPicker selected={part} onSelect={(value) => { setPart(value); setPage(1); }} />
              <label className="filter-control">
                <span>Region</span>
                <select value={region} onChange={(event) => { setRegion(event.target.value); setPage(1); }}>
                  <option value="">All regions</option>
                  {regions.map((item) => <option value={item} key={item}>{item}</option>)}
                </select>
              </label>
              <label className="filter-control">
                <span>Forecast week</span>
                <select value={week} onChange={(event) => { setWeek(event.target.value); setPage(1); }}>
                  <option value="">All 14 weeks</option>
                  {Array.from({ length: 14 }, (_, index) => String(index + 1)).map((item) => (
                    <option value={item} key={item}>Week {item}</option>
                  ))}
                </select>
              </label>
              <div className="filter-spacer" />
              <Button variant="quiet" icon={Download} onClick={() => void reload()}>Refresh</Button>
            </div>
            {part && (
              <div className="forecast-chart-wrap">
                <div className="subsection-heading">
                  <div>
                    <h3>Stored forecast signal</h3>
                    <p>{part.part_id} · {part.part_name} · forecast quantities by saved region</p>
                  </div>
                  <span className="unit-chip">Units / week</span>
                </div>
                {chartRequest.loading
                  ? <div className="skeleton skeleton-chart" />
                  : chartRequest.error
                    ? <StateCard kind="error" title="Forecast chart unavailable" description={chartRequest.error} />
                    : !trend.data.length
                      ? <StateCard title="No series for this selection" description="Choose a different part or clear the region filter." icon={ChartNoAxesCombined} />
                      : <LineTrend data={trend.data} series={trend.series} valueLabel="units" />}
              </div>
            )}
            {loading ? <div className="skeleton skeleton-table" /> : error ? (
              <StateCard kind="error" title="Forecast rows unavailable" description={error} />
            ) : !data || data.results.length === 0 ? (
              <StateCard title="No forecast rows match these filters" description="No values are inferred or substituted. Clear one or more filters to view saved rows." icon={ChartNoAxesCombined} />
            ) : (
              <>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th><button onClick={() => changeSort("part_id")}>Part <SortGlyph active={sortKey === "part_id"} direction={sortDirection} /></button></th>
                        <th><button onClick={() => changeSort("region")}>Region <SortGlyph active={sortKey === "region"} direction={sortDirection} /></button></th>
                        <th><button onClick={() => changeSort("week")}>Week <SortGlyph active={sortKey === "week"} direction={sortDirection} /></button></th>
                        <th className="numeric-cell"><button onClick={() => changeSort("forecast_qty")}>Forecast <SortGlyph active={sortKey === "forecast_qty"} direction={sortDirection} /></button></th>
                        <th className="numeric-cell">P90</th>
                        <th className="numeric-cell">P95</th>
                        <th className="numeric-cell">P99</th>
                        <th><button onClick={() => changeSort("model")}>Model <SortGlyph active={sortKey === "model"} direction={sortDirection} /></button></th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tableRows.map((row, index) => (
                        <tr key={`${row.part_id}-${row.region}-${row.week ?? row.week_start_date}-${index}`}>
                          <td><strong className="table-primary">{row.part_id}</strong><span className="table-secondary">{row.part_name || row.category || "Catalogue detail not included"}</span></td>
                          <td>{row.region || "Not recorded"}</td>
                          <td>{row.week_start_date ? safeString(row.week_start_date) : `Week ${safeString(row.week)}`}</td>
                          <td className="numeric-cell"><strong>{formatNumber(row.forecast_qty, 1)}</strong></td>
                          <td className="numeric-cell">{formatNumber(row.forecast_p90, 1)}</td>
                          <td className="numeric-cell">{formatNumber(row.forecast_p95, 1)}</td>
                          <td className="numeric-cell">{formatNumber(row.forecast_p99, 1)}</td>
                          <td><span className="mono-chip">{row.candidate_id || row.model || "Not recorded"}</span></td>
                          <td><StatusBadge status={row.forecast_status || "provisional"} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <PaginationBar page={page} pageSize={data.page_size} total={data.total} onChange={setPage} />
              </>
            )}
          </Panel>
          <div className="forecast-footnote">Source: run {currentRun.run_id}. Future-demand forecasts have no realized actuals here; accuracy, bias, and calibration are shown only when saved KPI records provide them.</div>
        </>
      )}
    </>
  );
}

function SortGlyph({ active, direction }: { active: boolean; direction: "asc" | "desc" }) {
  return <span className={`sort-glyph ${active ? "sort-active" : ""}`}>{active ? direction === "asc" ? "↑" : "↓" : "↕"}</span>;
}
