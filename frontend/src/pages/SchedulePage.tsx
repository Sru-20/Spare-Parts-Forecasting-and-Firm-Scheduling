import { useMemo, useState } from "react";
import { AlertTriangle, ArrowUpRight, ClipboardList, FilterX, Info } from "lucide-react";
import { Link } from "react-router-dom";
import type { Part, SavedRow, SavedRows } from "../api/types";
import { formatDate, formatNumber, safeString } from "../api/format";
import { queryString, useApiData } from "../api/hooks";
import { useRuns } from "../app/RunContext";
import { Button, InlineNotice, PageHeader, Panel, StateCard, StatusBadge } from "../design-system/ui";
import { PartPicker, PaginationBar, RunRequired } from "./shared";

type ScheduleSortKey = "part_id" | "week" | "order_qty" | "release_date" | "due_date" | "status";

function listPath(runId: string | null, part: Part | null, status: string, criticality: string, week: string, page: number) {
  if (!runId) return null;
  return `/${"schedules"}/?${queryString({
    run_id: runId,
    page,
    page_size: 100,
    part_id: part?.part_id,
    status: status || undefined,
    criticality: criticality || undefined,
    week: week || undefined,
  })}`;
}

function compareRows(rows: SavedRow[], key: ScheduleSortKey, direction: "asc" | "desc") {
  return [...rows].sort((a, b) => {
    const first = a[key] ?? "";
    const second = b[key] ?? "";
    const delta = Number.isFinite(Number(first)) && Number.isFinite(Number(second))
      ? Number(first) - Number(second)
      : String(first).localeCompare(String(second), undefined, { numeric: true, sensitivity: "base" });
    return direction === "asc" ? delta : -delta;
  });
}

export function SchedulePage() {
  const { currentRun } = useRuns();
  const [part, setPart] = useState<Part | null>(null);
  const [status, setStatus] = useState("");
  const [criticality, setCriticality] = useState("");
  const [week, setWeek] = useState("");
  const [page, setPage] = useState(1);
  const [sortKey, setSortKey] = useState<ScheduleSortKey>("week");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  const { data, loading, error, reload } = useApiData<SavedRows>(
    listPath(currentRun?.run_id ?? null, part, status, criticality, week, page),
  );
  const firmPath = currentRun
    ? `/schedules/?${queryString({ run_id: currentRun.run_id, page: 1, page_size: 1, status: "firm" })}`
    : null;
  const unresolvedPath = currentRun
    ? `/schedules/?${queryString({ run_id: currentRun.run_id, page: 1, page_size: 1, status: "unresolved" })}`
    : null;
  const firmRequest = useApiData<SavedRows>(firmPath);
  const unresolvedRequest = useApiData<SavedRows>(unresolvedPath);
  const rows = useMemo(() => compareRows(data?.results ?? [], sortKey, sortDirection), [data, sortDirection, sortKey]);
  const reset = () => {
    setPart(null);
    setStatus("");
    setCriticality("");
    setWeek("");
    setPage(1);
  };
  const changeSort = (key: ScheduleSortKey) => {
    if (sortKey === key) setSortDirection((value) => value === "asc" ? "desc" : "asc");
    else {
      setSortKey(key);
      setSortDirection("asc");
    }
  };

  if (!currentRun) {
    return (
      <>
        <PageHeader eyebrow="Execution planning" title="Firm schedule" description="Inspect saved simulated orders, supplier constraints, frozen windows, and unresolved paths." />
        <RunRequired />
      </>
    );
  }

  const totalFirm = firmRequest.data?.total;
  const totalUnresolved = unresolvedRequest.data?.total;

  return (
    <>
      <PageHeader
        eyebrow="Execution planning"
        title="Firm schedule"
        description="Inspect the saved order paths and the checks attached to each one. The application does not issue purchase orders."
        actions={<Link className="button button-secondary" to="/assistant">Ask AI <ArrowUpRight size={14} /></Link>}
      />
      <InlineNotice tone="warning" icon={AlertTriangle}>
        Phase 5 remains PARTIAL. Only rows explicitly marked firm are classified as firm; unresolved rows remain excluded from executable orders. This simulated schedule is provisional.
      </InlineNotice>
      <div className="schedule-summary-strip">
        <div className="schedule-summary-card">
          <span className="summary-icon summary-firm"><ClipboardList size={16} /></span>
          <span><small>Firm order rows</small><strong>{totalFirm === undefined ? "Not measurable" : formatNumber(totalFirm)}</strong></span>
          <StatusBadge status="firm" />
        </div>
        <div className="schedule-summary-card">
          <span className="summary-icon summary-unresolved"><AlertTriangle size={16} /></span>
          <span><small>Unresolved rows</small><strong>{totalUnresolved === undefined ? "Not measurable" : formatNumber(totalUnresolved)}</strong></span>
          <StatusBadge status="unresolved" />
        </div>
        <div className="schedule-summary-note">
          <Info size={15} />
          <span>Counts come from exact saved status filters. They are not procurement release counts.</span>
        </div>
      </div>

      <Panel title="Order paths" description={`Selected run ${currentRun.run_id} · ${currentRun.scheduling_policy} · checker ${currentRun.checker_version}`} className="page-panel">
        <div className="filter-toolbar">
          <PartPicker selected={part} onSelect={(value) => { setPart(value); setPage(1); }} />
          <label className="filter-control">
            <span>Order status</span>
            <select value={status} onChange={(event) => { setStatus(event.target.value); setPage(1); }}>
              <option value="">All statuses</option>
              <option value="firm">Firm</option>
              <option value="unresolved">Unresolved</option>
            </select>
          </label>
          <label className="filter-control">
            <span>Criticality</span>
            <select value={criticality} onChange={(event) => { setCriticality(event.target.value); setPage(1); }}>
              <option value="">All criticality</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
              <option value="Low">Low</option>
            </select>
          </label>
          <label className="filter-control">
            <span>Week</span>
            <select value={week} onChange={(event) => { setWeek(event.target.value); setPage(1); }}>
              <option value="">All weeks</option>
              {Array.from({ length: currentRun.horizon_weeks }, (_, index) => String(index + 1)).map((item) => (
                <option value={item} key={item}>Week {item}</option>
              ))}
            </select>
          </label>
          <Button variant="quiet" icon={FilterX} onClick={reset}>Clear</Button>
          <Button variant="quiet" onClick={() => void reload()}>Refresh</Button>
        </div>
        {loading ? <div className="skeleton skeleton-table" /> : error ? (
          <StateCard kind="error" title="Schedule rows unavailable" description={error} />
        ) : !data || !data.results.length ? (
          <StateCard title="No schedule rows match these filters" description="Clear filters or choose a different part. The page will not replace unresolved quantities with inferred values." icon={ClipboardList} />
        ) : (
          <>
            <div className="table-scroll">
              <table className="data-table schedule-table">
                <thead>
                  <tr>
                    <th><SortButton label="Part" sortKey="part_id" active={sortKey === "part_id"} direction={sortDirection} onClick={changeSort} /></th>
                    <th><SortButton label="Week" sortKey="week" active={sortKey === "week"} direction={sortDirection} onClick={changeSort} /></th>
                    <th className="numeric-cell"><SortButton label="Order qty" sortKey="order_qty" active={sortKey === "order_qty"} direction={sortDirection} onClick={changeSort} /></th>
                    <th>Supplier / mode</th>
                    <th><SortButton label="Release" sortKey="release_date" active={sortKey === "release_date"} direction={sortDirection} onClick={changeSort} /></th>
                    <th><SortButton label="Due" sortKey="due_date" active={sortKey === "due_date"} direction={sortDirection} onClick={changeSort} /></th>
                    <th><SortButton label="Status" sortKey="status" active={sortKey === "status"} direction={sortDirection} onClick={changeSort} /></th>
                    <th>Independent checks</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => {
                    const failedChecks = listValueForChecks(row.failed_checks);
                    return (
                      <tr key={`${row.run_id}-${row.part_id}-${row.week}-${row.supplier_id}-${index}`}>
                        <td><strong className="table-primary">{row.part_id}</strong><span className="table-secondary">{row.part_name || row.criticality || "Catalogue detail not included"}</span></td>
                        <td><span className="week-pill">{safeString(row.week, "Not recorded")}</span></td>
                        <td className="numeric-cell"><strong>{formatNumber(row.order_qty)}</strong><span className="table-secondary">units</span></td>
                        <td><strong>{row.supplier_id || "Not recorded"}</strong><span className="table-secondary">{row.supply_mode || "Supply mode not recorded"}</span></td>
                        <td>{formatDate(row.release_date)}</td>
                        <td>{formatDate(row.due_date)}</td>
                        <td><StatusBadge status={row.status || "unresolved"} /></td>
                        <td>
                          {failedChecks.length
                            ? <span className="constraint-fail" title={failedChecks.join(", ")}>{failedChecks.join(", ")}</span>
                            : <span className={row.status === "firm" ? "constraint-pass" : "constraint-pending"}>
                                {row.status === "firm" ? "No recorded failures" : "Not cleared"}
                              </span>}
                        </td>
                        <td>
                          <Link
                            className="icon-link"
                            to={`/assistant?part_id=${encodeURIComponent(row.part_id ?? "")}&run_id=${encodeURIComponent(currentRun.run_id)}`}
                            aria-label={`Ask AI about ${row.part_id ?? "part"}`}
                            title="Ask AI about this order"
                          >
                            <ArrowUpRight size={15} />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <PaginationBar page={page} pageSize={data.page_size} total={data.total} onChange={setPage} />
          </>
        )}
      </Panel>
      <div className="forecast-footnote">Source: versioned schedule result v{data?.result_version ?? "—"}. All quantities, dates, status, and checker findings are displayed from the saved schedule; no order quantities are computed in the browser.</div>
    </>
  );
}

function SortButton({
  label,
  sortKey,
  active,
  direction,
  onClick,
}: {
  label: string;
  sortKey: ScheduleSortKey;
  active: boolean;
  direction: "asc" | "desc";
  onClick: (key: ScheduleSortKey) => void;
}) {
  return <button onClick={() => onClick(sortKey)}>{label}<span className={`sort-glyph ${active ? "sort-active" : ""}`}>{active ? direction === "asc" ? "↑" : "↓" : "↕"}</span></button>;
}

function listValueForChecks(value: unknown) {
  if (Array.isArray(value)) return value.map(String);
  if (typeof value !== "string" || !value.trim()) return [];
  return value.split(/[|;,]/).map((item) => item.trim()).filter(Boolean);
}
