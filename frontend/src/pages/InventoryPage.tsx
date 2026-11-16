import { useMemo, useState } from "react";
import { AlertTriangle, ArrowUpRight, Boxes, PackageSearch } from "lucide-react";
import { Link } from "react-router-dom";
import type { Part, SavedRows, Warning } from "../api/types";
import { formatDate, formatNumber } from "../api/format";
import { queryString, useApiData } from "../api/hooks";
import { useRuns } from "../app/RunContext";
import { Button, InlineNotice, PageHeader, Panel, StateCard, StatusBadge } from "../design-system/ui";
import { PartPicker, PaginationBar, RunRequired } from "./shared";

function inventoryPath(
  runId: string | null,
  part: Part | null,
  criticality: string,
  week: string,
  page: number,
) {
  if (!runId) return null;
  return `/schedules/?${queryString({
    run_id: runId,
    page,
    page_size: 100,
    part_id: part?.part_id,
    criticality: criticality || undefined,
    week: week || undefined,
  })}`;
}

export function InventoryPage() {
  const { currentRun } = useRuns();
  const [part, setPart] = useState<Part | null>(null);
  const [criticality, setCriticality] = useState("");
  const [week, setWeek] = useState("");
  const [page, setPage] = useState(1);
  const path = inventoryPath(currentRun?.run_id ?? null, part, criticality, week, page);
  const { data, loading, error } = useApiData<SavedRows>(path);
  const warningsPath = currentRun
    ? `/warnings/?${queryString({ run_id: currentRun.run_id, page: 1, page_size: 100 })}`
    : null;
  const warningsRequest = useApiData<{ results: Warning[] }>(warningsPath);
  const rows = data?.results ?? [];
  const supplierCount = useMemo(
    () => new Set(rows.map((row) => row.supplier_id).filter(Boolean)).size,
    [rows],
  );
  const partCount = useMemo(
    () => new Set(rows.map((row) => row.part_id).filter(Boolean)).size,
    [rows],
  );

  if (!currentRun) {
    return (
      <>
        <PageHeader eyebrow="Supply visibility" title="Inventory & supply" description="Review order-level projected inventory, expected receipts, supplier exposure, and saved warnings." />
        <RunRequired />
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Supply visibility"
        title="Inventory & supply"
        description="Inspect saved inventory positions alongside procurement paths and supplier context."
        actions={<Link className="button button-secondary" to="/schedule">Open schedule <ArrowUpRight size={14} /></Link>}
      />
      <InlineNotice tone="warning" icon={AlertTriangle}>
        The D2 inventory snapshot dated 26 Sep 2026 is a post-window simulated comparison, not historical operational inventory evidence. Displayed positions are simulated inputs and outputs.
      </InlineNotice>
      <div className="inventory-context-strip">
        <div><span className="inventory-context-icon"><Boxes size={16} /></span><span><small>Saved order rows in view</small><strong>{data ? formatNumber(data.total) : loading ? "Loading" : "Not measurable"}</strong></span></div>
        <div><span className="inventory-context-icon"><PackageSearch size={16} /></span><span><small>Parts on this page</small><strong>{formatNumber(partCount)}</strong></span></div>
        <div><span className="inventory-context-icon"><ArrowUpRight size={16} /></span><span><small>Suppliers on this page</small><strong>{formatNumber(supplierCount)}</strong></span></div>
        <div className="inventory-context-footnote">Page-level distinct counts only · no physical stock feed connected</div>
      </div>

      <div className="inventory-page-grid">
        <Panel title="Projected inventory by saved order path" description="Repeated part/week position values can appear for multiple supplier orders; no summation is performed." className="page-panel">
          <div className="filter-toolbar">
            <PartPicker selected={part} onSelect={(value) => { setPart(value); setPage(1); }} />
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
            <div className="filter-spacer" />
            <Button variant="quiet" onClick={() => { setPart(null); setCriticality(""); setWeek(""); setPage(1); }}>Clear filters</Button>
          </div>
          {loading ? <div className="skeleton skeleton-table" /> : error ? (
            <StateCard kind="error" title="Inventory rows unavailable" description={error} />
          ) : !data || !rows.length ? (
            <StateCard title="No inventory rows match these filters" description="Choose another part or clear filters to inspect saved order paths." icon={Boxes} />
          ) : (
            <>
              <div className="table-scroll">
                <table className="data-table inventory-table">
                  <thead>
                    <tr>
                      <th>Part / week</th>
                      <th className="numeric-cell">Opening usable</th>
                      <th className="numeric-cell">Expected receipts</th>
                      <th className="numeric-cell">Projected inventory</th>
                      <th className="numeric-cell">Backlog</th>
                      <th>Supplier</th>
                      <th>Risk flag</th>
                      <th>Status</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, index) => (
                      <tr key={`${row.part_id}-${row.week}-${row.supplier_id}-${index}`}>
                        <td><strong className="table-primary">{row.part_id}</strong><span className="table-secondary">Week {row.week || "—"} · {row.criticality || "Not recorded"}</span></td>
                        <td className="numeric-cell">{formatNumber(row.opening_usable_stock)}</td>
                        <td className="numeric-cell">{formatNumber(row.expected_receipts)}</td>
                        <td className="numeric-cell"><strong>{formatNumber(row.projected_inventory)}</strong></td>
                        <td className="numeric-cell">{formatNumber(row.backlog_qty)}</td>
                        <td><strong>{row.supplier_id || "Not recorded"}</strong><span className="table-secondary">{row.supply_mode || "Mode unavailable"}</span></td>
                        <td>{row.supplier_risk_flag || "Not defined"}</td>
                        <td><StatusBadge status={row.status || "unresolved"} /></td>
                        <td><Link className="icon-link" to={`/assistant?part_id=${encodeURIComponent(row.part_id ?? "")}&run_id=${encodeURIComponent(currentRun.run_id)}`} title={`Ask AI about ${row.part_id ?? "part"}`} aria-label={`Ask AI about ${row.part_id ?? "part"}`}><ArrowUpRight size={15} /></Link></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <PaginationBar page={page} pageSize={data.page_size} total={data.total} onChange={setPage} />
            </>
          )}
        </Panel>

        <Panel title="Warnings & supply context" description="Only warning records saved for this run are shown." className="inventory-warnings">
          {warningsRequest.loading ? <div className="skeleton skeleton-block" /> : warningsRequest.error ? (
            <StateCard kind="error" title="Warnings unavailable" description={warningsRequest.error} />
          ) : !warningsRequest.data?.results.length ? (
            <StateCard title="No saved warnings" description="No warning records are registered for this run." icon={AlertTriangle} />
          ) : (
            <div className="warning-list">
              {warningsRequest.data.results.map((warning) => (
                <article className="warning-detail" key={warning.id}>
                  <div className="warning-detail-head">
                    <StatusBadge status={warning.severity} />
                    <span>{warning.week ? formatDate(warning.week) : "Run-level"}</span>
                  </div>
                  <strong>{warning.code.replaceAll("_", " ")}</strong>
                  <p>{warning.recommendation || "No recommendation was saved."}</p>
                  {warning.part_id && <span className="mono-chip">{warning.part_id}</span>}
                </article>
              ))}
            </div>
          )}
          <div className="panel-footnote">The Phase 6 report records supplier-risk flags as undefined where no approved threshold exists. Missing risk flags are not treated as low risk.</div>
        </Panel>
      </div>
      <div className="forecast-footnote">Source: run {currentRun.run_id}; result file version {data?.result_version ?? "—"}. Values and dates come from the versioned schedule artifact; the frontend does not recalculate stock positions or order quantities.</div>
    </>
  );
}
