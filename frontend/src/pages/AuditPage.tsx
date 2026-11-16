import { useMemo, useState } from "react";
import { Fingerprint, Search, ShieldCheck } from "lucide-react";
import type { AuditRecord, Pagination } from "../api/types";
import { formatDate } from "../api/format";
import { queryString, useApiData } from "../api/hooks";
import { useRuns } from "../app/RunContext";
import { PageHeader, Panel, StateCard } from "../design-system/ui";
import { PaginationBar } from "./shared";

export function AuditPage() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [action, setAction] = useState("");
  const [runId, setRunId] = useState("");
  const { runs } = useRuns();
  const path = `/audit/?${queryString({ page, page_size: 50 })}`;
  const { data, loading, error } = useApiData<Pagination<AuditRecord>>(path);
  const actions = useMemo(
    () => [...new Set((data?.results ?? []).map((record) => record.action))].sort(),
    [data],
  );
  const rows = useMemo(() => {
    const query = search.trim().toLowerCase();
    return (data?.results ?? []).filter((record) => {
      if (action && record.action !== action) return false;
      if (runId && record.run_id !== runId) return false;
      if (!query) return true;
      return [
        record.action,
        record.actor.username,
        record.proposal_id ?? "",
        record.run_id ?? "",
        record.record_hash,
      ].some((value) => value.toLowerCase().includes(query));
    });
  }, [action, data, runId, search]);

  return (
    <>
      <PageHeader
        eyebrow="Governance"
        title="Audit trail"
        description="Inspect append-only proposal, approval, application, and revert events, with their hash-chain references."
      />
      <div className="audit-integrity-banner">
        <span className="audit-integrity-icon"><ShieldCheck size={17} /></span>
        <div><strong>Append-only history</strong><span>Records are served from the backend audit ledger. Hash values are displayed as recorded; this page does not claim independent chain verification.</span></div>
      </div>
      <Panel title="Audit records" description="Search and filters below apply to the currently loaded page; use pagination to inspect other records." className="page-panel">
        <div className="filter-toolbar">
          <label className="search-field audit-search">
            <Search size={15} aria-hidden="true" />
            <span className="sr-only">Search audit page</span>
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search action, actor, run, or hash" />
          </label>
          <label className="filter-control">
            <span>Action</span>
            <select value={action} onChange={(event) => setAction(event.target.value)}>
              <option value="">All actions on page</option>
              {actions.map((item) => <option value={item} key={item}>{item.replaceAll("_", " ")}</option>)}
            </select>
          </label>
          <label className="filter-control">
            <span>Run</span>
            <select value={runId} onChange={(event) => setRunId(event.target.value)}>
              <option value="">All runs on page</option>
              {runs.map((run) => <option value={run.run_id} key={run.run_id}>{run.run_id.slice(0, 8)}</option>)}
            </select>
          </label>
        </div>
        {loading ? <div className="skeleton skeleton-table" /> : error ? (
          <StateCard kind="error" title="Audit history unavailable" description={error} />
        ) : !rows.length ? (
          <StateCard title="No audit records match" description={data?.results.length ? "Try clearing one or more page filters." : "No audit records are saved for this workspace."} icon={Fingerprint} />
        ) : (
          <>
            <div className="audit-record-list">
              {rows.map((record) => (
                <article className="audit-record" key={record.audit_id}>
                  <div className="audit-record-main">
                    <span className="audit-record-icon"><Fingerprint size={16} /></span>
                    <div className="audit-record-title">
                      <strong>{record.action.replaceAll("_", " ")}</strong>
                      <span>{record.actor.username} · {record.actor.role} · {formatDate(record.created_at)}</span>
                    </div>
                    <div className="audit-record-ids">
                      {record.run_id && <span>Run {record.run_id.slice(0, 8)}</span>}
                      {record.proposal_id && <span>Proposal {record.proposal_id.slice(0, 8)}</span>}
                    </div>
                  </div>
                  <div className="hash-chain-row">
                    <span><small>Previous hash</small><code>{record.previous_hash || "GENESIS"}</code></span>
                    <span className="hash-connector">→</span>
                    <span><small>Record hash</small><code>{record.record_hash}</code></span>
                  </div>
                  <details className="audit-details">
                    <summary>View recorded payload</summary>
                    <div className="audit-payload-grid">
                      <PayloadBlock title="Before values" value={record.before_values} />
                      <PayloadBlock title="After values" value={record.after_values} />
                      <PayloadBlock title="Validation" value={record.validation} />
                      <PayloadBlock title="Details" value={record.details} />
                    </div>
                  </details>
                </article>
              ))}
            </div>
            <PaginationBar page={page} pageSize={50} total={data?.count ?? 0} onChange={setPage} />
          </>
        )}
      </Panel>
    </>
  );
}

function PayloadBlock({ title, value }: { title: string; value: unknown }) {
  return (
    <div className="audit-payload-block">
      <strong>{title}</strong>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </div>
  );
}
