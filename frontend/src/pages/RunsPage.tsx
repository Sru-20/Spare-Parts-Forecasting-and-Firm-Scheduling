import { useState } from "react";
import { ArrowUpRight, Database, Fingerprint, Layers3 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import type { Pagination, Run } from "../api/types";
import { formatDate, formatNumber } from "../api/format";
import { queryString, useApiData } from "../api/hooks";
import { useRuns } from "../app/RunContext";
import { Button, InlineNotice, PageHeader, Panel, StateCard, StatusBadge } from "../design-system/ui";
import { PaginationBar } from "./shared";

export function RunsPage() {
  const [page, setPage] = useState(1);
  const { currentRun, selectRun, refreshRuns } = useRuns();
  const navigate = useNavigate();
  const path = `/runs/?${queryString({ page, page_size: 25 })}`;
  const { data, loading, error, reload } = useApiData<Pagination<Run>>(path);

  const chooseRun = async (run: Run) => {
    selectRun(run.run_id);
    await refreshRuns();
    navigate("/");
  };

  return (
    <>
      <PageHeader
        eyebrow="Governance"
        title="Run history"
        description="Review immutable run metadata and the versioned outputs registered with each planning run."
        actions={<Button variant="secondary" icon={Database} onClick={() => void reload()}>Refresh</Button>}
      />
      <InlineNotice tone="warning" icon={Fingerprint}>
        A saved run marked completed indicates registered outputs. It does not mean the protected Phase 7 final-test evaluation has been consumed or passed.
      </InlineNotice>
      <Panel title="Registered runs" description="Select a run to make it active across forecasts, schedule, inventory, and KPI views." className="page-panel">
        {loading ? <div className="skeleton skeleton-table" /> : error ? (
          <StateCard kind="error" title="Run history unavailable" description={error} />
        ) : !data?.results.length ? (
          <StateCard title="No run outputs registered" description="An administrator can register a completed run with verified result file paths and hashes. No placeholder run is shown." icon={Layers3} />
        ) : (
          <>
            <div className="run-card-list">
              {data.results.map((run) => (
                <article className={`run-card ${currentRun?.run_id === run.run_id ? "run-card-selected" : ""}`} key={run.run_id}>
                  <div className="run-card-main">
                    <span className="run-card-icon"><Layers3 size={17} /></span>
                    <div className="run-card-title">
                      <div><strong>{run.model_version}</strong><StatusBadge status={run.status} /></div>
                      <span>{formatDate(run.created_at)} · {run.run_id}</span>
                    </div>
                    <Button
                      variant={currentRun?.run_id === run.run_id ? "primary" : "secondary"}
                      icon={ArrowUpRight}
                      disabled={run.status !== "completed" || currentRun?.run_id === run.run_id}
                      onClick={() => void chooseRun(run)}
                    >
                      {currentRun?.run_id === run.run_id ? "Active run" : "Use this run"}
                    </Button>
                  </div>
                  <div className="run-card-meta">
                    <span><small>Horizon</small><strong>{formatNumber(run.horizon_weeks)} weeks</strong></span>
                    <span><small>Scheduling policy</small><strong>{run.scheduling_policy}</strong></span>
                    <span><small>Configuration</small><strong>{run.config_version}</strong></span>
                    <span><small>Checker</small><strong>{run.checker_version}</strong></span>
                    <span><small>Outputs</small><strong>{run.results.length} registered</strong></span>
                  </div>
                  <details className="run-card-details">
                    <summary>Inspect hashes and output references</summary>
                    <div className="run-detail-grid">
                      <HashValue label="Data version" value={run.data_version} hash={run.data_hash} />
                      <HashValue label="Config version" value={run.config_version} hash={run.config_hash} />
                      <div className="run-output-list">
                        <strong>Versioned output references</strong>
                        {run.results.map((reference) => (
                          <div key={`${reference.kind}-${reference.version}`} className="run-output-row">
                            <span className="mono-chip">{reference.kind} · v{reference.version}</span>
                            <span>{reference.relative_path}</span>
                            <code>{reference.file_hash}</code>
                          </div>
                        ))}
                        {!run.results.length && <span className="table-secondary">No outputs attached.</span>}
                      </div>
                    </div>
                  </details>
                </article>
              ))}
            </div>
            <PaginationBar page={page} pageSize={25} total={data.count} onChange={setPage} />
          </>
        )}
      </Panel>
    </>
  );
}

function HashValue({ label, value, hash }: { label: string; value: string; hash: string }) {
  return (
    <div className="run-hash-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <code>{hash}</code>
    </div>
  );
}
