import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { PropsWithChildren } from "react";
import { apiRequest } from "../api/client";
import type { Pagination, Run } from "../api/types";

interface RunContextValue {
  runs: Run[];
  currentRun: Run | null;
  loading: boolean;
  error: string | null;
  selectRun: (runId: string) => void;
  refreshRuns: () => Promise<void>;
}

const RunContext = createContext<RunContextValue | null>(null);
const RUN_KEY = "nova-current-run";

export function RunProvider({ children }: PropsWithChildren) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState(() => window.sessionStorage.getItem(RUN_KEY) ?? "");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await apiRequest<Pagination<Run>>("/runs/?page=1");
      setRuns(page.results);
      setRunId((current) => {
        const valid = page.results.some((run) => run.run_id === current);
        const nextId = valid
          ? current
          : page.results.find((run) => run.status === "completed")?.run_id ?? "";
        if (nextId) window.sessionStorage.setItem(RUN_KEY, nextId);
        else window.sessionStorage.removeItem(RUN_KEY);
        return nextId;
      });
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Run history could not be loaded.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  const selectRun = useCallback((nextId: string) => {
    setRunId(nextId);
    if (nextId) window.sessionStorage.setItem(RUN_KEY, nextId);
    else window.sessionStorage.removeItem(RUN_KEY);
  }, []);

  const value = useMemo(
    () => ({
      runs,
      currentRun: runs.find((run) => run.run_id === runId) ?? null,
      loading,
      error,
      selectRun,
      refreshRuns,
    }),
    [runs, runId, loading, error, selectRun, refreshRuns],
  );
  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRuns() {
  const value = useContext(RunContext);
  if (!value) throw new Error("useRuns must be used within RunProvider");
  return value;
}
