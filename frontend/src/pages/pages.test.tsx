import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { AssistantReply, AuditRecord, Conversation, KpiResult, Proposal, Run, SavedRows } from "../api/types";
import { AssistantPage } from "./AssistantPage";
import { AuditPage } from "./AuditPage";
import { ForecastsPage } from "./ForecastsPage";
import { KpisPage } from "./KpisPage";
import { SchedulePage } from "./SchedulePage";

const mocks = vi.hoisted(() => ({
  runContext: {
    currentRun: null as Run | null,
    runs: [] as Run[],
    loading: false,
    error: null as string | null,
    selectRun: vi.fn(),
    refreshRuns: vi.fn(),
  },
  user: { id: 2, username: "approver", email: "", role: "approver", is_active: true },
  notify: vi.fn(),
}));

vi.mock("../app/RunContext", () => ({ useRuns: () => mocks.runContext }));
vi.mock("../app/auth/AuthContext", () => ({ useAuth: () => ({ user: mocks.user }) }));
vi.mock("../app/ToastContext", () => ({ useToast: () => ({ notify: mocks.notify }) }));

const run: Run = {
  run_id: "8a9c1bbd-34ee-4bc0-9cb2-61374a8367f7",
  created_at: "2026-11-16T00:00:00Z",
  data_version: "test-fixture",
  data_hash: "a".repeat(64),
  config_version: "test-v1",
  config_hash: "b".repeat(64),
  model_version: "lightgbm_quantile::random_1",
  horizon_weeks: 14,
  scheduling_policy: "risk_adjusted_mrp",
  checker_version: "checker-fixture",
  status: "completed",
  duration_seconds: null,
  version: 1,
  results: [],
};

function response(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function savedRows(rows: Array<Record<string, string>>, total = rows.length): SavedRows {
  return {
    run_id: run.run_id,
    result_kind: "schedule",
    result_version: 1,
    page: 1,
    page_size: 100,
    total,
    results: rows,
  };
}

function paginated<T>(results: T[], count = results.length) {
  return { count, next: null, previous: null, results };
}

function renderPage(element: ReactNode) {
  return render(<MemoryRouter>{element}</MemoryRouter>);
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mocks.runContext.currentRun = run;
  mocks.runContext.runs = [run];
  mocks.user = { id: 2, username: "approver", email: "", role: "approver", is_active: true };
  mocks.notify.mockReset();
  fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
    response({ count: 0, next: null, previous: null, results: [] }),
  );
  vi.stubGlobal("fetch", fetchMock);
});

describe("saved planning pages", () => {
  it("renders stored forecasts, preserves missing quantiles, and paginates", async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://local");
      if (url.pathname.endsWith("/forecasts/")) {
        return response({
          ...savedRows([
            {
              part_id: "P0001",
              region: "East",
              week: "1",
              forecast_qty: "12.5",
              forecast_p90: "",
              candidate_id: "lightgbm_quantile::random_1",
              forecast_status: "PROVISIONAL",
            },
          ], 250),
          result_kind: "forecast",
        });
      }
      if (url.pathname.endsWith("/parts/")) return response(paginated([]));
      return response(paginated([]));
    });

    renderPage(<ForecastsPage />);
    expect(await screen.findByText("12.5")).toBeInTheDocument();
    expect(screen.getAllByText("Not measurable").length).toBeGreaterThan(0);
    expect(screen.getByText("lightgbm_quantile::random_1")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /next page/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("page=2"),
      expect.any(Object),
    ));
  });

  it("keeps firm and unresolved schedule rows visibly distinct", async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://local");
      if (url.pathname.endsWith("/schedules/")) {
        const status = url.searchParams.get("status");
        if (status === "firm") return response(savedRows([], 1));
        if (status === "unresolved") return response(savedRows([], 1));
        return response(savedRows([
          { run_id: run.run_id, part_id: "P0002", week: "1", order_qty: "30", status: "firm", failed_checks: "", supplier_id: "S001" },
          { run_id: run.run_id, part_id: "P0003", week: "2", order_qty: "0", status: "unresolved", failed_checks: "supplier_capacity", supplier_id: "S002" },
        ]));
      }
      return response(paginated([]));
    });

    renderPage(<SchedulePage />);
    const firmRow = await screen.findByText("P0002");
    const unresolvedRow = screen.getByText("P0003");
    expect(within(firmRow.closest("tr")!).getByText("Firm")).toBeInTheDocument();
    expect(within(unresolvedRow.closest("tr")!).getByText("Unresolved")).toBeInTheDocument();
    expect(screen.getByText("supplier_capacity")).toBeInTheDocument();
    expect(screen.getByText(/unresolved rows remain excluded from executable orders/i)).toBeInTheDocument();
  });

  it("renders only saved KPI values and labels missing measures", async () => {
    const result: KpiResult = {
      id: 1,
      run_id: run.run_id,
      name: "fill_rate",
      value: 0.875,
      unit: "ratio",
      basis: "simulated",
      denominator: 1200,
      details: { baseline_value: 0.81, formula: "fulfilled / demand" },
      created_at: "2026-11-16T00:00:00Z",
    };
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://local");
      if (url.pathname.endsWith("/kpis/")) return response(paginated([result]));
      return response(paginated([]));
    });

    renderPage(<KpisPage />);
    expect((await screen.findAllByText("87.5%")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Not measurable").length).toBeGreaterThan(0);
    expect(screen.getByText(/simulated data/i)).toBeInTheDocument();
    expect(screen.getByText("fulfilled / demand")).toBeInTheDocument();
  });

  it("raises read-only alerts from saved bias and tracking-signal records", async () => {
    const records: KpiResult[] = [
      {
        id: 2,
        run_id: run.run_id,
        name: "forecast_bias",
        value: -0.12,
        unit: "ratio",
        basis: "measured",
        denominator: 800,
        details: { formula: "Σ(forecast − actual) / Σ(actual)" },
        created_at: "2026-11-16T00:00:00Z",
      },
      {
        id: 3,
        run_id: run.run_id,
        name: "tracking_signal",
        value: 4.5,
        unit: "number",
        basis: "measured",
        denominator: 800,
        details: { formula: "Σ(forecast − actual) / MAD" },
        created_at: "2026-11-16T00:00:00Z",
      },
    ];
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://local");
      if (url.pathname.endsWith("/kpis/")) return response(paginated(records));
      return response(paginated([]));
    });

    renderPage(<KpisPage />);
    expect(await screen.findAllByText("Alert")).toHaveLength(2);
    expect(screen.getAllByText("-12.0%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("4.5").length).toBeGreaterThan(0);
    expect(screen.getByText(/provisional display threshold/i)).toBeInTheDocument();
    expect(screen.getByText(/browser does not recalculate/i)).toBeInTheDocument();
  });

  it("shows audit hashes without claiming chain verification", async () => {
    const record: AuditRecord = {
      audit_id: "record-1",
      action: "proposal_created",
      actor: { id: 1, username: "planner", role: "planner" },
      proposal_id: "proposal-1",
      run_id: run.run_id,
      before_values: [{ order_qty: 10 }],
      after_values: [{ order_qty: 12 }],
      validation: { feasible: true },
      details: {},
      created_at: "2026-11-16T00:00:00Z",
      previous_hash: "previous-hash",
      record_hash: "record-hash",
    };
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://local");
      if (url.pathname.endsWith("/audit/")) return response(paginated([record]));
      return response(paginated([]));
    });

    renderPage(<AuditPage />);
    expect(await screen.findByText("record-hash")).toBeInTheDocument();
    expect(screen.getByText(/does not claim independent chain verification/i)).toBeInTheDocument();
    await userEvent.click(screen.getByText("View recorded payload"));
    expect(screen.getByText(/"order_qty": 10/)).toBeInTheDocument();
  });
});

describe("controlled proposal interaction", () => {
  it("requires an explicit authorized confirmation action", async () => {
    const proposal: Proposal = {
      proposal_id: "a33b08be-64cb-4221-8c43-286e047f7012",
      version: 1,
      source_run_id: run.run_id,
      reverts_proposal_id: null,
      source_run_version: 1,
      source_schedule_hash: "c".repeat(64),
      request_text: "Increase P0001 by 10%",
      intent: { percent_change: "10.00" },
      before_values: [{ row_index: 0, part_id: "P0001", supplier_id: "S001", week: 1, order_qty: 10 }],
      after_values: [{ row_index: 0, part_id: "P0001", supplier_id: "S001", week: 1, order_qty: 12 }],
      validation: { feasible: true, checked_orders: 1, failed_checks: [] },
      impact: { order_qty_before: 10, order_qty_after: 12, order_qty_delta: 2, orders_changed: 1 },
      warnings: ["Simulated schedule only."],
      status: "validated",
      proposer: { id: 1, username: "planner", role: "planner" },
      approver: null,
      expires_at: "2026-11-16T01:00:00Z",
      created_at: "2026-11-16T00:00:00Z",
      updated_at: "2026-11-16T00:00:00Z",
      applied_result: null,
    };
    const reply: AssistantReply = {
      conversation_id: "conversation-1",
      message_id: 2,
      response: "A proposal is ready for explicit approval.",
      intent: "propose_change",
      language: "en",
      metadata: { proposal, auto_applied: false, explicit_confirmation_required: true },
    };
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://local");
      if (url.pathname.endsWith("/assistant/messages/") && init?.method === "POST") return response(reply);
      if (url.pathname.includes("/proposals/") && url.pathname.includes("/confirm/")) {
        return response({ ...proposal, status: "applied", version: 2 });
      }
      if (url.pathname.endsWith("/assistant/conversations/")) return response(paginated([]));
      if (url.pathname.includes("/assistant/conversations/")) {
        const conversation: Conversation = {
          conversation_id: "conversation-1",
          language: "en",
          created_at: "2026-11-16T00:00:00Z",
          updated_at: "2026-11-16T00:00:00Z",
          messages: [
            {
              id: 1,
              direction: "user",
              content: "Increase P0001 by 10%",
              intent: "propose_change",
              language: "en",
              metadata: {},
              created_at: "2026-11-16T00:00:00Z",
            },
            {
              id: 2,
              direction: "assistant",
              content: reply.response,
              intent: reply.intent,
              language: reply.language,
              metadata: reply.metadata,
              created_at: "2026-11-16T00:00:00Z",
            },
          ],
        };
        return response(conversation);
      }
      return response(paginated([]));
    });

    renderPage(<AssistantPage />);
    await userEvent.type(screen.getByRole("textbox", { name: /ask a question/i }), "Increase P0001 by 10%");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("A proposal is ready for explicit approval.")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/confirm/"),
      expect.objectContaining({ method: "POST" }),
    );
    const confirmButton = await screen.findByRole("button", { name: "Confirm & apply" });
    await userEvent.click(confirmButton);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/confirm/"),
      expect.objectContaining({ method: "POST" }),
    ));
  });
});
