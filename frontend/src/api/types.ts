export type UserRole = "viewer" | "planner" | "approver" | "admin";

export interface User {
  id: number;
  username: string;
  email: string;
  role: UserRole;
  is_active: boolean;
}

export interface Pagination<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}

export interface ResultReference {
  kind: "forecast" | "schedule" | "kpi" | "warning";
  relative_path: string;
  file_hash: string;
  version: number;
  created_at: string;
}

export interface Run {
  run_id: string;
  created_at: string;
  data_version: string;
  data_hash: string;
  config_version: string;
  config_hash: string;
  model_version: string;
  horizon_weeks: number;
  scheduling_policy: string;
  checker_version: string;
  status: "queued" | "running" | "completed" | "failed";
  duration_seconds: number | null;
  version: number;
  results: ResultReference[];
}

export interface Part {
  part_id: string;
  part_number: string;
  part_name: string;
  component: string;
  category: string;
  material: string;
  vehicle_type: string;
  criticality: string;
  lifecycle_status: string;
}

export type SavedRow = Record<string, string>;

export interface SavedRows {
  run_id: string;
  result_kind: "forecast" | "schedule";
  result_version: number;
  page: number;
  page_size: number;
  total: number;
  results: SavedRow[];
}

export interface KpiResult {
  id: number;
  run_id: string;
  name: string;
  value: number | null;
  unit: string;
  basis: string;
  denominator: number | null;
  details: Record<string, unknown>;
  created_at: string;
}

export interface Warning {
  id: number;
  run_id: string;
  part_id: string | null;
  code: string;
  severity: string;
  week: string | null;
  evidence: Record<string, unknown>;
  recommendation: string;
  created_at: string;
}

export interface ProposalChange {
  row_index: number;
  part_id: string;
  supplier_id: string;
  week: string | number;
  order_qty: number;
}

export interface Proposal {
  proposal_id: string;
  version: number;
  source_run_id: string;
  reverts_proposal_id: string | null;
  source_run_version: number;
  source_schedule_hash: string;
  request_text: string;
  intent: Record<string, unknown>;
  before_values: ProposalChange[];
  after_values: ProposalChange[];
  validation: {
    feasible?: boolean;
    failed_checks?: string[];
    [key: string]: unknown;
  };
  impact: Record<string, unknown>;
  warnings: string[];
  status: "validated" | "stale" | "canceled" | "expired" | "applied" | "rejected";
  proposer: Pick<User, "id" | "username" | "role">;
  approver: Pick<User, "id" | "username" | "role"> | null;
  expires_at: string;
  created_at: string;
  updated_at: string;
  applied_result: ResultReference | null;
}

export interface AuditRecord {
  audit_id: string;
  action: string;
  actor: Pick<User, "id" | "username" | "role">;
  proposal_id: string | null;
  run_id: string | null;
  before_values: unknown[];
  after_values: unknown[];
  validation: Record<string, unknown>;
  details: Record<string, unknown>;
  created_at: string;
  previous_hash: string;
  record_hash: string;
}

export interface ConversationMessage {
  id: number;
  direction: "user" | "assistant";
  content: string;
  intent: string;
  language: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface Conversation {
  conversation_id: string;
  language: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
}

export interface AssistantReply {
  conversation_id: string;
  message_id: number;
  response: string;
  intent: string;
  language: string;
  metadata: {
    auto_applied?: boolean;
    clarification_required?: boolean;
    explicit_confirmation_required?: boolean;
    proposal?: Proposal;
    [key: string]: unknown;
  };
}

export interface JwtResponse {
  access: string;
  refresh: string;
  user: User;
}

export interface ApiErrorPayload {
  detail?: string;
  [key: string]: unknown;
}
