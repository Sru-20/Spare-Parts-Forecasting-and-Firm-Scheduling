import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, Check, ChevronDown, ChevronRight, CircleHelp, FilePlus2, LockKeyhole, MessageSquareText, Send, ShieldCheck, Sparkles, UserRound } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import type { AssistantReply, Conversation, ConversationMessage, Pagination, Proposal } from "../api/types";
import { apiRequest } from "../api/client";
import { errorDetail, formatDate, formatNumber } from "../api/format";
import { useApiData } from "../api/hooks";
import { useAuth } from "../app/auth/AuthContext";
import { useRuns } from "../app/RunContext";
import { useToast } from "../app/ToastContext";
import { Button, InlineNotice, PageHeader, StateCard, StatusBadge } from "../design-system/ui";

function isProposal(value: unknown): value is Proposal {
  return typeof value === "object" && value !== null && "proposal_id" in value && "status" in value;
}

function randomKey() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function formatDateTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en-IN", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

export function AssistantPage() {
  const [searchParams] = useSearchParams();
  const { currentRun } = useRuns();
  const { user } = useAuth();
  const { notify } = useToast();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [activeAction, setActiveAction] = useState<string | null>(null);
  const prefilledPart = useRef(false);
  const conversationList = useApiData<Pagination<Conversation>>("/assistant/conversations/?page=1&page_size=30");
  const conversationPath = activeId ? `/assistant/conversations/${activeId}/` : null;
  const conversationDetail = useApiData<Conversation>(conversationPath);
  const proposalsRequest = useApiData<Pagination<Proposal>>("/proposals/?page=1&page_size=30");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!activeId && conversationList.data?.results.length) {
      setActiveId(conversationList.data.results[0]?.conversation_id ?? null);
    }
  }, [activeId, conversationList.data]);

  useEffect(() => {
    if (conversationDetail.data) setMessages(conversationDetail.data.messages);
  }, [conversationDetail.data]);

  useEffect(() => {
    const partId = searchParams.get("part_id");
    if (partId && !prefilledPart.current) {
      prefilledPart.current = true;
      setDraft(`Explain the saved schedule for part ${partId}.`);
    }
  }, [searchParams]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages]);

  const canPlan = Boolean(user && ["planner", "approver", "admin"].includes(user.role));

  const startConversation = () => {
    setActiveId(null);
    setMessages([]);
    setDraft("");
  };

  const sendMessage = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = draft.trim();
    if (!message || sending) return;
    setSending(true);
    setDraft("");
    const optimisticId = -Date.now();
    const idempotencyKey = randomKey();
    const optimisticUser: ConversationMessage = {
      id: optimisticId,
      direction: "user",
      content: message,
      intent: "",
      language: conversationDetail.data?.language ?? "en",
      metadata: {},
      created_at: new Date().toISOString(),
    };
    setMessages((items) => [...items, optimisticUser]);
    try {
      const reply = await apiRequest<AssistantReply>("/assistant/messages/", {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey },
        body: JSON.stringify({
          message,
          conversation_id: activeId ?? undefined,
          run_id: currentRun?.run_id,
          idempotency_key: idempotencyKey,
        }),
      });
      setActiveId(reply.conversation_id);
      setMessages((items) => [
        ...items,
        {
          id: reply.message_id,
          direction: "assistant",
          content: reply.response,
          intent: reply.intent,
          language: reply.language,
          metadata: reply.metadata,
          created_at: new Date().toISOString(),
        },
      ]);
      conversationList.reload();
      proposalsRequest.reload();
    } catch (requestError) {
      setMessages((items) => items.filter((item) => item.id !== optimisticId));
      setDraft(message);
      notify(errorDetail(requestError), "error");
    } finally {
      setSending(false);
    }
  };

  const performProposalAction = async (
    proposal: Proposal,
    action: "confirm" | "cancel" | "revert",
  ) => {
    setActiveAction(`${proposal.proposal_id}:${action}`);
    try {
      const headers = action === "cancel" ? undefined : { "Idempotency-Key": randomKey() };
      const body = action === "confirm"
        ? JSON.stringify({
            proposal_version: proposal.version,
            run_version: proposal.source_run_version,
          })
        : undefined;
      const requestOptions = {
        method: "POST" as const,
        ...(headers ? { headers } : {}),
        ...(body ? { body } : {}),
      };
      const updated = await apiRequest<Proposal>(
        `/proposals/${proposal.proposal_id}/${action}/`,
        requestOptions,
      );
      setMessages((items) => items.map((item) => {
        const existing = item.metadata.proposal;
        if (!isProposal(existing) || existing.proposal_id !== updated.proposal_id) return item;
        return { ...item, metadata: { ...item.metadata, proposal: updated } };
      }));
      conversationDetail.reload();
      conversationList.reload();
      proposalsRequest.reload();
      notify(
        action === "confirm" ? "Proposal applied to the simulated schedule." :
          action === "cancel" ? "Proposal canceled; the schedule was not changed." :
            "A new authorized revert was created for the simulated schedule.",
        "success",
      );
    } catch (requestError) {
      notify(errorDetail(requestError), "error");
    } finally {
      setActiveAction(null);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Controlled assistant"
        title="Ask AI"
        description="Ask in English, Hindi, Hinglish, or Marathi. The assistant retrieves approved data and interprets intent; deterministic services calculate and validate every proposal."
        actions={<Button variant="secondary" icon={FilePlus2} onClick={startConversation}>New conversation</Button>}
      />
      <InlineNotice tone="info" icon={LockKeyhole}>
        Read-only by default. “Looks good” in chat is not approval; only a separate authorized confirmation action can apply a proposal to a simulated schedule.
      </InlineNotice>
      <div className="assistant-layout">
        <aside className="assistant-history">
          <div className="assistant-history-heading"><span>Conversations</span><span>{conversationList.data?.count ?? 0}</span></div>
          {conversationList.loading ? <div className="skeleton skeleton-block" /> : conversationList.error ? (
            <StateCard kind="error" title="History unavailable" description={conversationList.error} />
          ) : !conversationList.data?.results.length ? (
            <div className="assistant-history-empty"><MessageSquareText size={17} /><span>Your saved conversations will appear here.</span></div>
          ) : (
            <div className="conversation-list">
              {conversationList.data.results.map((conversation) => {
                const firstUser = conversation.messages.find((item) => item.direction === "user");
                return (
                  <button
                    type="button"
                    key={conversation.conversation_id}
                    className={`conversation-link ${activeId === conversation.conversation_id ? "conversation-active" : ""}`}
                    onClick={() => setActiveId(conversation.conversation_id)}
                  >
                    <span className="conversation-link-icon"><MessageSquareText size={15} /></span>
                    <span className="conversation-link-copy">
                      <strong>{firstUser?.content || "New conversation"}</strong>
                      <small>{formatDate(conversation.updated_at)} · {conversation.language.toUpperCase()}</small>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </aside>

        <section className="assistant-workspace" aria-label="Assistant conversation">
          <div className="assistant-thread-heading">
            <span className="assistant-avatar"><Bot size={17} /></span>
            <div><strong>Planning assistant</strong><small>{currentRun ? `Using selected run ${currentRun.run_id.slice(0, 8)}` : "No run selected"}</small></div>
            <span className="assistant-mode"><ShieldCheck size={13} /> Controlled</span>
          </div>
          <div className="assistant-thread">
            {conversationDetail.loading && activeId ? <div className="skeleton skeleton-block" /> : conversationDetail.error ? (
              <StateCard kind="error" title="Conversation could not be loaded" description={conversationDetail.error} />
            ) : messages.length ? (
              messages.map((message) => (
                <MessageBubble
                  key={`${message.id}-${message.direction}`}
                  message={message}
                  currentUserId={user?.id ?? null}
                  role={user?.role ?? "viewer"}
                  onProposalAction={performProposalAction}
                  activeAction={activeAction}
                />
              ))
            ) : (
              <div className="assistant-welcome">
                <div className="assistant-welcome-icon"><Sparkles size={20} /></div>
                <h2>How can I help with this plan?</h2>
                <p>Ask about a forecast, order path, inventory position, warning, KPI, or catalogue category.</p>
                <div className="suggestion-grid">
                  {[
                    "Explain the schedule for this run",
                    "Show high-criticality warnings",
                    "Find parts in the Mechanical category",
                    "Explain forecast limits for this run",
                  ].map((suggestion) => (
                    <button type="button" key={suggestion} onClick={() => setDraft(suggestion)}>
                      <CircleHelp size={14} />{suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {sending && (
              <div className="message-row assistant-row">
                <span className="message-avatar"><Bot size={14} /></span>
                <div className="message-bubble assistant-bubble"><span className="typing-indicator"><i /><i /><i /></span><span>Checking approved results…</span></div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <form className="assistant-composer" onSubmit={(event) => void sendMessage(event)}>
            <label htmlFor="assistant-prompt" className="sr-only">Ask a question</label>
            <textarea
              id="assistant-prompt"
              maxLength={2000}
              rows={2}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask about saved forecasts, schedules, inventory, warnings, or KPIs…"
            />
            <div className="composer-footer">
              <span>{canPlan ? "Planner actions create proposals only." : "Read-only access · no schedule changes."} {draft.length}/2000</span>
              <Button variant="primary" icon={Send} loading={sending} disabled={!draft.trim()}>Send</Button>
            </div>
          </form>
          <p className="assistant-disclaimer">No external model calls. Never use the response as a purchase-order instruction without the approved workflow.</p>
        </section>

        <aside className="proposal-rail">
          <div className="proposal-rail-heading"><span>Proposal activity</span><span>{proposalsRequest.data?.count ?? 0}</span></div>
          {proposalsRequest.loading ? <div className="skeleton skeleton-block" /> : proposalsRequest.error ? (
            <StateCard kind="error" title="Proposals unavailable" description={proposalsRequest.error} />
          ) : !proposalsRequest.data?.results.length ? (
            <div className="assistant-history-empty"><ShieldCheck size={17} /><span>No proposal records are available to this account.</span></div>
          ) : (
            <div className="proposal-list">
              {proposalsRequest.data.results.map((proposal) => (
                <ProposalCard
                  key={proposal.proposal_id}
                  proposal={proposal}
                  currentUserId={user?.id ?? null}
                  role={user?.role ?? "viewer"}
                  onAction={performProposalAction}
                  activeAction={activeAction}
                  compact
                />
              ))}
            </div>
          )}
        </aside>
      </div>
    </>
  );
}

function MessageBubble({
  message,
  currentUserId,
  role,
  onProposalAction,
  activeAction,
}: {
  message: ConversationMessage;
  currentUserId: number | null;
  role: string;
  onProposalAction: (proposal: Proposal, action: "confirm" | "cancel" | "revert") => Promise<void>;
  activeAction: string | null;
}) {
  const userMessage = message.direction === "user";
  const proposal = isProposal(message.metadata.proposal) ? message.metadata.proposal : null;
  const results = message.metadata.results;
  const safeMetadata = Object.fromEntries(
    Object.entries(message.metadata).filter(([key]) => !["proposal", "results"].includes(key)),
  );
  return (
    <article className={`message-row ${userMessage ? "user-row" : "assistant-row"}`}>
      <span className="message-avatar">{userMessage ? <UserRound size={14} /> : <Bot size={14} />}</span>
      <div className="message-content">
        <div className="message-speaker">{userMessage ? "You" : `Assistant · ${message.language.toUpperCase()}`}<span>{formatDate(message.created_at)}</span></div>
        <div className={`message-bubble ${userMessage ? "user-bubble" : "assistant-bubble"}`}>{message.content}</div>
        {!userMessage && (
          <>
            {message.metadata.auto_applied === false && message.metadata.quantity_source && (
              <div className="message-safety"><ShieldCheck size={13} /> No automatic write · quantities {String(message.metadata.quantity_source)}</div>
            )}
            {message.metadata.blocked === true && <div className="message-safety message-blocked"><LockKeyhole size={13} /> Request blocked by policy</div>}
            {proposal && (
              <ProposalCard
                proposal={proposal}
                currentUserId={currentUserId}
                role={role}
                onAction={onProposalAction}
                activeAction={activeAction}
              />
            )}
            {results !== undefined && <ResultDisclosure result={results} />}
            {Object.keys(safeMetadata).length > 0 && (
              <details className="metadata-disclosure">
                <summary><ChevronRight size={13} /> Response metadata</summary>
                <pre>{JSON.stringify(safeMetadata, null, 2)}</pre>
              </details>
            )}
          </>
        )}
      </div>
    </article>
  );
}

function ResultDisclosure({ result }: { result: unknown }) {
  const [open, setOpen] = useState(false);
  const count = Array.isArray(result)
    ? result.length
    : typeof result === "object" && result !== null && "total" in result
      ? String(result.total)
      : "saved result";
  return (
    <details className="metadata-disclosure" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary><ChevronDown size={13} /> Retrieved result · {count}</summary>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </details>
  );
}

function ProposalCard({
  proposal,
  currentUserId,
  role,
  onAction,
  activeAction,
  compact = false,
}: {
  proposal: Proposal;
  currentUserId: number | null;
  role: string;
  onAction: (proposal: Proposal, action: "confirm" | "cancel" | "revert") => Promise<void>;
  activeAction: string | null;
  compact?: boolean;
}) {
  const isProposer = currentUserId !== null && proposal.proposer.id === currentUserId;
  const canCancel = proposal.status === "validated" && (isProposer || role === "admin");
  const canConfirm = proposal.status === "validated" && ["approver", "admin"].includes(role) && !isProposer;
  const canRevert = proposal.status === "applied" && ["approver", "admin"].includes(role) && !isProposer;
  const busy = activeAction?.startsWith(`${proposal.proposal_id}:`) ?? false;
  const impact = proposal.impact;
  const before = Array.isArray(proposal.before_values) ? proposal.before_values : [];
  const after = Array.isArray(proposal.after_values) ? proposal.after_values : [];
  return (
    <article className={`proposal-card ${compact ? "proposal-compact" : ""}`}>
      <div className="proposal-card-heading">
        <div><span className="eyebrow">Typed proposal · v{proposal.version}</span><strong>Order quantity change</strong></div>
        <StatusBadge status={proposal.status} />
      </div>
      <p className="proposal-request">{proposal.request_text}</p>
      <div className="proposal-context-row">
        <span>Run <strong>{proposal.source_run_id.slice(0, 8)}</strong></span>
        <span>Expires <strong>{formatDateTime(proposal.expires_at)}</strong></span>
      </div>
      <div className="proposal-values">
        <div><span>Before</span><strong>{formatNumber(impact.order_qty_before)}</strong><small>units</small></div>
        <span className="proposal-arrow">→</span>
        <div><span>Proposed</span><strong>{formatNumber(impact.order_qty_after)}</strong><small>units</small></div>
        <div className="proposal-delta"><span>Δ</span><strong>{formatNumber(impact.order_qty_delta)}</strong><small>units</small></div>
      </div>
      <details className="proposal-details">
        <summary><ChevronDown size={13} /> Before / after rows · {before.length} changes</summary>
        <div className="proposal-row-list">
          {after.slice(0, 12).map((change, index) => {
            const prior = before[index];
            return (
              <div key={`${change.part_id}-${change.row_index}`} className="proposal-change-row">
                <strong>{change.part_id}</strong><span>{change.supplier_id} · week {change.week}</span>
                <span>{formatNumber(prior?.order_qty)} → <strong>{formatNumber(change.order_qty)}</strong></span>
              </div>
            );
          })}
          {after.length > 12 && <span className="panel-footnote">{after.length - 12} more rows are stored in the proposal.</span>}
        </div>
      </details>
      {proposal.validation.failed_checks?.length ? (
        <div className="proposal-warning"><LockKeyhole size={14} /><span>Validation failed: {proposal.validation.failed_checks.join(", ")}</span></div>
      ) : (
        <div className="proposal-validation"><Check size={14} /><span>Saved deterministic validation: {String(proposal.validation.checked_orders ?? "recorded")} orders · feasible {String(proposal.validation.feasible ?? "not recorded")}</span></div>
      )}
      {proposal.warnings.map((warning) => <div className="proposal-warning" key={warning}><CircleHelp size={14} /><span>{warning}</span></div>)}
      {(canCancel || canConfirm || canRevert) && (
        <div className="proposal-actions">
          {canCancel && <Button variant="secondary" loading={Boolean(busy && activeAction?.endsWith(":cancel"))} disabled={busy} onClick={() => void onAction(proposal, "cancel")}>Cancel proposal</Button>}
          {canConfirm && <Button variant="primary" loading={Boolean(busy && activeAction?.endsWith(":confirm"))} disabled={busy} onClick={() => void onAction(proposal, "confirm")}>Confirm & apply</Button>}
          {canRevert && <Button variant="danger" loading={Boolean(busy && activeAction?.endsWith(":revert"))} disabled={busy} onClick={() => void onAction(proposal, "revert")}>Create authorized revert</Button>}
        </div>
      )}
      {proposal.status === "validated" && !canConfirm && !canCancel && (
        <div className="proposal-permission-note">Only a different authorized approver can confirm. Approval is not inferred from chat.</div>
      )}
      {proposal.status === "validated" && canConfirm && (
        <div className="proposal-permission-note">This action applies a simulated schedule version; it does not connect to ERP or purchasing systems.</div>
      )}
    </article>
  );
}
