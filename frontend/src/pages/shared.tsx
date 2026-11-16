import { useState } from "react";
import { ArrowLeft, ArrowRight, Search, X } from "lucide-react";
import type { SavedRows, Part, Pagination } from "../api/types";
import { queryString, useApiData } from "../api/hooks";
import { Button, LoadingState, StateCard } from "../design-system/ui";

export function RunRequired({ title = "No registered run" }: { title?: string }) {
  return (
    <StateCard
      title={title}
      description="This view reads registered results only. An administrator must register a versioned run before saved planning data can appear."
    />
  );
}

export function RequestState({
  loading,
  error,
  empty,
  children,
}: {
  loading: boolean;
  error: string | null;
  empty: boolean;
  children: React.ReactNode;
}) {
  if (loading) return <LoadingState />;
  if (error) {
    return (
      <StateCard
        kind="error"
        title="Saved data could not be loaded"
        description={error}
      />
    );
  }
  if (empty) {
    return (
      <StateCard
        title="No saved records"
        description="There are no records for this selection. No values are inferred or substituted."
      />
    );
  }
  return children;
}

export function SavedRowsView({
  path,
  children,
}: {
  path: string | null;
  children: (rows: SavedRows, reload: () => void) => React.ReactNode;
}) {
  const request = useApiData<SavedRows>(path);
  return (
    <RequestState
      loading={request.loading}
      error={request.error}
      empty={!request.data || request.data.results.length === 0}
    >
      {request.data && children(request.data, request.reload)}
    </RequestState>
  );
}

export function PagedListView<T>({
  path,
  children,
}: {
  path: string | null;
  children: (page: Pagination<T>) => React.ReactNode;
}) {
  const request = useApiData<Pagination<T>>(path);
  return (
    <RequestState
      loading={request.loading}
      error={request.error}
      empty={!request.data || request.data.results.length === 0}
    >
      {request.data && children(request.data)}
    </RequestState>
  );
}

export function PaginationBar({
  page,
  pageSize,
  total,
  onChange,
}: {
  page: number;
  pageSize: number;
  total: number;
  onChange: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const first = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const last = Math.min(page * pageSize, total);
  return (
    <div className="pagination-bar">
      <span>
        Showing <strong>{first}-{last}</strong> of <strong>{total}</strong>
      </span>
      <div className="pagination-actions">
        <Button
          variant="quiet"
          icon={ArrowLeft}
          disabled={page <= 1}
          onClick={() => onChange(page - 1)}
          aria-label="Previous page"
        >
          Previous
        </Button>
        <span className="pagination-count">{page} / {pages}</span>
        <Button
          variant="quiet"
          icon={ArrowRight}
          disabled={page >= pages}
          onClick={() => onChange(page + 1)}
          aria-label="Next page"
        >
          Next
        </Button>
      </div>
    </div>
  );
}

export function PartPicker({
  selected,
  onSelect,
}: {
  selected: Part | null;
  onSelect: (part: Part | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const path = query.trim()
    ? `/parts/?${queryString({ q: query.trim(), page: 1 })}`
    : null;
  const { data, loading, error } = useApiData<Pagination<Part>>(path);

  if (selected) {
    return (
      <div className="selected-part">
        <span title={selected.part_name}>{selected.part_id} · {selected.part_name}</span>
        <Button variant="quiet" icon={X} onClick={() => onSelect(null)} aria-label="Clear selected part" />
      </div>
    );
  }

  return (
    <div className="part-picker">
      <label className="sr-only" htmlFor="part-search">Search catalogue parts</label>
      <div className="search-field">
        <Search size={15} aria-hidden="true" />
        <input
          id="part-search"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          placeholder="Search catalogue parts"
          autoComplete="off"
        />
      </div>
      {open && query.trim() && (
        <div className="part-picker-menu">
          {loading && <div className="picker-message">Searching catalogue…</div>}
          {error && <div className="picker-message picker-error">{error}</div>}
          {!loading && !error && !data?.results.length && (
            <div className="picker-message">No catalogue match for “{query}”.</div>
          )}
          {data?.results.map((part) => (
            <button
              type="button"
              className="part-picker-option"
              key={part.part_id}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                onSelect(part);
                setQuery("");
                setOpen(false);
              }}
            >
              <span className="part-option-id">{part.part_id}</span>
              <span className="part-option-name">{part.part_name}</span>
              <span className="part-option-category">{part.category || "Unclassified"}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function pathForRows(
  endpoint: "forecasts" | "schedules",
  runId: string | null,
  values: Record<string, string | number | undefined | null>,
) {
  if (!runId) return null;
  const query = queryString({ run_id: runId, page: 1, page_size: 100, ...values });
  return `/${endpoint}/?${query}`;
}

export function isoWeekLabel(value: string) {
  if (/^\d+$/.test(value)) return `Week ${value}`;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat("en", { month: "short", day: "numeric", timeZone: "UTC" }).format(parsed);
}
