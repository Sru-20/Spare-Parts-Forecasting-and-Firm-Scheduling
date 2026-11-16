export function numberFrom(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string" || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatNumber(value: unknown, maximumFractionDigits = 0) {
  const numeric = numberFrom(value);
  if (numeric === null) return "Not measurable";
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits,
    minimumFractionDigits: 0,
  }).format(numeric);
}

export function formatCurrency(value: unknown, currency = "INR") {
  const numeric = numberFrom(value);
  if (numeric === null) return "Not measurable";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(numeric);
}

export function formatPercent(value: unknown, storedAsPercent = false, digits = 1) {
  const numeric = numberFrom(value);
  if (numeric === null) return "Not measurable";
  const percentage = storedAsPercent ? numeric : numeric * 100;
  return `${new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(percentage)}%`;
}

export function formatDate(value: string | null | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

export function safeString(value: unknown, fallback = "Not recorded") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

export function listValue(value: unknown) {
  if (Array.isArray(value)) return value.map((item) => String(item)).filter(Boolean);
  if (typeof value === "string" && value.trim()) {
    return value.split(/[|;,]/).map((item) => item.trim()).filter(Boolean);
  }
  return [];
}

export function errorDetail(error: unknown) {
  if (error instanceof Error) return error.message;
  return "The request could not be completed.";
}
