import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatNumber } from "../api/format";

export interface TrendSeries {
  key: string;
  label: string;
  color: string;
}

interface LineTrendProps {
  data: Array<Record<string, string | number | null>>;
  series: TrendSeries[];
  valueLabel: string;
}

const colors = [
  "var(--electric-blue)",
  "var(--teal)",
  "var(--violet-accent)",
  "var(--purple-accent)",
];

interface TrendTooltipItem {
  dataKey?: string | number;
  name?: string;
  color?: string;
  value?: string | number;
}

function TrendTooltip({
  active,
  payload,
  label,
  valueLabel,
}: {
  active?: boolean;
  payload?: TrendTooltipItem[];
  label?: string | number;
  valueLabel: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-heading">{String(label)}</div>
      {payload.map((entry) => (
        <div className="chart-tooltip-row" key={String(entry.dataKey)}>
          <span>
            <i style={{ background: entry.color }} />
            {entry.name}
          </span>
          <strong>{formatNumber(entry.value, 1)} {valueLabel}</strong>
        </div>
      ))}
    </div>
  );
}

export function LineTrend({ data, series, valueLabel }: LineTrendProps) {
  return (
    <>
      <div className="chart-legend">
        {series.map((item) => (
          <span key={item.key}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 5" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fill: "var(--muted)", fontSize: 13 }}
              axisLine={{ stroke: "var(--line)" }}
              tickLine={false}
              minTickGap={18}
            />
            <YAxis
              tick={{ fill: "var(--muted)", fontSize: 13 }}
              axisLine={false}
              tickLine={false}
              width={44}
            />
            <Tooltip content={<TrendTooltip valueLabel={valueLabel} />} />
            {series.map((item) => (
              <Line
                key={item.key}
                type="monotone"
                dataKey={item.key}
                name={item.label}
                stroke={item.color}
                strokeWidth={2.25}
                dot={{ r: 2, strokeWidth: 0 }}
                activeDot={{ r: 4, strokeWidth: 0 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

export function chartColor(index: number) {
  return colors[index % colors.length] ?? colors[0]!;
}
