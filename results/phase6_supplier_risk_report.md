# Supplier risk and reorder points in MRP (v4.2 A3/A4)

**Basis:** synthetic held-validation demand paths; one D2 snapshot. No historical
schedule accuracy is claimed.

## What changed

- Planning lead time per supplier slot: `lead + (1 - OTD) x average delay`.
- Order quantity is grossed up by `1 / (1 - reject rate)` before MOQ and order
  multiple rounding.
- The traditional baseline ignores both adjustments, by design.
- `supplier_risk_score = 1 - reliability_pct / 100` is display only.
  `supplier_risk_level` is low/medium/high by population tertiles of the
  SOB-weighted part score.
- `reorder_point = lead-time point demand + safety stock`;
  `reorder_trigger_week` is the first week projected inventory is at or below
  it.

## Ablation on identical demand paths

| Variant | Part-origins | Unresolved | Order lines | Order qty | Expected rejection loss | Mean safety stock | Projected unmet units | High-risk large-share warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Traditional baseline | 563 | 121 | 6718 | 766550 | 23022 | 776.6 | 225978 | 1582 |
| Risk-adjusted | 563 | 130 | 6342 | 788666 | 23825 | 801.8 | 243181 | 1258 |

Risk-adjusted order quantity changes by
**2.89%** versus the baseline.
Projected unmet units change by **17203**
because longer expected lead times delay receipts; higher order quantity does
not guarantee earlier coverage. This is a measured MRP-projection trade, not a
service improvement. No realized-savings claim is made.

## Reorder point comparison with D2

| Criticality | Part-origins | Median reorder point | Median current | Median delta | Median delta % | Above current | Below current |
|---|---:|---:|---:|---:|---:|---:|---:|
| High | 189 | 587.7 | 54.0 | 384.4 | 1046.9% | 187 | 2 |
| Low | 134 | 195.8 | 28.0 | 121.8 | 419.4% | 128 | 6 |
| Medium | 240 | 267.1 | 34.0 | 201.2 | 633.7% | 233 | 7 |

D2 `current_reorder_point_qty` is a snapshot value with no documented derivation.
The delta is reported, not treated as an error in either direction.

## Risk levels

high: 53, low: 53, medium: 52
parts.

## Limits

- The `high_share_warning_pct` threshold of
  50% is a provisional display
  parameter (`provisional_editable`), not a business rule.
- Reliability is used only for the display score; planning uses OTD and
  rejection.
- The simulator is not rerun, so no service-level claim is made for the
  risk-adjusted plan.
- The D2 snapshot postdates the validation origins; initial inventory is the
  same synthetic initialization as Phase 5.

## Outputs

`results/phase6_supplier_risk_weekly.csv`,
`results/phase6_supplier_risk_orders.csv`,
`results/phase6_supplier_risk_plans.csv`,
`results/phase6_supplier_risk_comparison.csv`,
`results/phase6_supplier_risk_summary.json`.
