# Phase 5 Appendix A — Scheduling factor traceability

The validation origins precede D2's single snapshot (`as_of_date=2026-09-26`).
Consequently the snapshot is not known at those historical forecast origins;
Phase 5 reuses it only as a synthetic initialization, identically for each
validation path. It is not treated as historical state or as a forecast feature.

| Factor | Verified source columns | Known at validation forecast origin? | Missing-data handling | Status |
|---|---|---|---|---|
| Lead time | D2 `sup1_lead_time_days`, `sup2_lead_time_days`, `sup3_lead_time_days`; corresponding `supN_sob_pct` | No; D2 snapshot is later than the held origins. Known at the snapshot date and used for simulated planning only. | Ignore blank supplier slots; use SOB-weighted lead time over present eligible suppliers. If no valid supplier/lead time remains, return unresolved, never zero. | Direct snapshot input; historical availability approximated by synthetic reuse |
| Inventory | D2 `on_hand_qty`, `reserved_qty`, `blocked_qty`, `available_qty`; `in_transit_qty` | No at historical origins; D2 is used as synthetic initialization only. | Require exact `on_hand_qty - reserved_qty - blocked_qty = available_qty`; mismatch or missing required fields is an explicit error/unresolved result. Do not impute stock. | Direct snapshot input; initialization is approximated |
| Open orders and pending requirements | D2 `open_po_qty`, `open_po_due_this_month_qty`, `open_po_due_later_qty`, `earliest_open_po_eta`, `in_transit_qty`, `backorder_qty` | No at historical origins; dates/quantities are post-origin snapshot values. | Require due buckets to reconcile to `open_po_qty`; count the split once. No individual PO IDs/dates exist. In the synthetic replay, due-this-month and in-transit receipts map to week 1, due-later to week 4. Use `backorder_qty` once as the labelled pending-requirement assumption; no ETA means use the documented synthetic bucket rather than zeroing quantity. | Approximated |
| Supplier / production capacity and allocation | D2 `supply_mode`, `supN_id`, `supN_type`, `supN_sob_pct`, `supN_monthly_capacity`, `supN_capacity_committed_pct`, `supN_moq`, `supN_order_multiple`, `supN_otd_pct`, `supN_avg_delay_days`, `supN_reject_pct`, `supN_reliability_pct` | No at historical origins; snapshot values drive the synthetic scheduling replay. | Blank supplier slots are unavailable and not filled. Available capacity is monthly capacity times one minus committed share. If no eligible capacity can cover the requirement, retain only feasible allocations and mark unresolved. | Direct snapshot inputs with approximated availability/reliability; SOB is a target |
| Criticality and service level | D2 `criticality`, `service_level_target`; D3 `criticality`, `service_level_target` | No at held forecast origins; D2/D3 snapshot metadata is used for validation simulation only. | Use D2 target only when present and within (0,1); otherwise unresolved. No default target is invented. | Direct snapshot input; synthetic historical reuse |

## SOB evidence

The executed D2 inspection found 158/158 rows with supplier SOB values summing
to exactly 100%. In `generate_datasets.py`, shares are assigned from patterns
of 100%, a 60/40-or-70/30-or-80/20 split, or 50/30/20, then used as weights
when distributing synthetic supplier load. The files contain no contract,
penalty, or rejection rule making those shares binding. Therefore SOB is treated
as a target allocation: deviations in realized allocations are recorded in
percentage points and are not hard constraint violations.

## Lead-time and snapshot limits

Orders released at week `r` arrive at `r + max(1, ceil(lead_time_days / 7))`.
The supplied 14-week forecast is extended for planning lead demand beyond 14
weeks using the mean of its final four forecast weeks; orders still retain their
calculated arrival week and are marked `arrives_beyond_horizon`, without
counting as in-horizon receipts. The actual demand path is never extended for
scoring.

D2 has no individual inbound order identifiers or receipt dates, and its
snapshot postdates every held validation origin. Synthetic receipt offsets
therefore make the inbound quantities usable for a repeatable simulator, not
historical claims about receipt timing. This limitation must remain visible
beside simulated KPIs.
