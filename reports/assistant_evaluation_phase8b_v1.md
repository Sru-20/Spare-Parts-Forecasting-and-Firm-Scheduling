# Phase 8B assistant evaluation

**Claim scope:** deterministic rule-based intent extraction tested on generated
multilingual paraphrase fixtures; no fluent-human review was performed.

- Requests: 192 labeled; development 128;
  held-out test 64.
- Held-out intent accuracy: 1.000.
- Held-out entity accuracy: 1.000
  (16 applicable cases).
- Blocked-intent recognition: 1.000
  (16 injection, bypass, invalid-percentage and hard-constraint phrase fixtures).
  This offline score measures intent labeling, not end-to-end authorization or
  schedule validation; those are exercised by database-backed API tests.
- Cross-language paraphrase consistency: 1.000.
- Incorrect auto-apply rate: 0.000.
- Invented-quantity rate: 0.000.
- Local intent median / p95 latency: 0.021 /
  0.032 ms; external model calls/cost: 0 / 0.
- Audit completeness is not applicable to offline intent-only samples; lifecycle
  writes are covered by database-backed API tests and persist hash-chained audit
  entries.

The held-out set uses the third paraphrase variant; the first two are the
development set. These are generated test phrases, not independent human
requests. Accuracy indicates paraphrase robustness only, not production
language understanding or fluent multilingual quality.

Per-language held-out intent accuracy: {'en': {'samples': 16, 'intent_accuracy': 1.0}, 'hi': {'samples': 16, 'intent_accuracy': 1.0}, 'hinglish': {'samples': 16, 'intent_accuracy': 1.0}, 'mr': {'samples': 16, 'intent_accuracy': 1.0}}.

Request-level labels and predictions: `results/assistant_requests_phase8b_v1.csv`.
Metric details: `results/assistant_evaluation_phase8b_v1.json`.
