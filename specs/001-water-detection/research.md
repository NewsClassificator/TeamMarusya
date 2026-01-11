# Phase 0 Research — Water Content Detection

**Feature**: /home/termenater/vibe/specs/001-water-detection/spec.md  
**Date**: 2025-12-22  
**Status**: Completed — no open NEEDS CLARIFICATION items

## Findings

- **Decision**: Expose water detection via HTTP POST `/api/water-detection` with JSON input.
  - **Rationale**: Aligns with existing FastAPI pattern and keeps frontend thin while supporting reuse.
  - **Alternatives considered**: CLI invocation (would complicate frontend orchestration); WebSocket/streaming (unnecessary for single-shot analysis).

- **Decision**: Accept `text` 20–10,000 chars, optional `include_features` (default true), optional `language` hint.
  - **Rationale**: Bounds prevent trivial/oversized payloads and align with spec’s need for meaningful samples without perf risk.
  - **Alternatives considered**: Unlimited text (risks latency/memory); tighter cap (<5k) (might block long articles).

- **Decision**: Return `is_water`, `label`, `confidence`, `features`, optional `interpretations`, `version`, and `errors` array on failures.
  - **Rationale**: Matches spec transparency goals and supports UI explanations; errors field standardizes validation feedback.
  - **Alternatives considered**: Minimal binary response (less helpful for UX and debugging); omitting interpretations (would reduce user understanding).

- **Decision**: Default to deterministic model artifact already bundled with `code/water/water_analyzer.py`; no external calls.
  - **Rationale**: Satisfies constitution for reproducibility and offline use; leverages existing trained model.
  - **Alternatives considered**: Remote inference (breaks offline requirement); retraining (out of scope for integration).

- **Decision**: UI renders label + confidence inline with analyzed text and reveals interpretations on demand.
  - **Rationale**: Provides immediate signal without clutter; preserves layout and accessibility.
  - **Alternatives considered**: Separate results page (adds friction); auto-expanding full feature list (risks noise).

- **Decision**: Fallback behavior on timeout/failure returns neutral “status unavailable” and keeps page usable.
  - **Rationale**: Upholds graceful degradation requirement and prevents blocking other content.
  - **Alternatives considered**: Hard failure/blank space (poor UX); silent retry loop (possible flicker/inconsistency).
