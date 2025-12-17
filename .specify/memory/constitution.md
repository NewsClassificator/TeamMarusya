<!--
Sync Impact Report
Version: 1.0.0 → 1.1.0
Modified Principles: Python as Source of Truth → clarified execution boundaries; Explicit Interface Between Frontend and Backend → mandated contract artifacts; Single Responsibility per Module → tightened enforcement
Added sections: Governance > Authority & Scope; Governance > Amendment Process; Governance > Compliance & Review
Removed sections: None
Templates requiring updates: ✅ .specify/templates/plan-template.md; ✅ .specify/templates/spec-template.md
Follow-up TODOs: None
-->

# Rocker//Score Constitution

## Core Principles

### I. Python as Source of Truth
All core logic (parsing articles, sentiment analysis, data extraction) MUST live in Python packages.
Next.js is strictly a presentation and orchestration layer and MUST NOT implement business logic,
ML logic, heuristics, or data extraction rules. Python code MUST remain executable independently of
the frontend through a CLI or HTTP entry point.

---

### II. Explicit Interface Between Frontend and Backend
Interaction between Next.js and Python MUST be explicit, deterministic, and defined in repository
contracts. Allowed interaction methods are HTTP APIs (FastAPI/Flask) or CLI invocation with JSON
input/output. Hidden state sharing, implicit runtime assumptions, and logic duplication are
forbidden. Request and response schemas MUST be documented and versioned before integration.

---

### III. Single Responsibility per Module
Each Python module owns exactly one responsibility:
- Parser: extract title, text, author, date from URL
- Sentiment analyzer: evaluate emotional tone of text
- Frontend: user input, visualization, orchestration

Modules MUST NOT mix unrelated tasks, and any deviation requires a documented justification before
merging.

---

### IV. Frontend as Thin Client
Next.js frontend:
- Collects user input (article URL or text)
- Sends input to Python layer
- Displays structured results

Frontend MUST NOT:
- Implement ML logic
- Re-implement parsing rules
- Modify model behavior

All computation happens in Python; frontend changes MUST NOT affect inference outcomes.

---

### V. Deterministic and Reproducible Execution
Given the same input, Python code MUST produce the same output.

Requirements:
- Fixed model versions and seeds
- Explicit preprocessing steps
- No hidden randomness unless documented and justified

This is mandatory for debugging, evaluation, and grading.

---

## Architecture Constraints

### Technology Stack
- Backend: Python 3.x
- ML: PyTorch + HuggingFace Transformers
- Parsing: requests / BeautifulSoup / newspaper3k (or equivalent)
- Frontend: Next.js (App Router)
- Environment management: uv

No alternative stacks are introduced without justification.

---

### Runtime Model
- Python runs as a separate process or service
- Frontend communicates via HTTP or CLI
- Local development must work without cloud dependencies

The system MUST be runnable on a local machine (Steam Deck / Linux).

---

## Development Workflow

### Incremental Development
Development proceeds in small, verifiable steps:
1. Python script works standalone
2. Script exposes a clean interface (function / CLI / API)
3. Frontend calls this interface
4. Output is rendered

No step is skipped.

---

### Validation Before Integration
Before connecting to frontend, Python code MUST:
- Run without errors
- Produce structured output (dict / JSON)
- Handle invalid input gracefully

Broken backend code MUST NOT be “fixed” on frontend side.

---

## Governance

### Authority & Scope
This constitution supersedes all other development preferences. Complexity is considered a bug unless
proven otherwise.

### Amendment Process
Changes that move logic from Python to the frontend, introduce implicit coupling, or break
reproducibility MUST be justified with evidence and reviewed before merging. Amendments require a
pull request that: (1) cites the impacted principles, (2) updates the version, and (3) includes a
Sync Impact Report summarizing downstream template updates.

### Versioning Policy
Constitution versions follow semantic versioning:
- MAJOR: Backward-incompatible shifts to principles or governance
- MINOR: New principles/sections or materially expanded guidance
- PATCH: Clarifications or editorial corrections

### Compliance & Review
Constitution checks MUST run during plan creation and before merges that touch architecture or
interfaces. Any approved deviations MUST be recorded in the plan's Complexity Tracking table with a
clear sunset plan. Release candidates MUST confirm deterministic execution and Python ownership of
logic before shipping.

**Version**: 1.1.0  
**Ratified**: 2025-12-14  
**Last Amended**: 2025-12-14
