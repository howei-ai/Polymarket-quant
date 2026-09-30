<div align="center">

# Polymarket-quant

[English](README.md) · [简体中文](README.zh-CN.md)

### Agentic Alpha Research with Deterministic Execution Governance

**Let AI research and propose. Let deterministic systems constrain, audit, and control execution authority.**

![Agent Shadow](https://img.shields.io/badge/K3%2FJev%20Agent%20Shadow-MERGED-2ea44f?style=flat-square)
![Post-merge CI](https://img.shields.io/badge/Post--merge%20CI-PASS-2ea44f?style=flat-square)
![Formal Backtest](https://img.shields.io/badge/Formal%20Backtest-NOT%20RUN-f0ad4e?style=flat-square)
![LIVE](https://img.shields.io/badge/LIVE-STOP-d73a49?style=flat-square)

</div>

---

`std0-quant` is an auditable quantitative research system for **Polymarket BTC 5-minute Up/Down market microstructure research**.

The system separates reasoning from authority:

- **K3 / Agent layer:** analyzes point-in-time public state and produces structured research.
- **Jev:** evaluates fixed, typed questions over the same bounded research state.
- **Deterministic code:** enforces evidence, provenance, coverage, schema, and execution boundaries.
- **Execution layer:** stays outside model authority.

> **K3 = REASONER · JEV = JUDGE · DETERMINISTIC CODE = AUTHORITY · SHADOW = ONLY OUTPUT**

AI can research and propose. AI cannot hold LIVE credentials, bypass deterministic gates, mutate formal cohort state, perform an actual publication, authorize a formal backtest, or submit real orders.

## Architecture

![std0-quant architecture](docs/assets/architecture.jpg)

```text
Point-in-Time Data / Features
            │
            ▼
      K3 Analyst Team
            │
            ▼
       Bull / Bear
            │
            ▼
    Research Manager
            │
            ▼
     Jev Typed Judge
            │
            ▼
   Deterministic Gate
            │
            ▼
        SHADOW ONLY
```

The current K3 / Jev layer is a **research classification layer**, not a trading engine.

```text
AGENT SHADOW ACCEPT
        ≠
STRATEGY EXECUTION SHADOW
        ≠
FORMAL BACKTEST
        ≠
PRODUCTION ELIGIBLE
        ≠
LIVE
```

## K3 / Jev Agent Shadow

The agent-shadow implementation lives under:

```text
src/std0_quant/research/agent_shadow/
```

Core properties:

- fixed analyst roles;
- one bounded Bull/Bear debate round;
- one Research Manager synthesis step;
- explicit role-specific instructions;
- typed Jev question contracts;
- explicit Jev model version pinning;
- request / evidence hash binding;
- point-in-time input allowlists;
- deterministic fail-closed classification;
- temporary-only shadow artifact writing;
- no order or production-authority capability.

### Deterministic authority

The final shadow classification is made by code, not by model text.

Possible outcomes:

```text
SHADOW_ACCEPT
SHADOW_REJECT
BLOCKED
```

A model cannot directly turn a research opinion into an executable instruction.

## Point-in-Time Boundary

The agent layer only receives explicitly allowlisted pre-cutoff public features.

Examples include:

- BTC price / return / realized-volatility features;
- BTC trade-count / volume / signed-flow features;
- Polymarket mid / spread / depth / OBI features;
- short-horizon book-update features;
- explicit observation timestamps;
- measured coverage fields.

Unknown or forbidden fields fail closed. Observed timestamps must not exceed the decision cutoff.

## Coverage Gates

The current research boundary keeps the existing coverage thresholds unchanged:

```text
BTC_PRE30  >= 0.99
BOOK_PRE10 >= 0.99
```

Passing a coverage or numeric-sanity rule does **not** imply:

- formal cohort membership;
- formal strategy eligibility;
- complete lineage;
- profitability;
- backtest approval;
- LIVE readiness.

## Publication / Provenance

Publication / Provenance v2 is merged into `main` as a versioned publication layer.

It is designed to preserve:

- behavioral truth;
- reconciliation evidence;
- provenance membership;
- artifact identity;
- hash-bound auditability.

Important distinction:

```text
PUBLICATION / PROVENANCE CODE MERGED
                ≠
ACTUAL PUBLISH EXECUTED
```

The repository does **not** claim that an actual publication has been run.

## Safety Boundary

The LLM / agent layer is not part of the millisecond execution path.

The agent layer cannot:

- load private keys or LIVE credentials;
- submit real venue orders;
- bypass deterministic risk controls;
- mutate frozen research semantics;
- overwrite pinned feature or provenance artifacts;
- write the formal cohort;
- authorize a formal backtest;
- perform an actual publication;
- authorize Production Eligibility.

The system is designed to **fail closed** when identity, point-in-time state, evidence, provenance, coverage, schema, or audit hashes are inconsistent.

## Current Status

| Layer | Status |
| --- | :---: |
| Sanity Audit v2 | FROZEN |
| 243-conflict forensics | CLOSED / PASS |
| Publication / Provenance v2 code | MERGED TO `main` |
| Actual publish | NOT RUN |
| K3 / Jev agent research shadow | MERGED TO `main` |
| Post-merge GitHub Actions CI | PASS |
| Formal cohort | EMPTY |
| Formal backtest | NOT RUN |
| Strategy execution shadow | NOT STARTED |
| Real order submission | DISABLED |
| **LIVE execution** | **STOP** |

Verified code baseline after PR #28:

```text
main = 1cbca74943225ec40fd46bfafcd77888b2a8150f
```

K3 / Jev feature commit:

```text
e0095ed8de17708cc66fd21c287d1c4ce4f51d22
```

## Repository Layout

```text
src/std0_quant/
├── alpha/
├── events/
├── execution/
├── factors/
├── features/
├── registry/
├── research/
│   └── agent_shadow/
└── risk/

tests/
release/
manifest/
bootstrap/
```

## Quick Start

```bash
git clone https://github.com/howei-ai/Polymarket-quant.git
cd Polymarket-quant
```

This repository is intended for:

- quantitative research;
- point-in-time feature analysis;
- agent-assisted research;
- execution validation;
- provenance / publication governance;
- production-readiness controls.

It should **not** be interpreted as:

- an AI auto-profit bot;
- proof of profitability;
- a completed formal backtest;
- a deployed LIVE trading system;
- completed production trading authorization.

## Governance Model

The repository follows an explicit staged workflow:

```text
develop / test
      ↓
independent review
      ↓
local commit
      ↓
remote push
      ↓
pull request
      ↓
CI
      ↓
merge
      ↓
post-merge CI
      ↓
separate server / publish / backtest / LIVE gates
```

Authorization for one stage does not automatically authorize the next stage.

## Design Principle

```text
Observe
  ↓
Research
  ↓
Candidate
  ↓
Validate
  ↓
Version
  ↓
Govern
  ↓
Shadow
```

Never:

```text
Model Opinion → Production
```

---

<div align="center">

**K3 REASONS · JEV JUDGES · DETERMINISTIC CODE DECIDES · LIVE STAYS GATED**

</div>
