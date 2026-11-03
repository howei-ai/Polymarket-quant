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
- provider routes: `official` (Anthropic / OpenAI APIs) and `relay_mtu` (the third-party relay `https://mtu.mtuopenai.xyz`, key `STD0_RELAY_API_KEY`). Base URLs and key variables are pinned per route; relay clients refuse to build while official credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_ADMIN_KEY`, `OPENAI_WEBHOOK_SECRET`, `ANTHROPIC_AUTH_TOKEN`) are present in the given or the process environment. Transports enforce a per-call route policy: a relay call is refused unless it runs inside an explicit opt-in (`ShadowRunner(..., allow_unverified_relay=True)` or `route_policy(allow_unverified_relay=True)`), and all calls of one row must share one route, however the models are wrapped (checked on the calling thread; nested `route_policy` scopes can only narrow the enclosing one). `ShadowRunner` also blocks before any call when registered models use a relay without the opt-in (`RELAY_NOT_ALLOWED`), use different routes or put a relay model next to an unregistered one (`MODEL_ROUTE_MIXED`), or mix offline stand-ins with real clients (`TEST_CLIENT_MIXED`). Artifacts carry `route` and `provenance` (`OFFICIAL`, `UNVERIFIED_RELAY`, `TEST_CLIENT` only when both models are stand-ins, `UNSPECIFIED`), and the schema ties both to the call routes;
- model provenance (artifact schema `k3_jev_agent_shadow_v4`): provider labels and per-call records are taken only from the registered `K3Client` / `JevClient` around the registered transports; other model objects are labelled `unspecified` and their self-reported call records are discarded, and inconsistent provenance or a budget overrun blocks the row;
- offline test stand-ins run under a runtime guard (an audit hook) that, on the calling thread and for the duration of the provider call, refuses new sockets, connections, DNS lookups, thread starts, `subprocess` / `os.system` / `os.exec*` / `os.spawn*` / `posix_spawn` / `fork` and ctypes calls (a first-time import that loads a native library inside the call is refused too).

### Threat model for model provenance and offline stand-ins

Covered (accidental misuse): a test stand-in that is, plainly holds, hides or lazily builds a real SDK client and opens a new connection during its call; a factory-built client swapped into a test transport (or a fake into a real one); swapped bindings or prices (including `ModelBinding` subclasses); instance-level overrides of the registered clients' and transports' methods; a registered client whose transport was replaced; calls that leave no record; forged or relabelled model objects; inconsistent call records; budget overruns. These are refused, recorded as `CLIENT_UNTRUSTED` / `BUDGET_EXCEEDED`, or block the row with `MODEL_PROVENANCE_INVALID` / `BUDGET_OVERRUN`.

Not covered (needs signed artifacts or process isolation): deliberate tampering by code running in the same process (for example rebuilding the registered classes around a genuine client, or editing guard state); network I/O on threads, event loops or connections that existed before the guarded call (for example a stand-in reusing a pooled keep-alive connection of an already-used real client); native code that raises no audit events, including child processes started by `multiprocessing`'s default spawn method on macOS; official calls made from threads started inside a route-policy scope (context variables do not reach them, so they are not checked against the row's route; relay calls there are still refused); monkeypatching methods of a genuine factory-built client (the test suite uses this as a seam, and such records are labelled as real calls). The relay route is outside every verification above: the relay sees every request in full, and its backend model, billing and data handling cannot be verified (only the served-model name and output schema are checked; request ids are not returned). Rows from registered relay models are labelled `UNVERIFIED_RELAY` (rows involving unregistered or wrapped models are `UNSPECIFIED`) and must be kept apart from official results; only synthetic data may be sent through the relay until real research data is separately authorised. Overall, Shadow artifacts are research evidence of what the in-process code recorded, not tamper-proof attestations.

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
