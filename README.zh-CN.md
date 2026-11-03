<div align="center">

# Polymarket-quant

[English](README.md) · [简体中文](README.zh-CN.md)

### Agentic Alpha Research with Deterministic Execution Governance

**让 AI 负责研究与提出候选，让确定性系统负责约束、审计与执行权限。**

![Agent Shadow](https://img.shields.io/badge/K3%2FJev%20Agent%20Shadow-MERGED-2ea44f?style=flat-square)
![Post-merge CI](https://img.shields.io/badge/Post--merge%20CI-PASS-2ea44f?style=flat-square)
![Formal Backtest](https://img.shields.io/badge/Formal%20Backtest-NOT%20RUN-f0ad4e?style=flat-square)
![LIVE](https://img.shields.io/badge/LIVE-STOP-d73a49?style=flat-square)

</div>

---

`std0-quant` 是一个面向 **Polymarket BTC 5 分钟 Up/Down 市场微观结构研究** 的可审计量化研究系统。

系统把“推理”与“权力”严格分离：

- **K3 / Agent 层：**分析 point-in-time 公共状态，生成结构化研究结果；
- **Jev：**针对同一受限研究状态回答固定、强类型问题；
- **确定性代码：**约束 evidence、provenance、coverage、schema 与执行边界；
- **执行层：**始终位于模型权限之外。

> **K3 = REASONER · JEV = JUDGE · DETERMINISTIC CODE = AUTHORITY · SHADOW = ONLY OUTPUT**

AI 可以研究、分析、提出候选；AI 不能持有 LIVE 凭证、绕过确定性 gate、修改 formal cohort、执行 actual publish、自行授权 formal backtest，或提交真实订单。

## 架构

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

当前 K3 / Jev 层是**研究分类层**，不是交易引擎。

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

Agent Shadow 实现在：

```text
src/std0_quant/research/agent_shadow/
```

核心约束：

- 固定 analyst roles；
- 仅一轮受限 Bull/Bear debate；
- 一次 Research Manager synthesis；
- 明确的 role-specific instructions；
- 强类型 Jev question contracts；
- 显式 Jev model version pinning；
- request / evidence hash binding；
- point-in-time input allowlists；
- 确定性 fail-closed classification；
- 仅允许写入临时 shadow artifact；
- 不具备订单或生产权限。
- 模型来源校验（产物 schema `k3_jev_agent_shadow_v3`）：供应商标签和逐次调用记录只取自注册的 `K3Client` / `JevClient` 及其注册 transport；其他模型对象标为 `unspecified`，其自报的调用记录会被丢弃；来源不一致或预算超支时该行判为 BLOCKED；
- 离线测试替身在调用期间受运行时防线（审计钩子）约束：仅针对发起调用的线程、仅在调用期间，拒绝新建 socket、建立连接、DNS 解析、启动线程、`subprocess` / `os.system` / `os.exec*` / `os.spawn*` / `posix_spawn` / `fork` 以及 ctypes 调用（调用期间首次导入并加载原生库也会被拒绝）。

### 模型来源与离线替身的威胁边界

覆盖（误用）：测试替身本身是、明显持有、隐藏或在调用时临时创建真实 SDK 客户端，并在调用期间新建连接；把工厂创建的真实客户端换进测试 transport（或把假客户端换进正式 transport）；替换绑定或价格（包括 `ModelBinding` 子类）；在实例上覆盖注册客户端或 transport 的方法；注册客户端的 transport 被替换；调用没有留下记录；伪造或改标签的模型对象；不一致的调用记录；预算超支。这些情况会被拒绝、记录为 `CLIENT_UNTRUSTED` / `BUDGET_EXCEEDED`，或令该行以 `MODEL_PROVENANCE_INVALID` / `BUDGET_OVERRUN` 判为 BLOCKED。

不覆盖（需要签名产物或进程隔离）：同一进程内代码的故意篡改（例如围绕真实客户端重建注册类、修改防线状态）；在受保护调用开始前就已存在的线程、事件循环或连接上的网络 I/O（例如替身复用一个已用过的真实客户端连接池里的 keep-alive 连接）；不触发审计事件的原生代码，包括 macOS 上 `multiprocessing` 默认 spawn 方式启动的子进程；对工厂创建的真实客户端的方法做 monkeypatch（测试套件把它当作测试接缝，这类记录会被标为真实调用）。因此 Shadow 产物是进程内代码所记录内容的研究证据，而不是防篡改的证明。

### 确定性权威

最终 Shadow 分类由代码决定，而不是模型文本决定。

可能结果：

```text
SHADOW_ACCEPT
SHADOW_REJECT
BLOCKED
```

模型不能把研究意见直接转换为可执行指令。

## Point-in-Time 边界

Agent 层只能接收显式 allowlist 内、且位于 decision cutoff 之前的公共特征。

例如：

- BTC price / return / realized-volatility；
- BTC trade-count / volume / signed-flow；
- Polymarket mid / spread / depth / OBI；
- 短周期 book-update；
- 显式 observation timestamps；
- measured coverage fields。

未知字段或禁止字段直接 fail closed。Observed timestamp 不得晚于 decision cutoff。

## Coverage Gates

当前研究边界保持原有 coverage thresholds，不降低：

```text
BTC_PRE30  >= 0.99
BOOK_PRE10 >= 0.99
```

通过 coverage 或 numeric-sanity 规则**不代表**：

- 已进入 formal cohort；
- 已具备正式策略资格；
- lineage 已完整闭合；
- 已证明盈利；
- 已获 formal backtest 批准；
- 已具备 LIVE readiness。

## Publication / Provenance

Publication / Provenance v2 已作为 versioned publication layer 合并到 `main`。

它用于保留：

- behavioral truth；
- reconciliation evidence；
- provenance membership；
- artifact identity；
- hash-bound auditability。

必须区分：

```text
PUBLICATION / PROVENANCE CODE MERGED
                ≠
ACTUAL PUBLISH EXECUTED
```

当前仓库状态**不声称 actual publish 已执行**。

## 安全边界

LLM / Agent 层不在毫秒级执行路径中。

Agent 层不能：

- 读取 private keys 或 LIVE credentials；
- 提交真实 venue orders；
- 绕过 deterministic risk controls；
- 修改 frozen research semantics；
- 原地覆盖 pinned feature / provenance artifacts；
- 写入 formal cohort；
- 授权 formal backtest；
- 执行 actual publish；
- 授权 Production Eligibility。

当 identity、point-in-time state、evidence、provenance、coverage、schema 或 audit hash 不一致时，系统按 **fail closed** 处理。

## 当前状态

| 层 | 状态 |
| --- | :---: |
| Sanity Audit v2 | FROZEN |
| 243-conflict forensics | CLOSED / PASS |
| Publication / Provenance v2 code | 已合并到 `main` |
| Actual publish | NOT RUN |
| K3 / Jev agent research shadow | 已合并到 `main` |
| Post-merge GitHub Actions CI | PASS |
| Formal cohort | EMPTY |
| Formal backtest | NOT RUN |
| Strategy execution shadow | NOT STARTED |
| Real order submission | DISABLED |
| **LIVE execution** | **STOP** |

PR #28 合并后的已验证代码基线：

```text
main = 1cbca74943225ec40fd46bfafcd77888b2a8150f
```

K3 / Jev feature commit：

```text
e0095ed8de17708cc66fd21c287d1c4ce4f51d22
```

## 仓库结构

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

本仓库用于：

- quantitative research；
- point-in-time feature analysis；
- agent-assisted research；
- execution validation；
- provenance / publication governance；
- production-readiness controls。

本仓库**不应被解释为**：

- AI 自动盈利机器人；
- 盈利能力证明；
- 已完成 formal backtest；
- 已部署 LIVE trading system；
- 已完成 production trading authorization。

## Governance Model

仓库遵循显式分阶段工作流：

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

一个阶段的授权不会自动延伸到下一个阶段。

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

绝不：

```text
Model Opinion → Production
```

---

<div align="center">

**K3 REASONS · JEV JUDGES · DETERMINISTIC CODE DECIDES · LIVE STAYS GATED**

</div>
