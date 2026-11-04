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

当前注册模型为 K3 `gpt-6-astra`（7 次研究调用）和 Jev `gpt-6.1-sol`（1 次判断），均使用 OpenAI Responses、`high` effort、严格 JSON 和 `store=False`，不回退到其他模型。`build_shadow_models(k3_client, jev_client, ...)` 现在接收两个 OpenAI 客户端，旧 Anthropic K3 transport 已移除。两个 transport 升至 v5，产物 schema 仍为 v4。[官方模型规格](https://developers.openai.com/api/docs/models/gpt-6-astra)列出的 Astra 标准输入／输出价为每百万 token $10／$50。缓存读取按普通输入价保守计入，缓存写入按 2.5 倍计入，预留也覆盖该写入倍率。缓存计数缺失或异常时保留合法的原始 token 数，至少按全额预留结算；较高的已知用量仍会保守计入并锁定预算。中转实际账单不能据此保证。

回放 worker v3 在第一次调用前保存 `replay_binding.json`，绑定双方模型、运行限额、价格、通道、测试／真实来源、提示词版本和脱敏设置。续跑允许调整预算和行数；配置不同，或旧产物缺少绑定文件时，需要新输出目录，包括旧 Fable 回放。SDK 客户端在正常完成或初始化失败后都会关闭。

`python scripts/summarize_agent_shadow_replay.py --input-dir /tmp/shadow-replay` 只读 v3 回放目录并向 stdout 输出 JSON 汇总。工具校验 artifact 哈希、schema 和绑定一致性，分别统计所有尝试、每行最后状态及已终结结果；最终接受率以所有已终结行为分母，包含最终 BLOCKED 行，排除未终结及产物中不存在的行。重试费用累计计入，已定价记录（含预留估算）的费用以十进制字符串求和，null 费用保留为未定价记录计数。未终结行可能已耗尽重试次数。输出仅含聚合数据，不含行标识、request id 或模型文本；读取范围为绑定和 artifacts，不调用模型、不写文件。哈希检查表示完整性，不证明原始证据真实性或中转账单。

`python scripts/run_agent_shadow_replay.py --input-dir /tmp/frozen-features --plan-only --out-dir /tmp/shadow-replay --route relay_mtu --max-rows 5 --budget-usd 50` 还会输出已校验的续跑规划，区分已终结、新行、可重试和次数耗尽的行；保持实际 worker 的输入顺序，按 `--max-rows` 选择本轮候选行。规划读取冻结输入、绑定／artifacts 及路径配置，跳过 dotenv 文件并保持进程环境不变；不创建输出目录或客户端，输出不含行标识、真实时间、模型文本或凭证。非目录的输出位置或父级会被拒绝。预算按新一轮运行计算，历史费用不从中扣减。配置预留估算复用实际预留规则，按各模型的最大输入和输出限额计算（目前每个可能调用模型的行约 $41.76），是注册价格估算，不能预测实际用量或验证中转账单；启动预算门槛单独显示，规划不能预测运行中途的停止。`--max-attempts`（默认 3）与 `--row-headroom-usd`（默认 $1.50）同时用于规划和实际运行，并写入 run manifest。真实中转回放仍需 `--allow-unverified-relay`。

报告用量对应的费用超出有限数值表示范围时，释放并收取预留金额，完整费用记为未知（`cost_usd=null`、`cost_not_representable`），同时锁定预算。`BUDGET_OVERRUN` 阻断该行及后续调用。

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
- 供应商通道：`official`（Anthropic / OpenAI 官方 API）与 `relay_mtu`（第三方中转站 `https://mtu.mtuopenai.xyz`，密钥 `STD0_RELAY_API_KEY`）。每条通道的地址和密钥变量都写死在代码里；只要传入的环境或进程环境中存在官方凭证（`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`OPENAI_ADMIN_KEY`、`OPENAI_WEBHOOK_SECRET`、`ANTHROPIC_AUTH_TOKEN`），就拒绝构建中转客户端。transport 在每次调用时执行通道策略：中转调用必须处于显式开启的范围内（`ShadowRunner(..., allow_unverified_relay=True)` 或 `route_policy(allow_unverified_relay=True)`），否则被拒绝；同一行的所有调用必须使用同一条通道，无论模型外面包了几层（在发起调用的线程上检查；嵌套的 `route_policy` 作用域只能收紧外层）。`ShadowRunner` 还会在任何调用之前拦截：注册模型走中转却未开启（`RELAY_NOT_ALLOWED`）、两边通道不同或中转模型旁边是未注册模型（`MODEL_ROUTE_MIXED`）、测试替身与真实客户端混用（`TEST_CLIENT_MIXED`）。产物带 `route` 与 `provenance`（`OFFICIAL`、`UNVERIFIED_RELAY`、仅当两边都是替身时才为 `TEST_CLIENT`、`UNSPECIFIED`），schema 会校验它们与调用记录的通道一致；
- 模型来源校验（产物 schema `k3_jev_agent_shadow_v4`）：供应商标签和逐次调用记录只取自注册的 `K3Client` / `JevClient` 及其注册 transport；其他模型对象标为 `unspecified`，其自报的调用记录会被丢弃；来源不一致或预算超支时该行判为 BLOCKED；
- 离线测试替身在调用期间受运行时防线（审计钩子）约束：仅针对发起调用的线程、仅在调用期间，拒绝新建 socket、建立连接、DNS 解析、启动线程、`subprocess` / `os.system` / `os.exec*` / `os.spawn*` / `posix_spawn` / `fork` 以及 ctypes 调用（调用期间首次导入并加载原生库也会被拒绝）。

### 模型来源与离线替身的威胁边界

覆盖（误用）：测试替身本身是、明显持有、隐藏或在调用时临时创建真实 SDK 客户端，并在调用期间新建连接；把工厂创建的真实客户端换进测试 transport（或把假客户端换进正式 transport）；替换绑定或价格（包括 `ModelBinding` 子类）；在实例上覆盖注册客户端或 transport 的方法；注册客户端的 transport 被替换；调用没有留下记录；伪造或改标签的模型对象；不一致的调用记录；预算超支。这些情况会被拒绝、记录为 `CLIENT_UNTRUSTED` / `BUDGET_EXCEEDED`，或令该行以 `MODEL_PROVENANCE_INVALID` / `BUDGET_OVERRUN` 判为 BLOCKED。

不覆盖（需要签名产物或进程隔离）：同一进程内代码的故意篡改（例如围绕真实客户端重建注册类、修改防线状态）；在受保护调用开始前就已存在的线程、事件循环或连接上的网络 I/O（例如替身复用一个已用过的真实客户端连接池里的 keep-alive 连接）；不触发审计事件的原生代码，包括 macOS 上 `multiprocessing` 默认 spawn 方式启动的子进程；在通道策略作用域内新启动的线程里发起的官方调用（上下文变量不会传入这些线程，因此不会按该行的通道核对；其中的中转调用仍会被拒绝）；对工厂创建的真实客户端的方法做 monkeypatch（测试套件把它当作测试接缝，这类记录会被标为真实调用）。中转通道不在上述任何核验范围内：中转站能看到每个请求的全部内容，其后端模型、计费和数据处理都无法核实（只核对返回的模型名和输出 schema，且拿不到 request id）。注册中转模型产出的行标记为 `UNVERIFIED_RELAY`（涉及未注册或被包装模型的行为 `UNSPECIFIED`），必须与官方结果分开；在另行授权真实研究数据之前，只允许经中转站发送合成数据。总之，Shadow 产物是进程内代码所记录内容的研究证据，而不是防篡改的证明。

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
