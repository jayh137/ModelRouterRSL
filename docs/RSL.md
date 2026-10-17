# RSL 第一版运行说明

RSL 以已有 ModelRecommender 为基础，通过执行记录和显式反馈更新任务类别 × 模型的历史表现。它不会微调候选模型。训练只创建候选版本；独立评测、激活和回滚分别执行。

## 开关与部署

- 全局默认关闭：`ROUTER_RSL_ENABLED=false`。启用后重启服务，独立数据库由 `ROUTER_RSL_DB` 指定，默认 `rsl.db`。
- 每个 routing profile 需要显式配置 `recommendation_enabled=true`、`rsl_enabled=true` 和 `rsl_mode=collect|active`。`collect` 采集、保留原有推荐结果；`active` 应用已激活的版本。
- `recommendation_mode` 应为 `auto_top1`；shadow/off/preview 模式不能用来验证实际生效的学习路由。
- 管理 API 需要 `ROUTER_ENABLE_ADMIN_API=true`，统一使用 `Authorization: Bearer <ADMIN_API_KEY>`。仅在本机或受保护的内部网络开放。
- RSL 不扩大 profile 的模型权限；候选仍受 allowed/deny、能力、禁用与不可用检查限制。
- 无定时训练、自动探索或自动发布。CLI 是本机管理入口，依赖本机文件权限；HTTP 管理接口需要管理密钥。

建议首次验证时关闭该 profile 的关键词、语义和 cascade 路由，并使用 `model=auto` 的普通请求。独立评测重放推荐器的选择；它不执行供应商请求，也不能证明额外的 ensemble/cascade 链路已经通过端到端验证。

## 可复制的管理流程

以下 PowerShell 示例在服务运行、真实 provider/model 已配置、全局 RSL 已开启后执行。将模型占位符替换为该服务已配置的 routing key。管理密钥来自环境变量，不写进脚本。

### 1. 创建采集 profile

```powershell
$base = "http://127.0.0.1:8000"
$adminHeaders = @{ Authorization = "Bearer $env:ADMIN_API_KEY" }
$profileConfig = @{
    allowed_models = @("YOUR_ROUTING_KEY_A", "YOUR_ROUTING_KEY_B")
    recommendation_enabled = $true
    recommendation_mode = "auto_top1"
    keyword_routing_enabled = $false
    semantic_routing_enabled = $false
    cascade_enabled = $false
    rsl_enabled = $true
    rsl_mode = "collect"
}
$created = Invoke-RestMethod -Method Post -Uri "$base/admin/routing-profiles" -Headers $adminHeaders -ContentType "application/json" -Body (@{
    name = "rsl-pilot"
    description = "RSL collection pilot"
    config_json = ($profileConfig | ConvertTo-Json -Depth 10 -Compress)
} | ConvertTo-Json -Depth 12)
$profileId = [string]$created.id

function Invoke-RslPost($Path, $Payload) {
    Invoke-RestMethod -Method Post -Uri "$base/admin/rsl/$Path" -Headers $adminHeaders -ContentType "application/json" -Body ($Payload | ConvertTo-Json -Depth 30)
}
```

已有 profile 应先读取并合并其 `config_json`，不要直接覆盖原有模型限制。上游 API 的 `config_json` 是 **JSON 字符串**，不是嵌套对象。

业务请求携带内部服务密钥和 `X-Routing-Profile-Id: <profileId>`，使用 `model: "auto"`。响应头 `X-RSL-Decision-Id` 用于关联反馈，`X-RSL-Policy-Version` 表示该次请求固定使用的版本。流式请求应等待结束，再查询调用完成状态。

### 2. 查找调用并提交反馈

```powershell
$decisions = Invoke-RestMethod -Uri "$base/admin/rsl/decisions?profile_id=$profileId" -Headers $adminHeaders
$decisions | Select-Object decision_id, request_id, model, task

# 从业务保存的响应头取实际决策 ID。
$decisionId = "REPLACE_WITH_X_RSL_DECISION_ID"
$detail = Invoke-RestMethod -Uri "$base/admin/rsl/decisions/$decisionId" -Headers $adminHeaders
$detail.attempts | Select-Object attempt_id, model, kind, status, cost, latency_ms
```

把反馈关联到实际经过验证的 `kind=generation`、`status=completed` 调用。cascade/重试可能产生多个 attempt；不能只按初始推荐模型判断。失败、取消或 judge 调用不接受答案质量反馈。

下面的 `score=1.0`、`passed=true` 仅适用于你的客观测试确实全部通过的情况；应替换为实际指标。不要因为 HTTP 200 或回答较长就提交该评分。

```powershell
$attemptId = "REPLACE_WITH_VERIFIED_GENERATION_ATTEMPT_ID"
$feedback = Invoke-RslPost "feedback" @{
    attempt_id = $attemptId
    source = "objective"
    source_id = "unit-tests-v1:$attemptId"
    score = 1.0
    passed = $true
    confidence = 1.0
}
```

`source` 支持 `objective|manual|user|judge|rule`；`source_id` 应包含评估器/规则版本和事件 ID。同一个 `source/source_id` 的相同反馈幂等返回；同键不同内容会被拒绝。管理调用者负责确保来源真实可信，API 不替你验证测试是否实际运行过。

### 3. 训练候选版本

积累足够反馈后执行：

```powershell
$versionsBefore = Invoke-RestMethod -Uri "$base/admin/rsl/versions?profile_id=$profileId" -Headers $adminHeaders
$previousVersionId = $versionsBefore.active.version_id

$candidate = Invoke-RslPost "train" @{
    profile_id = $profileId
    min_samples = 5
    prior_strength = 5.0
    min_confidence = 0.8
}
$candidate.version_id
```

默认每个“任务类别 × 模型”至少需要 5 个不同请求组的可信反馈。相同内容的重复请求不用于凑够独立组数；样本不足时训练报错且保持当前版本。每次调用最多选择一条标签，优先级为客观测试、用户/人工、规则、Judge，再比较置信度和时间。训练保留已使用记录的身份与数据截止时间。

历史质量分使用原始基线作为固定先验，按反馈置信度加权，避免多轮把已有观测重复计入先验。没有反馈的模型不被标为失败。cascade 自动采集的基础规则/Judge 信号置信度为 0.4/0.6，低于默认训练门槛 0.8，不参与默认学习；不要通过随意降低门槛将弱信号变成可信标签。

### 4. 准备独立评测数据

将真实测量的验证集存放到未跟踪的 `local-data/heldout.json`。外层是 `{"dataset_id": "...", "cases": [...]}`，至少 **20 个不同消息内容的案例**。以下仅展示一个 case 的字段结构；带尖括号的值必须替换为真实结果，此模板本身不能提交：

```json
{
  "dataset_id": "business-holdout-v1",
  "cases": [
    {
      "case_id": "heldout-001",
      "messages": [
        {"role": "user", "content": "<独立验证问题>"}
      ],
      "models": {
        "YOUR_ROUTING_KEY_A": {
          "score": "<实测数值，0到1>",
          "cost": "<实际总费用，非负数值>",
          "latency_ms": "<实际延迟毫秒，非负数值>"
        },
        "YOUR_ROUTING_KEY_B": {
          "score": "<实测数值，0到1>",
          "cost": "<实际总费用，非负数值>",
          "latency_ms": "<实际延迟毫秒，非负数值>"
        }
      }
    }
  ]
}
```

每个 case 必须覆盖该 profile 和请求允许的全部已配置候选。可选 `allowed_models` 表示该条请求的更窄权限范围，不能用它隐藏本应参与比较的候选。模型 key 使用 routing key；所有成本使用同一货币和计费口径，应包含被测路径的重试和评估调用费用。缺失费用不是 0；缺失候选观测不能直接做成完整评测。

不需要填写 `group_id` 或 `task`：服务根据消息内容重算去重指纹，并使用生产分类器推导任务类别；若手动声明 task，必须与分类器一致。改变 case ID 不会绕过训练集重叠检查。已用于验证的数据保留隔离标记，后续训练继续排除这些组。

评测时冻结一份模型配置与健康/可用性快照，对当前版本和候选版本分别运行同一推荐器；路由函数不会收到模型真实分数。这里只回放选择，不发起外部模型调用。

### 5. 评测、激活并切换 active 模式

```powershell
$dataset = Get-Content -Raw -Encoding utf8 "local-data/heldout.json" | ConvertFrom-Json
$evaluation = Invoke-RslPost "evaluate" @{
    profile_id = $profileId
    version_id = $candidate.version_id
    dataset_id = $dataset.dataset_id
    cases = @($dataset.cases)
}
$evaluation.report.metrics
if (-not $evaluation.passed) { throw "独立评测未通过，保留当前版本。" }

$activated = Invoke-RslPost "activate" @{
    profile_id = $profileId
    version_id = $candidate.version_id
    evaluation_id = $evaluation.evaluation_id
}

$profileConfig.rsl_mode = "active"
Invoke-RestMethod -Method Put -Uri "$base/admin/routing-profiles/$profileId" -Headers $adminHeaders -ContentType "application/json" -Body (@{
    config_json = ($profileConfig | ConvertTo-Json -Depth 10 -Compress)
} | ConvertTo-Json)
```

HTTP 和 CLI 使用固定发布门槛，不接受调用者传入 `passed` 或放宽评测阈值：

| 发布门槛 | 默认要求 |
| --- | --- |
| 数据完整性 | 至少 20 个案例，无训练重叠、无重复消息组、完整候选观测 |
| 平均质量 | 候选不低于当前版本 |
| 每类任务质量 | 任一任务类别均不下降 |
| 总费用 | 候选不高于当前版本 |
| p95 延迟 | 候选不超过当前版本的 1.10 倍 |

不满足质量/成本/延迟门槛会保存失败报告，激活被拒绝；数据不完整或有泄漏则直接报错。激活时再次核对版本、评测哈希、当前基线和路由配置；评测后若配置或当前版本变化，必须重新评测。切换 `collect` 到 `active` 是允许的发布步骤。

最低 20 题是程序验收门槛，不代表统计上充分；真实发布应使用代表性任务和足够样本，进一步观察线上质量与成本。

### 6. 回滚

```powershell
Invoke-RslPost "rollback" @{
    profile_id = $profileId
    version_id = $previousVersionId
}
Invoke-RestMethod -Uri "$base/admin/rsl/versions?profile_id=$profileId" -Headers $adminHeaders
```

目标必须是同一 profile 曾经激活的版本；初始基线也可作为回滚目标。不能用 rollback 绕过评测直接发布未激活的候选。激活指针在事务中更新；已开始的请求继续使用其固定快照，新请求读取新版本。数据和审计历史不会被删除。

## 本机 CLI

从仓库根目录运行，使用与服务相同的数据库路径。CLI 的路径参数显式覆盖环境变量；不要假定它自动读取 `.env`。以下占位 ID 替换为上一步输出：

```bash
python -m app.rsl.worker --db rsl.db versions --profile-id 1
python -m app.rsl.worker --db rsl.db feedback --attempt-id ATTEMPT_ID --source objective --source-id tests-v1:ATTEMPT_ID --score 1 --passed --confidence 1
python -m app.rsl.worker --db rsl.db train --profile-id 1
python -m app.rsl.worker --db rsl.db evaluate --profile-id 1 --version-id CANDIDATE_ID --router-db router.db --dataset local-data/heldout.json
python -m app.rsl.worker --db rsl.db activate --profile-id 1 --version-id CANDIDATE_ID --evaluation-id EVALUATION_ID --router-db router.db
python -m app.rsl.worker --db rsl.db rollback --profile-id 1 --version-id PREVIOUSLY_ACTIVE_ID
```

`--db` 是顶层参数，必须放在子命令前；`--router-db` 用于评测/激活时加载真实路由配置。CLI feedback 的评分同样必须基于实际验证结果。命令本身不自动切换 profile 的 `rsl_mode`。

## 管理接口速查

所有路径均以 `/admin/rsl` 为前缀。

| 方法与路径 | 用途 |
| --- | --- |
| `GET /decisions?profile_id=...&limit=100` | 查询 profile 的决策 |
| `GET /decisions/{decision_id}` | 查询单次决策、所有 attempts、费用完整性 |
| `POST /feedback` | 提交显式反馈 |
| `POST /train` | 创建候选历史版本 |
| `POST /evaluate` | 重放独立矩阵并保存报告 |
| `GET /evaluations/{evaluation_id}` | 查询报告及是否通过 |
| `POST /activate` | 激活已通过评测的候选 |
| `POST /rollback` | 恢复历史激活版本 |
| `GET /versions?profile_id=...` | 查询 active、版本和激活审计记录 |

`profile_id` 在 RSL API 中是字符串；建议始终使用真实 routing profile 数字 ID 的字符串形式。默认 `__global__` 仅为兼容入口，不替代业务 profile 设置。RSL 关闭时接口返回 404；schema/完整性不符合时返回 422。完整请求 schema 见服务 `/docs`。

## 隐私、统计与验收

RSL 持久化消息内容指纹、长度等特征、候选、版本与调用元数据，不保存 prompt/回答原文。内容指纹用于去重，不是匿名化保证。attempt 记录区分 generation 和 judge；重试、cascade 和流式完成/失败/取消需要分别查看。费用不完整时 `total_cost=null`，`known_cost` 只是已知部分，不能当作全部成本。上游日志及外部评估器各自的数据保留行为仍需按部署环境检查。

将真实评测文件放在未跟踪的 `local-data/`，不要提交密钥、原文或数据库。备份时保存路由配置与 RSL 数据库，恢复后核对 profile、模型 ID 及版本状态。跨多个实例时应使用同一受支持的本地数据库部署方式；本版没有分布式训练协调或后台调度服务。

```bash
python -m unittest discover -s tests -v
python -m app.rsl.worker demo
```

demo 在临时目录中创建数据库，以纯合成数据完成采集、训练、独立评测、激活和回滚，退出时清理；不修改生产数据库、不调用真实模型、不声明真实质量提升。
