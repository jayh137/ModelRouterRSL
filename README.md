# ModelRouterRSL

项目仓库：[jayh137/ModelRouterRSL](https://github.com/jayh137/ModelRouterRSL)（私有，需要具有访问权限的 GitHub 账号）。

在 [NewYearEve-Godsheep/ModelRouterCourse](https://github.com/NewYearEve-Godsheep/ModelRouterCourse) 基础上实现第一版 RSL（Recursive Self-Learning）模型路由闭环：**反馈采集 → 历史表现更新 → 独立评测 → 显式激活 → 版本回滚**。

本项目保留上游 OpenAI 兼容接口、多供应商、routing profile、关键词/语义路由、cascade、ensemble 与模型可用性检查。RSL 更新 Router 的历史表现信号，不微调候选大模型。它默认关闭，不会在服务启动时生成质量数据或自动发布策略。

## 第一版范围

| 能力 | 行为 |
| --- | --- |
| 反馈采集 | 记录请求决策、候选和实际调用；通过受保护的管理接口提交有来源的显式质量反馈 |
| 历史表现 | 按 routing profile 隔离，按任务类别与模型聚合，保留先验与样本量信息 |
| 独立评测 | 使用与训练样本分离的完整候选结果矩阵，对比当前与候选策略 |
| 版本管理 | 训练产物不可变，评测通过后显式激活，支持回滚和恢复基线 |

HTTP 调用成功、回答长度或缺失反馈都不等于质量标签。合成 demo 仅用于验收代码流程，不能说明真实模型质量或成本已经改善。第一版不包含定时自训练、自动探索或自动上线。

## 快速启动

需要 **Python 3.11+** 和 Git。先使用有仓库访问权限的 GitHub 账号完成 Git 认证，再克隆；后续命令从仓库根目录执行。

```powershell
git clone https://github.com/jayh137/ModelRouterRSL.git
cd ModelRouterRSL
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env

# 两个独立的本机密钥，当前终端有效。跨终端使用时写入未跟踪的 .env。
$env:ROUTER_INTERNAL_API_KEY = (& .\.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(32))")
$env:ADMIN_API_KEY = (& .\.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(32))")
.\.venv\Scripts\python -m app.main
```

Linux/macOS 使用 `python3 -m venv .venv` 和 `.venv/bin/python`，将 `Copy-Item` 换为 `cp`，通过 `export` 设置相同环境变量。服务默认只监听 `127.0.0.1:8000`。

- 健康检查：`GET http://127.0.0.1:8000/health`
- 接口文档：`http://127.0.0.1:8000/docs`
- 默认 `config/providers.yaml` 为 `providers: {}`。服务可启动，但未配置真实模型时无法处理模型请求。
- 配置结构见 [config/providers.example.yaml](config/providers.example.yaml)。替换真实供应商地址与模型 ID，密钥使用环境变量，不要提交密钥。
- 初次启动会将 providers 配置写入 `router.db`。之后使用管理 API 修改配置；已有数据库不会被 YAML 文件自动覆盖。

内部应用调用 `POST /v1/chat/completions` 时传入 `Authorization: Bearer <ROUTER_INTERNAL_API_KEY>`、`X-User-Id`，并可附带 `X-Routing-Profile-Id`、`X-Request-Id` 与 `X-Allowed-Models`。管理端使用独立的 `Authorization: Bearer <ADMIN_API_KEY>`；仅在本机或受保护的内部服务中开放。

## 启用 RSL

1. 配置真实 provider/model 与 routing profile，并先验证原有路由能正常处理请求。
2. 设置 `ROUTER_RSL_ENABLED=true`，重启服务。独立 SQLite 数据库由 `ROUTER_RSL_DB` 指定，默认 `rsl.db`。
3. 在该 routing profile 的 `config_json` 中设置以下字段，先进入采集模式：

```json
{
  "recommendation_enabled": true,
  "rsl_enabled": true,
  "rsl_mode": "collect"
}
```

4. 采集请求并提交可信质量反馈，训练候选版本，用独立评测集评估。
5. 评测通过后显式激活版本，将该 profile 的 `rsl_mode` 改为 `active` 才应用学习结果。需要停止使用时切回 `collect` 或禁用该 profile 的 RSL。
6. 观察实际表现，必要时调用回滚接口。禁用 RSL 不会删除既有数据。

RSL 的 `active` 模式仍受原有 `allowed_models`、`deny_models`、能力要求和模型不可用检查限制。全局开关、profile 开关、学习版本与推荐模式必须配合设置，具体流程与输入格式见 [docs/RSL.md](docs/RSL.md)。

## 本地验收

激活虚拟环境后执行，或将 `python` 替换为上述虚拟环境解释器路径：

```bash
python -m unittest discover -s tests -v
python -m app.rsl.worker demo
python -m app.rsl.worker --help
```

测试与 demo 不需要供应商账号，不调用真实模型。demo 使用临时数据库与纯合成数据，不写入生产 `rsl.db`。GitHub Actions 在 Python 3.11、3.12、3.13 上运行相同离线验收。

## 配置与数据

| 文件/变量 | 用途 |
| --- | --- |
| `config/providers.yaml` | 服务监听地址、provider/model 配置；初始为空 |
| `config/task_profiles.yaml` | 任务分类关键词与目标权重；初始全部归入 `general` |
| `config/model_traits.yaml` | 可解释的模型能力先验；不作为真实反馈 |
| `config/model_runtime_signals.yaml` | 可选的可用性统计；不作为回答质量 |
| `config/model_performance_history.yaml` | 可选基线历史数据；初始为空 |
| `ROUTER_DB` | 上游路由配置及技术日志数据库，默认 `router.db` |
| `ROUTER_RSL_DB` | RSL 决策、反馈、版本、评测及激活状态，默认 `rsl.db` |
| `ROUTER_RSL_ENABLED` | 全局 RSL 开关，默认关闭 |
| `ROUTER_ENABLE_ADMIN_API` / `ADMIN_API_KEY` | 管理 API 开关与独立管理密钥 |

基础安装不包含 `sentence-transformers`；仅在确实需要语义路由时单独安装并配置嵌入模型。RSL 不依赖该大包。

不要将 `.env`、数据库、真实请求原文或内部评测数据加入仓库。可将本机评测文件放在已忽略的 `local-data/`。RSL 默认保存路由特征与记录引用；上游技术日志与可选评估器的原有数据行为需在部署时一并检查。

## 上游与许可

上游来源：[NewYearEve-Godsheep/ModelRouterCourse](https://github.com/NewYearEve-Godsheep/ModelRouterCourse)，基于提交 [`e9fe3ef2763a3b241255315c9b6ccd3138d5dbf4`](https://github.com/NewYearEve-Godsheep/ModelRouterCourse/commit/e9fe3ef2763a3b241255315c9b6ccd3138d5dbf4)。本派生实现保留并扩展上游代码，RSL 相关新增代码位于 `app/rsl/`、`app/api/admin/rsl.py` 及相关接入点。

导入的上游快照未提供 `LICENSE` 文件。本仓库未自行给整个项目附加开源许可证，也不据此声明对上游代码拥有再许可权；使用、分发或商业部署请确认上游授权条件。
