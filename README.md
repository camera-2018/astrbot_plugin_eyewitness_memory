# Eyewitness Memory · 群聊记忆

仓库名：`astrbot_plugin_eyewitness_memory`。目前为兼容已有安装，AstrBot 内部插件标识仍为 `astrbot_plugin_evidence_memory`，管理页面与数据目录沿用该标识；仓库改名不会迁移或清空记忆。

自动提取群聊记忆、按群检索并核对原文的 AstrBot 插件。管理页面使用 React + TypeScript + shadcn/ui，参考 Repeat 插件的深色管理界面，直接复用 AstrBot 登录。

> v0.1.3。仅保留启用／停用开关。新安装默认停用、群白名单为空；启用后，将核验结果和带时间、发言者的局部原文追加到本轮用户消息。当前支持 QQ / aiocqhttp。

## 能做什么

- 从白名单群的真实消息中后台提取偏好、目标、事件、约定；保留发言者稳定 ID、时间及来源引用。
- 跳过配置的机器人账号、低信息复读，拒绝无效来源；转述或主体不一致的候选进入待审核。
- 回复前自动检索、相关性判断、原文核验；不需要主 Agent 主动调用 LLM 工具。
- 无证据、无关、超时、超预算时允许零注入；正常聊天继续。
- 已注入的片段随会话保存，不回改历史、不写 system prompt；同版本仍在历史中时不重复注入。插件不把自己的召回结果再次作为原文采集。
- 可选 Qdrant 语义检索；未配置或失败时使用中文二元词/英文词 FTS5 检索。
- AstrBot 插件页面：浏览/搜索、来源对照、版本、编辑/停用/删除、召回记录、试召回、配置和人物数据删除。
- 不再启动独立 HTTP 监听，不需要单独端口、Token 输入或第二套登录。

## 架构

```text
QQ 群消息 → 去重/脱敏 → SQLite 原文 → 后台提取 → 有来源的记忆
                                                     ↓
                                               Qdrant 可选索引

主模型请求 → 门控 → 群内检索 → 相关性审核 → 原文核验 → 追加原文片段
                                      ↓
                                可查询的召回记录

React + shadcn 页面 ← AstrBot Pages Bridge / AstrBot 登录 → SQLite / 记忆引擎
```

SQLite 是权威存储；Qdrant 不存原文和摘要，只存向量、作用域和版本。查询结果必须经过 SQLite 校验。索引更新由持久化 outbox 驱动，失败退避重试。

## 安装前准备

1. AstrBot **4.28.1+、Python 3.11+**；辅助 Chat Provider 已配置。
2. 仓库地址：[camera-2018/astrbot_plugin_eyewitness_memory](https://github.com/camera-2018/astrbot_plugin_eyewitness_memory)。手动安装时，在 AstrBot 根目录运行 `git clone https://github.com/camera-2018/astrbot_plugin_eyewitness_memory.git data/plugins/astrbot_plugin_evidence_memory`，沿用内部插件目录名；已有安装不要再克隆第二份。
3. 仓库包含 `pages/console/` 构建产物，安装到机器人时不需要 Node.js。修改前端后重新构建并提交产物。
4. 依赖由 AstrBot 根据 `requirements.txt` 安装。
5. 在 AstrBot 中打开本插件的管理页面。

### 管理页面与旧版迁移

从 AstrBot 的插件页面进入，或访问 `/#/plugin-page/astrbot_plugin_evidence_memory/console`。权限由 AstrBot 管理，不是匿名公开管理 API。

旧版 SQLite 和策略自动沿用，插件标识及数据目录不变。`panel_host`、`panel_port`、`panel_public_url` 和旧 `panel.token` 不再使用。已有 Docker 端口映射可在下次维护时移除，不必为页面更新重建容器。

v0.1.2 移除了影子模式；旧配置中的 `shadow` 自动迁移为停用，避免升级意外开启注入。原来已启用或停用的配置不变，历史试运行记录保留。

v0.1.3 自动升级 SQLite 为 schema 2，保留已有消息与记忆。原代码不支持直接打开新版库，回滚前需同步恢复备份。旧消息没有准确发送时间，显示「采集时间（发送时间未知）」；新消息优先使用 OneBot 原始事件的 `time`，缺失/非法则仍标为采集时间，不使用适配器的接收时间冒充发送时间。

### 原文片段与缓存

- 每条直接来源前后各取最多 10 条同群消息，补充已采集的引用父消息；按发送时间（缺失则采集时间）排序，同秒按采集先后、ID 稳定排序。多个来源去重后最多 50 条、18,000 字符，直接来源优先保留。
- 面板显示局部上下文、直接来源标记、逐句昵称/用户 ID/正文/时间，时间统一带 `+08:00`。仅覆盖本插件已采集、仍保留的消息；不是完整聊天历史，可能跨越时间间隔。
- 辅助模型核验这段原文；通过后主模型获得核验摘要与有界原文片段，而非仅摘要。总注入默认 8,000 字符（可配 200–16,000），过长时优先保留直接来源和核验证据；连这些都放不下则不注入。邻近插话不等于已核实事实。
- 新片段只追加到 `extra_user_content_parts`，不修改原 prompt、system prompt 或既有历史；不再使用 `mark_as_temp()`。插件保持已注入历史的前缀，不能保证提供方缓存命中率，其他插件、会话压缩/reset 仍可能改变请求。
- 已注入片段是不可变历史快照；后续新版本追加，不回写旧片段。删除插件记忆不会擦除 AstrBot 的既有会话历史；彻底清除需另外处理对应会话及备份。

### 首次配置

1. 打开页面「设置」。
2. 选择已配置的辅助模型 Provider ID；支持 JSON 输出、遵守约束的模型更合适。
3. 白名单填完整作用域，例如 `default:GroupMessage:123456`，一行一个。不是群名，也不是单独群号。
4. 填写其他机器人的用户 ID，避免把机器人输出提取成人物事实。
5. 从一个群开始，确认模型、费用预算与群范围后将「插件开关」设为启用。
6. 可选配置 Embedding Provider 与 Qdrant URL。Collection 前缀必须以 `evidence_memory_` 开头；不能使用其他插件的 collection。
7. 在「召回记录」和「召回测试」中检查来源与核验结果；需要时可停用插件。

记忆策略保存在插件自己的 SQLite；AstrBot 插件配置仅保存可选 Qdrant Key。模型凭据仍由 AstrBot 管理。

## 运行预算与边界

- 默认每 40 条或最旧未处理消息等待 15 分钟，后台顺序处理一批；后台每 30 秒检查一次。单批最多 8 条候选。
- 在线最多 12 条候选、2 条核验结果；最多 **1 次相关性审核 + 2 次来源核验**。不是无限 Agent 循环。
- 默认在线总预算 3 秒；这是可配置上限，慢模型可能频繁超时而零注入。可根据召回记录的耗时调整。
- 启用后在主回复前等待核验，失败或超时则继续正常聊天。停用后不采集、不提取、不自动召回。
- 后台提取默认超时 60 秒，可单独调整到最多 180 秒；它与在线召回预算分离。
- 默认每日最多 200 次辅助生成调用，按 UTC 日切换；已发起的失败调用也计数。Embedding 调用不计入这个计数，其用量仍需在提供方观察。
- 注入上限按字符计算，不假装精确 token 数。
- 原文默认保留 30 天；仍被记忆引用的证据保留。Trace 默认保留 7 天且最多 2000 条。
- SQLite + WAL 达到配置保护线时停止新增采集；这不是整个磁盘的绝对硬配额，不覆盖 Qdrant、备份和其他插件。
- 不下载媒体；没有图像内容时不会凭 `[Image]` 构建事实。
- 不支持跨群共享、私聊记忆、图谱、自动覆盖业务状态、历史全量导入。定时任务/后台唤醒不在首版声明支持范围内。
- 模型仍可能误判玩笑或真实性。程序验证来源存在、引用匹配、作用域和版本；**逐字引用不等于语义一定正确**。应定期在面板检查记忆和原文。

改换 Qdrant 地址或 collection 前缀后会重新排队索引；旧服务/旧前缀中的向量不会被自动清空，避免误删数据。需要管理员单独确认清理。当前仍可访问的同前缀模型代际，删除记忆时会同步排队删除对应向量。

如果在 AstrBot 中修改了同一个 Embedding Provider ID 对应的模型，保存后在面板点击「重建向量索引」。修改 Provider ID 会自动排队；仅修改同 ID 底层模型需要手动重建，以免后台不断发起模型探测请求。

## 本地开发与验证

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
cd frontend
npm ci
npm run build
npx playwright install chromium
npm run test:e2e
```

独立演示，**只有合成数据、不连接生产环境、不调用真实模型**：

```bash
PYTHONPATH=. uv run python scripts/demo.py
```

该命令提供本地测试服务器，浏览器 E2E 注入模拟 AstrBot Bridge 后访问。直接打开不会伪造生产登录；实际页面应由 AstrBot 加载。演示退出后数据库销毁；固定演示令牌只保护测试 REST 服务，生产插件不使用它。

真实 AstrBot SDK 集成检查：

```bash
# 在一个已经安装 AstrBot 的隔离 Python 环境运行
PYTHONPATH=. python scripts/check_astrbot.py
```

检查插件导入、启动、真实请求对象尾部追加、跨轮历史保留、system/history 不变、平台时间采集、关闭和重载，不访问你的机器人。

## 仓库布局

```text
main.py                 AstrBot 适配和生命周期
evidence/               记忆引擎、SQLite、Qdrant、管理 API
frontend/               React + shadcn 源码、锁文件、浏览器测试
pages/console/          已构建的 AstrBot 插件管理页面
tests/                  隔离后端测试
scripts/                合成演示与 AstrBot 集成检查
docs/                   任务拆解、接口、安全、验证记录
.github/workflows/      CI
```

第三方 UI 组件由 shadcn CLI 生成，保留在前端源码中；见 `THIRD_PARTY_NOTICES.md`。本项目采用 MIT 许可证。
