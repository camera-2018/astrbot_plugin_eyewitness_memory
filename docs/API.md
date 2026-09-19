# AstrBot 插件管理 API

生产入口由 AstrBot 注册为 `/<plugin_name>/api`，只接受 POST，由 AstrBot 登录及 Pages Bridge 授权。前端调用 `bridge.apiPost("api", {path, method, body})`，不接收用户单独输入的令牌。处理器只分派列出的固定操作，不转发任意 URL。封装 JSON 校验上限 64 KiB（AstrBot 在此之前已读取请求体）。

下表保留便于开发测试的 REST 表示；Bridge 的 `path` 去掉 `/api/` 前缀，如 `memories?scope=...`，`method` 为表中方法。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/api/overview` | 计数、模式、后台状态、群作用域 |
| GET | `/api/providers` | AstrBot Chat / Embedding ID，不包含凭据 |
| GET / PUT | `/api/settings` | 读取/完整替换策略配置（强类型校验） |
| GET | `/api/memories?scope=&status=&q=&offset=` | 每页 30 条，总数 |
| GET | `/api/memories/{uuid}` | 摘要、直接来源 `sources`、局部原文 `context`、最近 20 个版本 |
| PUT | `/api/memories/{uuid}` | `{version,summary,status}` 乐观并发修改 |
| DELETE | `/api/memories/{uuid}` | 永久删除摘要、版本、引用；原文按保留策略处理 |
| GET | `/api/traces?scope=` | 最近 100 条召回记录 |
| POST | `/api/preview` | `{scope,query,sender_id?}` 试召回，不发群消息，可能调用模型 |
| POST | `/api/erase-subject` | `{scope,subject_id,confirm:"DELETE"}` 删除该群人物数据 |
| POST | `/api/reindex` | `{}` 将全部记忆排队重建索引，产生 Embedding 调用 |

鉴权失败由 AstrBot 返回 401/403；无效输入/版本冲突 400，不存在 404，未预期后端错误 503。错误响应不返回堆栈或提供方凭据。

管理页对拥有 AstrBot 相应权限的管理员开放，不面向普通群友。群隔离用于自动采集/检索；管理员可以跨群管理。

`context` 每条包括 `id, sender_id, sender_name, text, reply_id, created, sent_at, time, time_kind, is_source, quote`。`created` 始终是采集时间，`sent_at` 为可空的真实平台时间；`time` 是带 `+08:00` 的可读时间，`time_kind` 明确时间来源。每个来源前后各 10 条，合并最多 50 条/18,000 字符；数据可能因采集范围、保留期限和预算不完整。PUT 记忆返回同样的详情。

召回结果的 `selected[].context` 是核验时看到的片段快照，`injected_message_ids` 是其中真正放入本轮请求的消息 ID；`injection` 是最终追加文本。原文可见不等于整段均获事实背书。
