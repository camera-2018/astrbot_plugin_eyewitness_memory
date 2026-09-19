# v0.1.4 名称迁移

新安装统一使用 `astrbot_plugin_eyewitness_memory`，不需要迁移。

旧版 `astrbot_plugin_evidence_memory` 是不同的 AstrBot 插件身份。迁移时不要同时运行两份，也不要选择删除插件数据：

1. 备份旧代码、插件配置和 SQLite 数据；先停止旧插件，再备份数据库，包含尚未 checkpoint 的 WAL 数据。
2. 通过 AstrBot 卸载旧插件代码，保留配置和数据，解除旧消息处理器注册。
3. 将 `data/plugin_data/astrbot_plugin_evidence_memory` 改名为 `data/plugin_data/astrbot_plugin_eyewitness_memory`；将 `data/config/astrbot_plugin_evidence_memory_config.json` 改成新标识对应的文件名。若目标已存在，停止并人工核对，不能覆盖。
4. SQLite 的 `settings.data.collection` 前缀从 `evidence_memory_` 改为 `eyewitness_memory_`。其他策略、记忆、原文、版本和历史注入标记保持原样。没有配置 Qdrant 也需要修改此字段。
5. 若使用 Qdrant，备份旧前缀下属于本插件的 collection 配置和 points；创建新前缀 collection，复制向量、point ID、payload 与索引。核对数量和内容后才能清理旧 collection。不要修改其他插件索引。不必重新调用 Embedding 模型。
6. 将新版安装到 `data/plugins/astrbot_plugin_eyewitness_memory`，检查只注册一份插件、记忆与原文数量未减少、群范围和模型配置不变。
7. 管理页面改为 `/#/plugin-page/astrbot_plugin_eyewitness_memory/console`，旧书签需更新。

上述步骤需要管理员执行，不会在插件导入时擅自移动另一个插件目录或修改外部数据库。迁移失败时，保持插件停用，依据备份恢复旧目录、配置和索引，再加载旧代码；不要用空数据库覆盖原数据。
