# yuhanbolh 代码生成 Skill

本目录提供可下载分发的 `yuhanbolh-codegen`，帮助 AI 根据真实 yuhanbolh API 生成 Python 调用代码。参考基线为 0.6.5，使用时优先核对目标机器版本。

## 下载与安装

下载同目录的 [yuhanbolh-codegen.zip](yuhanbolh-codegen.zip)，解压后保留完整 `yuhanbolh-codegen` 文件夹，确保其根目录直接包含 SKILL.md；不要只复制 SKILL.md。

对于使用本地 Skills 目录的 Codex，把该文件夹复制到 `$CODEX_HOME/skills/`；未设置 CODEX_HOME 时使用用户目录下 `.codex/skills/`。已有同名 Skill 时先比较或备份，避免覆盖自定义内容。复制后在新会话中检查 Skill 是否已列出。其他支持 SKILL.md 的工具按其自身安装方式导入。

Skill 自身不需要安装 yuhanbolh 即可阅读参考；实际运行生成代码时需 Python >=3.11 和相应库依赖。它不会自动安装依赖、配置账户或运行交易。

## 调用示例

```text
使用 $yuhanbolh-codegen，读取本地 CSV 的日线数据，调用 MA 和 RSI，按日期对齐结果并保存到新的 CSV，添加中文注释。
```

```text
使用 $yuhanbolh-codegen，生成问财查询沪深300成分股的代码，区分查询失败与成功空表，不写数据库。
```

```text
使用 $yuhanbolh-codegen，生成通过标准大QMT桥接读取行情和持仓的 Python 脚本，地址和令牌从环境变量传入。
```

## 内容与维护

- `yuhanbolh-codegen/SKILL.md`：入口、场景选择和生成流程。
- `references/`：按模块拆分的 API 快照、示例和交易约束。
- `scripts/inspect_api.py`：AST 只读接口查询，不导入交易模块或执行其代码。
- `agents/openai.yaml`：Codex 界面信息，保留默认自动发现能力。

公开 API 变化后重新核对受影响模块参考及示例，再更新本机副本和 ZIP。ZIP 根目录包含单个 yuhanbolh-codegen 文件夹，不包含仓库源码、个人配置或项目记忆。SHA-256 见同目录 `yuhanbolh-codegen.zip.sha256`。
