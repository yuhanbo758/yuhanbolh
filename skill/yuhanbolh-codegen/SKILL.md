---
name: yuhanbolh-codegen
description: 基于 yuhanbolh Python 库生成、修改和排查调用代码，适用于行情与问财选股、可转债、技术指标、SQLite 数据处理、QMT 桥接及 xtquant/MT5 辅助程序。用户提到 yuhanbolh 或希望复用该库时使用；不负责库的提交、发布或实际交易执行。
---

# yuhanbolh 代码生成

将需求落实为使用真实公开 API 的 Python 代码，提供中文注释、输入约定、运行方式和验证结果。先用一句话明确输入、输出、运行环境和预期副作用；明确的需求直接实施，仅询问影响正确性的缺失信息。

## 核对版本和接口

参考快照来自 yuhanbolh 0.6.5（2026-09-06），要求 Python >=3.11。目标机器版本优先于快照；不要为匹配快照自动升级环境。

1. 根据 [API 模块索引](references/api-index.md) 只读取相关模块参考。
2. 有源码或已安装包时，用自带脚本核对签名与 docstring，不导入交易模块，不执行其顶层代码：

   ```powershell
   # 替换实际 Skill 目录；--repo 为可选的源码仓库路径。
   python "<Skill目录>/scripts/inspect_api.py" --repo "<仓库目录>" --name MA
   python "<Skill目录>/scripts/inspect_api.py" --name BridgeClient.order
   ```

3. 生成前核对选中函数实现中的输入列、返回结构、默认路径、异常处理与副作用。脚本输出源码位置；无目标源码时以附带参考为基线并说明版本尚未核对。未说明的契约不可猜测，必要时请求相应实现或数据样本。
4. 使用 `from yuhanbolh import <公开名称>`，避免星号导入触发所有延迟模块。不要把源码中同名但未公开导出的实现当作包级接口。

## 按场景生成

- **技术指标与数据处理**：阅读 `api-process_data.md` 和 [示例与数据契约](references/examples.md)。指标参数常为 `data, n`；保留时间索引，核对 OHLCV 列、窗口预热与 dropna 后的索引，不把 Series 当作完整 DataFrame。
- **问财与可转债**：阅读 `api-wencai.md`、`api-akshare_data.md`。问财使用 IWENCAI_API_KEY，可选 IWENCAI_BASE_URL；不要生成旧 pywencai Cookie 调用。query_type 支持 stock/fund/conbond/usstock。区分 WencaiError 与成功空表；请求失败时停止依赖该结果的写库、轮动或委托。
- **其他行情来源**：阅读 `api-get_data.md`、`api-edit_akshare.md`。按实现核对代码格式、周期、复权与返回字段；旧函数可能捕获异常返回 None，必须检查结果，不能因函数返回就宣布成功。
- **数据库、策略与定时处理**：阅读 `api-global_functions.md`、`api-create_strategy.md` 及相关处理模块。明确路径、表名和 replace/append/delete 行为；订单队列表写入也有交易后果。不要沿用源码中的个人默认路径。用户仅要求取数时，不附加保存步骤。
- **QMT 桥接**：阅读 [桥接与交易约束](references/trading.md) 和 `api-qmt_bridge.md`。外部 Python 用 BridgeClient；服务端经 export_qmt_bridge 导出后在大QMT运行。不要把完整 Python >=3.11 包导入大QMT内置 Python 3.6。
- **原生 xtquant 或 MT5**：阅读 `api-qmt_trade.md`、`api-mt5_trade.md` 或 `api-mt5_ic_custom.md` 以及交易约束。遵循用户指定环境，不静默替换大QMT ContextInfo、外部 xtquant 和 MT5。
- **邮件**：阅读 `api-send_email.md`；生成代码不等于授权发送。

## 实现和验证

优先复用库函数，将数据适配、参数验证和业务组合写在调用层；库无现成功能时明确新增辅助代码。凭据来自运行时安全输入或环境，不写进源码、命令参数或日志。Windows 示例使用 PowerShell，文件写入显式 UTF-8；桥接服务端编码例外见交易约束。

脚本包含明确依赖、必要导入、参数入口、中文注释和异常处理。纯计算用合成数据验证数值或输出结构；网络、终端、数据库和交易逻辑用 mock 或临时文件验证。写代码请求不授权执行实际下单、撤单、发邮件或修改业务数据库；已有明确执行授权时遵守其范围。

交付说明选用 API、运行方式、必需配置和实际验证程度。编译通过、mock 通过、在线只读查询、原生终端加载和真实成交是不同证据，不互相替代。
