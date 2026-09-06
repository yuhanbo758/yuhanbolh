# 桥接与交易约束

## 环境边界

- Python >=3.11：yuhanbolh 包、BridgeClient 外部调用端。
- 大QMT内置 Python 3.6：仅加载导出的独立服务端，使用 ContextInfo；export_qmt_bridge 输出服务端为 GBK，其余文件为 UTF-8。目标必须新建或为空，不启动服务，不创建业务库。
- 外部 xtquant：qmt_trade 等原生客户端函数，需用户终端环境支持，不按普通 pip 依赖推断可用性。
- MT5：mt5_trade / mt5_ic_custom，需要 MetaTrader5 运行环境及终端；桥接 health 不能证明其可用。

## 标准桥接

客户端验证 product=qmt_bridge_standard、api_version=1，不兼容仅凭相似路由猜测的其他桥接。默认地址 http://127.0.0.1:1693，用户配置优先。token 从安全运行时输入取得；timeout 必须大于60秒，默认75秒。

普通方法返回完整 JSON 信封，业务内容通过 response['data'] 取得；health 的身份、能力与就绪标志在顶层。events_sse 是迭代器，产出 event/id/data 原始帧。每次业务请求会核对能力和就绪状态。

查询账户别名 normal/credit/all；order 只接受 normal/credit 和 BUY/SELL。行情需核对服务端时间与陈旧性。

## 订单与恢复

- 保留默认 dry_run=True。实际执行需用户授权、服务端许可，以及 dry_run=False 和 confirm_live_order='LIVE_ORDER'；不要在演示中默认启用。
- user_order_id 是持久保存的稳定唯一请求号。异常或超时表示结果未知，先用 orders(request_id=原请求号) 并结合 trades/ledger 对账，禁止换号或盲目重试。
- 信用账户 op_type 必须显式核对，不猜测融资融券路由。strategy_name 非空且不超过20个 GBK 字节。
- 提交信封不等于成交；预演也不证明资金、券商资格或可成交。
- 事件恢复同时保存 stream_id 和 after_id；缺口、流变化或 bridge_gap 后查询对账，不能把空事件流认作没有成交。

## 旧数据库和原生辅助

方法名不是副作用契约：保存持仓、订单队列、轮动和账户维护可能写表、删除记录或触发后续交易。检查实现中的固定路径、异常后是否继续执行。只生成程序时用参数化路径、显式执行入口和 mock 验证；原生函数没有 dry_run 参数时不得虚构。
