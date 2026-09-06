# qmt_bridge API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## export_qmt_bridge

```python
export_qmt_bridge(output_dir)
```

导出独立桥接到新建或空目录，返回绝对 Path。

服务端输出为真实GBK字节，其他源码和文档为UTF-8。只生成文件，
不启动服务、不配置账户、不创建业务数据库；非空目录拒绝覆盖。

实现：`yuhanbolh/qmt_bridge/__init__.py:11`。

## BridgeError

```python
BridgeError
```

HTTP、业务或协议错误；订单报错不代表没有提交。

实现：`yuhanbolh/qmt_bridge/client.py:13`。

## BridgeClient

```python
BridgeClient
```

方法返回完整JSON信封，调用者通过response['data']取得业务内容。

实现：`yuhanbolh/qmt_bridge/client.py:23`。

## BridgeClient.__init__

```python
BridgeClient.__init__(self, base_url=DEFAULT_BASE_URL, token=DEFAULT_TOKEN, timeout=DEFAULT_TIMEOUT)
```

无 docstring；需检查源码实现与配套文档。

实现：`yuhanbolh/qmt_bridge/client.py:25`。

## BridgeClient.health

```python
BridgeClient.health(self, required_capabilities=(), require_ready=True)
```

验证产品/API及所需能力；require_ready=False可仅检查服务身份。

实现：`yuhanbolh/qmt_bridge/client.py:99`。

## BridgeClient.tick

```python
BridgeClient.tick(self, stock_code)
```

查询指定证券Tick，返回完整JSON信封；不订阅、不下单。

实现：`yuhanbolh/qmt_bridge/client.py:116`。

## BridgeClient.market

```python
BridgeClient.market(self, stock_code)
```

查询证券标准行情，返回完整JSON信封及服务端行情时间。

实现：`yuhanbolh/qmt_bridge/client.py:120`。

## BridgeClient.history_data

```python
BridgeClient.history_data(self, stock_code, period='1d', count=60, fields=None, start_time=None, end_time=None, dividend_type='none', fill_data=False, subscribe=False)
```

读取历史K线；默认不补数据、不订阅，缺失或陈旧数据由调用者核对。

实现：`yuhanbolh/qmt_bridge/client.py:124`。

## BridgeClient.assets

```python
BridgeClient.assets(self, account_type='all')
```

查询已启用账户资产；账户只能用normal/credit/all别名。

实现：`yuhanbolh/qmt_bridge/client.py:133`。

## BridgeClient.positions

```python
BridgeClient.positions(self, account_type='all')
```

查询已启用账户持仓，不修改账户或策略归属。

实现：`yuhanbolh/qmt_bridge/client.py:137`。

## BridgeClient.orders

```python
BridgeClient.orders(self, account_type='all', request_id=None)
```

有request_id时查询持久化请求，否则查询柜台委托；用于未知结果对账。

实现：`yuhanbolh/qmt_bridge/client.py:141`。

## BridgeClient.trades

```python
BridgeClient.trades(self, account_type='all')
```

查询柜台成交记录，查询恢复不等于收到原始成交回调。

实现：`yuhanbolh/qmt_bridge/client.py:145`。

## BridgeClient.order

```python
BridgeClient.order(self, stock_code, account_type, action, price, volume, user_order_id, strategy_name='http_bridge', op_type=None, dry_run=True, confirm_live_order=None)
```

默认只预演；真实报单需LIVE_ORDER及服务端许可，超时不得换号重报。

user_order_id必须稳定唯一；信用账户必须显式指定op_type。
返回提交信封不等于柜台成交，应查询orders/trades/ledger核对。

实现：`yuhanbolh/qmt_bridge/client.py:149`。

## BridgeClient.ledger

```python
BridgeClient.ledger(self, request_id=None, account_type='all')
```

按请求号/账户查询桥接SQLite账本，不创建或迁移外部业务库。

实现：`yuhanbolh/qmt_bridge/client.py:179`。

## BridgeClient.events

```python
BridgeClient.events(self, after_id=0, event_types=None, timeout=20, stream_id=None)
```

长轮询事件；游标须连同stream_id保存，缺口需通过查询对账。

实现：`yuhanbolh/qmt_bridge/client.py:184`。

## BridgeClient.events_sse

```python
BridgeClient.events_sse(self, after_id=0, event_types=None, stream_id=None, max_seconds=60, heartbeat=10)
```

有界SSE迭代器；返回原始事件帧，不自动提交订单或持久化消费游标。

实现：`yuhanbolh/qmt_bridge/client.py:193`。
