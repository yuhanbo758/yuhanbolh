# qmt_trade API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## save_stock_asset

```python
save_stock_asset(asset, db_path)
```

证券资产查询并保存到数据库manage_assets，参数：资产对象（固定）、数据库路径

实现：`yuhanbolh/qmt_trade.py:26`。

## save_positions

```python
save_positions(positions, db_path)
```

获取持仓数据并保存到数据库account_holdings，参数：持仓对象（固定）、数据库路径

实现：`yuhanbolh/qmt_trade.py:54`。

## save_daily_orders

```python
save_daily_orders(orders, db_path)
```

查询当日委托并保存到数据库daily_orders，参数：委托对象（固定）、数据库路径

实现：`yuhanbolh/qmt_trade.py:79`。

## save_daily_trades

```python
save_daily_trades(trades, db_path)
```

查询当日成交并保存到数据库daily_trades，参数：成交对象（固定）、数据库路径

实现：`yuhanbolh/qmt_trade.py:114`。

## calculate_remaining_holdings

```python
calculate_remaining_holdings(db_path, strategy_tables)
```

查询除策略外的其他持仓，并将它保存到数据表other_positions，参数：数据库路径、策略表名称列表

实现：`yuhanbolh/qmt_trade.py:149`。

## insert_buy_sell_data

```python
insert_buy_sell_data(place_order_table, security_code, order_price, order_volume, trade_direction, strategy_name, order_remark)
```

当为卖出时插入的数据，参数包括数据表名、证券代码、委托价格、委托数量、买卖方向、策略名称、委托备注

实现：`yuhanbolh/qmt_trade.py:182`。

## place_orders

```python
place_orders(db_path, table_name, trade_table_name, xt_trader, acc)
```

证券委托，参数分别是：数据库路径、委托数据表名称、成交数据表名称、判断处函数前缀（不改动）、账号（不改动）
注：报价类型主要有xtconstant.FIX_PRICE（限价）、xtconstant.LATEST_PRICE（最新介）、xtconstant.MARKET_PEER_PRICE_FIRST（对手方最优）、xtconstant.MARKET_MINE_PRICE_FIRST（本方最优）

实现：`yuhanbolh/qmt_trade.py:223`。

## place_order_based_on_asset

```python
place_order_based_on_asset(xt_trader, acc, xtdata)
```

比较沪深两市的一天期的买一国债逆回购，选择值大的进行卖出，参数分别是：交易对象（固定）、账号（固定）、数据对象（固定）

实现：`yuhanbolh/qmt_trade.py:329`。

## sort_and_update_table

```python
sort_and_update_table(table_name, db_path='<USER_PATH>')
```

对委托数据表进行排序并更新，先卖后买，先评分（操作）高后评分低。参数是表名和数据库路径

实现：`yuhanbolh/qmt_trade.py:395`。

## MyXtQuantTraderCallback

```python
MyXtQuantTraderCallback
```

定义qmt推送的类

实现：`yuhanbolh/qmt_trade.py:435`。

## MyXtQuantTraderCallback.__init__

```python
MyXtQuantTraderCallback.__init__(self, db_path='<USER_PATH>')
```

无 docstring；需检查源码实现与配套文档。

实现：`yuhanbolh/qmt_trade.py:436`。

## MyXtQuantTraderCallback.on_stock_asset

```python
MyXtQuantTraderCallback.on_stock_asset(self, asset)
```

资金变动推送  注意，该回调函数目前不生效

实现：`yuhanbolh/qmt_trade.py:440`。

## MyXtQuantTraderCallback.on_stock_trade

```python
MyXtQuantTraderCallback.on_stock_trade(self, trade)
```

成交变动推送，每增加一个策略都要往里增加保存数据表的名称

实现：`yuhanbolh/qmt_trade.py:472`。

## MyXtQuantTraderCallback.on_stock_position

```python
MyXtQuantTraderCallback.on_stock_position(self, positions)
```

持仓变动推送  注意，该回调函数目前不生效

实现：`yuhanbolh/qmt_trade.py:515`。

## MyXtQuantTraderCallback.on_disconnected

```python
MyXtQuantTraderCallback.on_disconnected(self)
```

连接断开
:return:

实现：`yuhanbolh/qmt_trade.py:544`。

## MyXtQuantTraderCallback.on_stock_order

```python
MyXtQuantTraderCallback.on_stock_order(self, order)
```

委托回报推送
:param order: XtOrder对象
:return:

实现：`yuhanbolh/qmt_trade.py:554`。

## MyXtQuantTraderCallback.on_order_error

```python
MyXtQuantTraderCallback.on_order_error(self, order_error)
```

委托失败推送
:param order_error:XtOrderError 对象
:return:

实现：`yuhanbolh/qmt_trade.py:563`。

## MyXtQuantTraderCallback.on_cancel_error

```python
MyXtQuantTraderCallback.on_cancel_error(self, cancel_error)
```

撤单失败推送
:param cancel_error: XtCancelError 对象
:return:

实现：`yuhanbolh/qmt_trade.py:572`。

## MyXtQuantTraderCallback.on_order_stock_async_response

```python
MyXtQuantTraderCallback.on_order_stock_async_response(self, response)
```

异步下单回报推送
:param response: XtOrderResponse 对象
:return:

实现：`yuhanbolh/qmt_trade.py:581`。

## MyXtQuantTraderCallback.on_account_status

```python
MyXtQuantTraderCallback.on_account_status(self, status)
```

:param response: XtAccountStatus 对象
:return:

实现：`yuhanbolh/qmt_trade.py:591`。

## save_daily_data

```python
save_daily_data(xt_trader, acc, db_path)
```

保存当日的持仓、委托和成交数据到数据库
:param xt_trader: 交易对象
:param acc: 账户信息
:param db_path: 数据库路径

实现：`yuhanbolh/qmt_trade.py:603`。

## cancel_all_orders

```python
cancel_all_orders(xt_trader, acc)
```

查询未成交的委托，然后进行逐一撤单，参数分别是：交易对象（固定）、账号（固定）

实现：`yuhanbolh/qmt_trade.py:648`。

## run_weekdays_at

```python
run_weekdays_at(time_str, function)
```

周一到周五运行函数

实现：`yuhanbolh/qmt_trade.py:672`。
