# mt5_trade API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## export_positions_to_db

```python
export_positions_to_db(db_path, table_name)
```

将持仓数据保存到数据库，参数分别是：数据库路径、表名

实现：`yuhanbolh/mt5_trade.py:13`。

## insert_into_db

```python
insert_into_db(db_path, result)
```

将委托后数据插入到成交历史数据到数据库，参数分别是：数据库路径、结果

实现：`yuhanbolh/mt5_trade.py:53`。

## save_unsettled_orders_to_db

```python
save_unsettled_orders_to_db(db_path, table_name)
```

将未成交的委托保存到数据库中，参数分别是：数据库路径、表名

实现：`yuhanbolh/mt5_trade.py:104`。

## execute_order_from_db

```python
execute_order_from_db(db_path, table_name)
```

读取数据库forex_order的委托订单，进行批量委托，参数有两个，一个是数据库路径，一个是表名。例如execute_order_from_db(db_path, "forex_order")

实现：`yuhanbolh/mt5_trade.py:137`。

## market_order_fn

```python
market_order_fn(conn, magic, symbol, volume, sl, tp, deviation, type, comment)
```

插入市价委托，参数：conn为数据库连接对象，magic为EA的magic number，symbol为交易品种，volume为交易量，sl为止损价，tp为止盈价，deviation为价格偏差，type为订单类型，comment为订单注释。例如market_order_fn(conn, magic, symbol, volume, sl, tp, deviation, type, comment)

实现：`yuhanbolh/mt5_trade.py:234`。

## limit_order_fn

```python
limit_order_fn(conn, magic, symbol, volume, price, sl, tp, deviation, type, comment)
```

插入限价委托，参数：conn为数据库连接对象，magic为EA的magic number，symbol为交易品种，volume为交易量，price为价格，sl为止损价，tp为止盈价，deviation为价格偏差，type为订单类型，comment为订单注释。例如limit_order_fn(conn, magic, symbol, volume, price, sl, tp, deviation, type, comment)

实现：`yuhanbolh/mt5_trade.py:271`。

## close_position_fn

```python
close_position_fn(conn, magic, symbol, volume, deviation, type, comment, position)
```

插入平仓委托，参数：conn为数据库连接对象，magic为EA的magic number，symbol为交易品种，volume为交易量，deviation为价格偏差，type为订单类型，comment为订单注释，position为持仓单号。例如close_position_fn(conn, magic, symbol, volume, deviation, type, comment, position)

实现：`yuhanbolh/mt5_trade.py:308`。

## cancel_order_fn

```python
cancel_order_fn(conn, magic, order)
```

插入撤单委托，参数：conn为数据库连接对象，magic为EA的magic number，order为订单号。例如cancel_order_fn(conn, magic, order)

实现：`yuhanbolh/mt5_trade.py:345`。

## cancel_pending_order

```python
cancel_pending_order(db_path, magic)
```

从 unsettled_orders 表中读取特定 magic 的数据，并将其插入到 forex_order 表中
:param db_path: 数据库文件路径
:param magic: 用于筛选的 magic 值

实现：`yuhanbolh/mt5_trade.py:364`。

## remove_unavailable_products_mt5

```python
remove_unavailable_products_mt5(db_path, table_name)
```

从成分股中清除不能在mt5交易的产品，参数分别是需要清除的数据库路径和表名。例如remove_unavailable_products_mt5(db_path, "成分股")

实现：`yuhanbolh/mt5_trade.py:393`。

## export_non_strategy_positions

```python
export_non_strategy_positions(db_path, tables, magic_values)
```

处理并保存“非策略持仓”，筛选出不是所有策略的持仓，即不在策略的成分股中，已经被剔除，每月运行一次。参数：数据库路径，策略表名，策略magic值（即策略值）。例如export_non_strategy_positions(db_path, tables, magic_values)

实现：`yuhanbolh/mt5_trade.py:441`。

## process_non_strategy_positions

```python
process_non_strategy_positions(db_path)
```

将“非策略持仓”表中的数据插入到forex_order表中，进行平仓持仓，每月运行一次。参数：数据库路径。例如process_non_strategy_positions(db_path)

实现：`yuhanbolh/mt5_trade.py:478`。
