# create_strategy API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## get_filtered_data

```python
get_filtered_data(db_path: str, table_names: list, output_table_name: str)
```

从SQLite数据库中读取多个表的数据，并筛选出对于每个'证券代码'，
所有'成交数量'乘以'买卖'的和大于0的行，并将合并结果保存到指定的数据表中。

:param db_path: 数据库文件的路径。
:param table_names: 数据表名称的列表。
:param output_table_name: 输出数据表的名称。

实现：`yuhanbolh/create_strategy.py:13`。

## process_and_merge_data

```python
process_and_merge_data(db_path: str, table_name: str, xt_trader: str, acc: str)
```

读取指定数据库表中的'证券代码'列，获取对应的行情数据和持仓量，
并将行情数据与原始表数据以及持仓数据合并，最后打印合并后的数据。

:param db_path: 数据库文件的路径。
:param table_name: 读取证券代码的数据表名称。
:param acc: 账户标识符。

实现：`yuhanbolh/create_strategy.py:71`。

## mole_hunting_delegation

```python
mole_hunting_delegation(db_path: str, table_name: str, acc: str, drawdown: float, active_thres: float, deal_thres: float, xt_trader: str)
```

读取指定数据库表中的'证券代码'列，获取对应的行情数据，
并根据止盈条件和买入条件处理数据，最后返回处理后的数据。

:param db_path: 数据库文件的路径。
:param table_name: 读取证券代码的数据表名称。
:param acc: 账户标识符。
:param drawdown: 触发止盈的最大回撤跌幅。
:param active_thres: 激活止盈的最大涨幅阈值。
:param deal_thres: 执行止盈的最小涨幅阈值。
:return: 处理后的DataFrame。

实现：`yuhanbolh/create_strategy.py:118`。
