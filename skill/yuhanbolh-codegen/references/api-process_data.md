# process_data API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## save_data

```python
save_data(function, db_path, original_table_name, new_table_name)
```

执行 function 以获取数据，并将该数据与 original_table_name 表的内容合并，
然后将合并后的数据保存到 new_table_name 表中。

参数:
function: callable
    获取数据的自定义函数。
db_path: str
    数据库文件的路径。
original_table_name: str
    原始数据表的名称。
new_table_name: str
    用于保存合并后数据的新表名称。

实现：`yuhanbolh/process_data.py:131`。

## MA

```python
MA(data, n)
```

获取简单移动平均线，参数有2个，一个是数据源，一个是日期。例如MA(data, 20)

实现：`yuhanbolh/process_data.py:197`。

## EMA

```python
EMA(data, n)
```

获取指数移动平均线，参数有2个，一个是数据源，一个是日期。例如EMA(data, 20)

实现：`yuhanbolh/process_data.py:203`。

## ichimoku_cloud

```python
ichimoku_cloud(data, conversion_periods, base_periods, lagging_span2_periods, displacement)
```

获取一目均衡表基准线 (data, conversion_periods, base_periods, lagging_span2_periods, displacement)
参数有5个，第一个是数据源，其他4个分别是一目均衡表基准线 (9, 26, 52, 26)，即ichimoku_cloud(data,9, 26, 52, 26)

实现：`yuhanbolh/process_data.py:210`。

## VWMA

```python
VWMA(data, n)
```

成交量加权移动平均线 VWMA (data, 20)，参数有2个，1个是数据源，另一个是日期，通过为20

实现：`yuhanbolh/process_data.py:249`。

## HullMA

```python
HullMA(data, n=9)
```

计算Hull MA船体移动平均线 Hull MA (data,9)，参数有2，一个是数据源，另一个是日期，一般为9。例如HullMA(data, 9)

实现：`yuhanbolh/process_data.py:258`。

## RSI

```python
RSI(data, n)
```

计算RSI指标，参数有2，一个为数据源，另一个为日期，一般为14，即RSI(data, 14)

实现：`yuhanbolh/process_data.py:273`。

## STOK

```python
STOK(data, n, m, t)
```

计算Stochastic，k是主线，d_signal是信号线，参数有4，一个是数据源，另外三个为日期，一般为STOK(data, 14, 3, 3)

实现：`yuhanbolh/process_data.py:286`。

## CCI

```python
CCI(data, n)
```

计算CCI指标，参数有2，一个是数据源，另一个是日期，一般为20，即CCI(data, 20)

实现：`yuhanbolh/process_data.py:307`。

## ADX

```python
ADX(data, n)
```

平均趋向指数ADX(14)，参数有2，一个是数据源，另一个是日期，一般为14，即ADX(data,14)

实现：`yuhanbolh/process_data.py:316`。

## AO

```python
AO(data)
```

计算动量震荡指标(AO)，参数只有一个，即数据源

实现：`yuhanbolh/process_data.py:348`。

## MTM

```python
MTM(data)
```

计算动量指标(10)，参数只有一个，即数据源

实现：`yuhanbolh/process_data.py:357`。

## MACD_Level

```python
MACD_Level(data, n_fast, n_slow)
```

计算MACD Lvel指标，参数有3个，第一个是数据源，其余两个为日期，一般取12和26，即MACD_Level(data, 12,26)

实现：`yuhanbolh/process_data.py:364`。

## Stoch_RSI

```python
Stoch_RSI(data, smoothK, smoothD, lengthRSI, lengthStoch)
```

计算Stoch_RSI(data,3, 3, 14, 14)，有5个参数，第1个为数据源

实现：`yuhanbolh/process_data.py:380`。

## WPR

```python
WPR(data, n)
```

计算威廉百分比变动，参数有2，第1是数据源，第二是日期，一般为14，即WPR(data, 14)

实现：`yuhanbolh/process_data.py:406`。

## BBP

```python
BBP(data, n)
```

计算Bull Bear Power牛熊力量(BBP)，参数有2，一个是数据源，另一个是日期，一般为20，但在tradingview取13，即BBP(data, 13)

实现：`yuhanbolh/process_data.py:417`。

## UO

```python
UO(data, n1, n2, n3)
```

计算Ultimate Oscillator终极震荡指标UO (data,7, 14, 28)，有4个参数，第1个是数据源，其他的是日期

实现：`yuhanbolh/process_data.py:425`。

## linear_regression_dfcf

```python
linear_regression_dfcf(data, days_list)
```

计算线性回归，参数分别是：数据源，日期列表，一般为[5, 10, 20, 30, 60]，即linear_regression_dfcf(data, [5, 10, 20, 30, 60])

实现：`yuhanbolh/process_data.py:438`。

## generate_stat_data

```python
generate_stat_data(stock_code)
```

从东财获取8年数据，计算各种指标指标，参数为股票代码。例如generate_stat_data("AAPL.NAS")

实现：`yuhanbolh/process_data.py:541`。

## calculate_xirr

```python
calculate_xirr(cash_flows, dates)
```

计算xirr年化收益率。参数分别是：现金流、日期。例如calculate_xirr(cash_flows, dates)

实现：`yuhanbolh/process_data.py:628`。

## calculate_annual_return

```python
calculate_annual_return(table_name)
```

计算年化收益率、现金流之和和净现值。参数分别是：表名。例如calculate_annual_return("外汇")

实现：`yuhanbolh/process_data.py:649`。

## get_processed_code

```python
get_processed_code(stock_code)
```

股票代码增加市场信息。参数是股票代码，例如get_processed_code("000001.SZ")

实现：`yuhanbolh/process_data.py:688`。

## tongda_code_convert

```python
tongda_code_convert(stock_code)
```

通达信代码转换。参数是股票代码，例如tongda_code_convert("000001.SZ")

实现：`yuhanbolh/process_data.py:714`。

## clean_execute_general_trade

```python
clean_execute_general_trade(conn)
```

清洗execute_general_trade表

实现：`yuhanbolh/process_data.py:725`。

## insert_order

```python
insert_order(cursor, conn, code, price, quantity, buy_sell, strategy, remark, datetime_str)
```

插入订单到place_general_order表

实现：`yuhanbolh/process_data.py:746`。

## delete_receive_condition_row

```python
delete_receive_condition_row(cursor, conn, rowid)
```

删除receive_condition表中的行

实现：`yuhanbolh/process_data.py:761`。

## delete_execute_general_trade_row

```python
delete_execute_general_trade_row(cursor, conn, strategy, code)
```

删除execute_general_trade表中的行

实现：`yuhanbolh/process_data.py:770`。

## process_price_grid

```python
process_price_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid)
```

处理价差网格策略

实现：`yuhanbolh/process_data.py:781`。

## process_amplitude_grid

```python
process_amplitude_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid)
```

处理振幅网格策略

实现：`yuhanbolh/process_data.py:850`。

## process_immediate_rows

```python
process_immediate_rows(rows, cursor, conn)
```

处理监控策略。参数分别是：行数据、数据库游标、数据库连接。例如process_immediate_rows(rows, cursor, conn)

实现：`yuhanbolh/process_data.py:938`。

## portfolio_rotation

```python
portfolio_rotation(order_note, order_quantity, strategy_name)
```

查询轮动候选并附加委托信息，本函数不写库、不下单。

查询失败向调用方抛出异常；成功无候选返回固定四列空表。
明确按证券代码字段读取，排除正股字段、空代码，避免生成错误委托。

实现：`yuhanbolh/process_data.py:1090`。

## process_scheduled_tasks

```python
process_scheduled_tasks(scheduled_tasks, cursor, conn)
```

处理问财轮动任务；查询失败或成功空表时跳过该任务的所有写库操作。

scheduled_tasks为八元素行序列，cursor/conn由调用方持有；
成功非空任务按既有逻辑更新委托数据，不在此函数执行柜台报单。

实现：`yuhanbolh/process_data.py:1135`。
