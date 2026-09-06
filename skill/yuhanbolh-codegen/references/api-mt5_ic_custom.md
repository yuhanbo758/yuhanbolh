# mt5_ic_custom API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## save_exchange_rates_to_db

```python
save_exchange_rates_to_db(db_path, table_name)
```

获取所需的所有汇率，计算兑换到CNH的汇率，并将结果保存到数据库中。

参数:
db_path: str
    数据库文件的路径。
table_name: str
    数据库中的表名，用于存储汇率数据。

实现：`yuhanbolh/mt5_ic_custom.py:60`。

## get_all_valuation_ratios_db

```python
get_all_valuation_ratios_db(db_path, table_name)
```

从数据库表中读取“价值代码”，然后获取相应的价值大师价格。

参数:
db_path: str
    数据库文件的路径。
table_name: str
    包含“价值代码”的数据库表名。

返回:
all_ratios: DataFrame
    包含所有价值大师比率数据的 DataFrame。

实现：`yuhanbolh/mt5_ic_custom.py:256`。

## get_mt5_data

```python
get_mt5_data(symbol)
```

从MT5获取指定品种的历史日线数据，并返回包含所有数据的DataFrame。

参数:
symbol: str
    希望获取数据的市场品种的符号。

返回:
df: DataFrame
    包含历史日线数据的DataFrame。

实现：`yuhanbolh/mt5_ic_custom.py:296`。

## get_mt5_data_with_days

```python
get_mt5_data_with_days(symbol, days=365 * 8)
```

从MT5获取指定品种的历史日线数据，日期长度可自定义，并返回包含所有数据的DataFrame。

参数:
symbol: str
    希望获取数据的市场品种的符号。
days: int
    希望获取的历史数据天数，默认为8年(365*8天)。

返回:
df: DataFrame
    包含历史日线数据的DataFrame。

实现：`yuhanbolh/mt5_ic_custom.py:335`。

## get_stock_list_from_db

```python
get_stock_list_from_db()
```

获取data数据中的第几行数据

实现：`yuhanbolh/mt5_ic_custom.py:381`。

## MA_zb

```python
MA_zb(data, n)
```

无 docstring；需检查源码实现与配套文档。

实现：`yuhanbolh/mt5_ic_custom.py:400`。

## EMA_zb

```python
EMA_zb(data, n)
```

获取指数移动平均线，参数有2个，一个是数据源，一个是日期

实现：`yuhanbolh/mt5_ic_custom.py:408`。

## ichimoku_cloud_zb

```python
ichimoku_cloud_zb(data, conversion_periods, base_periods, lagging_span2_periods, displacement)
```

获取一目均衡表基准线 (data, conversion_periods, base_periods, lagging_span2_periods, displacement)
参数有5个，第一个是数据源，其他4个分别是一目均衡表基准线 (9, 26, 52, 26)，即ichimoku_cloud(data,9, 26, 52, 26)

实现：`yuhanbolh/mt5_ic_custom.py:417`。

## VWMA_zb

```python
VWMA_zb(data, n)
```

成交量加权移动平均线 VWMA (data, 20)，参数有2个，1个是数据源，另一个是日期，通过为20

实现：`yuhanbolh/mt5_ic_custom.py:451`。

## HullMA_zb

```python
HullMA_zb(data, n=9)
```

计算Hull MA船体移动平均线 Hull MA (data,9)，参数有2，一个是数据源，另一个是日期，一般为9

实现：`yuhanbolh/mt5_ic_custom.py:464`。

## RSI_zb

```python
RSI_zb(data, n)
```

计算RSI指标，参数有2，一个为数据源，另一个为日期，一般为14，即RSI(data, 14)

实现：`yuhanbolh/mt5_ic_custom.py:483`。

## STOK_zb

```python
STOK_zb(data, n, m, t)
```

计算Stochastic，k是主线，d_signal是信号线，参数有4，一个是数据源，另外三个为日期，一般为STOK(data, 14, 3, 3)

实现：`yuhanbolh/mt5_ic_custom.py:502`。

## CCI_zb

```python
CCI_zb(data, n)
```

计算CCI指标，参数有2，一个是数据源，另一个是日期，一般为20，即CCI(data, 20)

实现：`yuhanbolh/mt5_ic_custom.py:527`。

## ADX_zb

```python
ADX_zb(data, n)
```

平均趋向指数ADX(14)，参数有2，一个是数据源，另一个是日期，一般为14，即ADX(data,14)

实现：`yuhanbolh/mt5_ic_custom.py:543`。

## AO_zb

```python
AO_zb(data)
```

计算动量震荡指标(AO)，参数只有一个，即数据源

实现：`yuhanbolh/mt5_ic_custom.py:588`。

## MTM_zb

```python
MTM_zb(data)
```

计算动量指标(10)，参数只有一个，即数据源

实现：`yuhanbolh/mt5_ic_custom.py:614`。

## MACD_Level_zb

```python
MACD_Level_zb(data, n_fast, n_slow)
```

MACD_1是以金叉和死叉进行判断，参数有3个，第一个是数据源，其余两个为日期，一般取12和26，即MACD(data, 12,26)

实现：`yuhanbolh/mt5_ic_custom.py:629`。

## Stoch_RSI_zb

```python
Stoch_RSI_zb(data, smoothK=3, smoothD=3, lengthRSI=14, lengthStoch=14)
```

计算Stoch_RSI(data,3, 3, 14, 14)，有5个参数，第1个为数据源

实现：`yuhanbolh/mt5_ic_custom.py:653`。

## WPR_zb

```python
WPR_zb(data, n)
```

计算威廉百分比变动，参数有2，第1是数据源，第二是日期，一般为14，即WPR(data, 14)

实现：`yuhanbolh/mt5_ic_custom.py:686`。

## BBP_zb

```python
BBP_zb(data, n)
```

计算Bull Bear Power牛熊力量(BBP)，参数有2，一个是数据源，另一个是日期，一般为20，但在tradingview取13，即BBP(data, 13)

实现：`yuhanbolh/mt5_ic_custom.py:714`。

## UO_zb

```python
UO_zb(data, n1, n2, n3)
```

计算Ultimate Oscillator终极震荡指标UO (data,7, 14, 28)，有4个参数，第1个是数据源，其他的是日期

实现：`yuhanbolh/mt5_ic_custom.py:740`。

## linear_regression_dfcf_zb

```python
linear_regression_dfcf_zb(data, years_list)
```

计算线性回归

实现：`yuhanbolh/mt5_ic_custom.py:771`。

## generate_stat_data_zb

```python
generate_stat_data_zb(stock_code)
```

无 docstring；需检查源码实现与配套文档。

实现：`yuhanbolh/mt5_ic_custom.py:792`。

## ex_fund_valuation

```python
ex_fund_valuation(db_path, table_name_guojin, table_name_result)
```

从指定的表中读取数据，获取估值比率，并将结果保存到新表中。

参数:
db_path: str
    数据库文件的路径。
table_name_guojin: str
    要从中读取数据的表名。
table_name_result: str
    用于保存结果的新表名。

返回:
all_data: DataFrame
    包含估值比率的完整数据集。

实现：`yuhanbolh/mt5_ic_custom.py:952`。

## ex_fund_forex_valuation

```python
ex_fund_forex_valuation(db_path, table_name_guojin, table_name_result)
```

从指定的表中读取数据，获取估值比率，并将结果保存到新表中。

参数:
db_path: str
    数据库文件的路径。
table_name_guojin: str
    要从中读取数据的表名。
table_name_result: str
    用于保存结果的新表名。

返回:
all_data: DataFrame
    包含估值比率的完整数据集。

实现：`yuhanbolh/mt5_ic_custom.py:1021`。

## calculate_totals

```python
calculate_totals(db_path, ea_id, magic)
```

参数：数据库路径、EA_id（平仓策略代码）、magic（持仓策略代码）

实现：`yuhanbolh/mt5_ic_custom.py:1089`。
