# get_data API



来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。



## get_tdx_market_address



```python

get_tdx_market_address(cfg_file)

```



从通达信配置文件获取行情IP地址



实现：`yuhanbolh/get_data.py:32`。



## get_financial_data



```python

get_financial_data(server_ip, server_port, data_function)

```



处理数据



实现：`yuhanbolh/get_data.py:59`。



## get_security_quotes



```python

get_security_quotes(market_codes)

```



获取盘口数据(买卖五档)

Args:

    market_codes: 列表，每个元素为元组(市场代码,股票代码)，如[(0,'000001'),(0,'000002')]



实现：`yuhanbolh/get_data.py:80`。



## get_security_bars



```python

get_security_bars(category, market, code, start, count)

```



获取K线数据

Args:

    category: K线类型（0 5分钟K线; 1 15分钟K线; 2 30分钟K线; 3 1小时K线; 4 日K线; 5 周K线; 6 月K线; 7 1分钟; 8 1分钟K线; 9 日K线; 10 季K线; 11 年K线）

    market: 市场代码（0:深圳, 1:上海）

    code: 股票代码

    start: 起始位置（0为最新）

    count: 数量，最高800



实现：`yuhanbolh/get_data.py:91`。



## get_security_count



```python

get_security_count(market)

```



获取市场股票数量

Args:

    market: 市场代码（0:深圳, 1:上海）



实现：`yuhanbolh/get_data.py:108`。



## get_index_bars



```python

get_index_bars(category, market, code, start, count)

```



获取指数K线数据

Args:

    category: K线类型（0:分时, 1:1分钟, 2:5分钟, 3:15分钟, 4:30分钟, 5:60分钟）

    market: 市场代码（0:深圳, 1:上海）

    code: 指数代码

    start: 起始位置（0为最新）

    count: 数量



实现：`yuhanbolh/get_data.py:119`。



## get_history_minute_time_data



```python

get_history_minute_time_data(market, code, date)

```



获取历史分钟数据

Args:

    market: 市场代码（0:深圳, 1:上海）

    code: 股票代码

    date: 日期，格式如20241115



实现：`yuhanbolh/get_data.py:136`。



## get_transaction_data



```python

get_transaction_data(market, code, start, count)

```



获取历史分笔成交

Args:

    market: 市场代码（0:深圳, 1:上海）

    code: 股票代码

    start: 起始位置（0为最新）

    count: 数量



实现：`yuhanbolh/get_data.py:151`。



## get_finance_info



```python

get_finance_info(market, code)

```



获取财务数据

Args:

    market: 市场代码（0:深圳, 1:上海）

    code: 股票代码



实现：`yuhanbolh/get_data.py:167`。



## json_to_dfcf



```python

json_to_dfcf(code, days=1, fqt=1, klt=101)

```



通过东方财富api获取K线数据，参数包括股票代码（000001.SZ的代码获取最新数），天数，复权类型，K线类型

`klt`：K 线周期，可选值包括 5（5 分钟 K 线）、15（15 分钟 K 线）、30（30 分钟 K 线）、60（60 分钟 K 线）、101（日 K 线）、102（周 K 线）、103（月 K 线）等。

`fqt`：复权类型，可选值包括 0（不复权）、1（前复权）、2（后复权）。



实现：`yuhanbolh/get_data.py:184`。



## json_to_dfcf_qmt



```python

json_to_dfcf_qmt(code, days=7 * 365, fqt=1)

```



通过类似000001.SZ的代码获取日线数据（东财api），参数3个，分别是：代码（必要），天数，复权类型



实现：`yuhanbolh/get_data.py:235`。



## json_to_dfcf_qmt_jyr



```python

json_to_dfcf_qmt_jyr(code, days=7 * 250, fqt=1)

```



与上面的函数相同，只是通数days参数为天数，而不是日期。比如100表示100个交易日的数据，而不是日期往前推100天。



实现：`yuhanbolh/get_data.py:286`。



## query_stock_data



```python

query_stock_data(stock_code, days_back=60, frequency='d', adjustflag='2')

```



从baostock获取股票数据，参数有4个：股票代码，周期，复权类型，指标列表

adjustflag默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据

adjustflag复权类型：不复权：3；后复权：1；前复权：2



实现：`yuhanbolh/get_data.py:338`。



## qmt_data_source



```python

qmt_data_source(stock_code, days=7 * 365)

```



通过qmt获取证券的7年K线历史数据，不包数据下载补充



实现：`yuhanbolh/get_data.py:373`。



## qmt_data_source_download



```python

qmt_data_source_download(stock_code, days=7 * 365)

```



通过qmt获取证券的7年K线历史数据，包含数据下载补充



实现：`yuhanbolh/get_data.py:425`。



## download_7_years_data



```python

download_7_years_data(stock_list)

```



从国金qmt中获取指定代码的近7年行情数据



实现：`yuhanbolh/get_data.py:473`。



## get_valuation_ratios



```python

get_valuation_ratios(code)

```



获取价值大师网的大师价值数据，参数为：股票代码



实现：`yuhanbolh/get_data.py:484`。



## position_close_process_data



```python

position_close_process_data(table_name, db_path='<USER_PATH>')

```



获取持仓和收盘价，从而获得整体的交易日期和现金流，然后保存到数据库到r"<USER_PATH>"



实现：`yuhanbolh/get_data.py:582`。



## get_exchange_rate



```python

get_exchange_rate(secid)

```



从东方财富网的API获取指定股票代码的汇率信息，并提取汇率数据。



参数:

secid: str

    股票代码，格式为 "市场代码.股票代码"，例如 "133.USDCNH"。



返回:

exchange_rate: float

    提取的汇率数据。



实现：`yuhanbolh/get_data.py:622`。



## get_snapshot



```python

get_snapshot(code_list: list[str])

```



从QMT获得行情数据，筛选出符合条件的标的数据



实现：`yuhanbolh/get_data.py:706`。



## stock_info_global_em



```python

stock_info_global_em()

```



东方财富-全球财经快讯

外部数据源链接省略；以目标版本实现为准。

:return: 全球财经快讯摘要

:rtype: pandas.DataFrame



实现：`yuhanbolh/get_data.py:768`。
