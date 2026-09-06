# wencai API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## WencaiError

```python
WencaiError
```

查询失败或数据不完整；不得将此异常作为空选股结果执行交易。

实现：`yuhanbolh/wencai.py:50`。

## get_wencai

```python
get_wencai(question, query_type='stock', loop=True, *, page_size=100, timeout=60)
```

按自然语言查询股票/基金/可转债/美股，返回原始业务字段 DataFrame。

loop=False 只取第一页。成功无匹配返回含证券代码、证券简称的空表；
配置、HTTP、协议或分页失败抛出 WencaiError，不返回部分数据。
IWENCAI_API_KEY 必填，IWENCAI_BASE_URL 可指定服务根地址或完整接口。

实现：`yuhanbolh/wencai.py:60`。

## wencai_conditional_query

```python
wencai_conditional_query(query, fetch_all=True, page_size=100, timeout=60)
```

获取可转债策略十列数据，余额单位亿元；无匹配返回固定列空表。

实现：`yuhanbolh/wencai.py:204`。

## get_satisfy_redemption

```python
get_satisfy_redemption(query)
```

获取强赎条件查询的六列数据；查询失败抛出 WencaiError。

实现：`yuhanbolh/wencai.py:211`。

## get_clean_data

```python
get_clean_data(question)
```

获取并清洗可转债十二列数据，保留满足强赎及强赎天计数。

实现：`yuhanbolh/wencai.py:216`。

## wencai_conditional_query_nz100

```python
wencai_conditional_query_nz100(query, db_path=None)
```

查询美股成分股；仅显式传入 db_path 时替换 nasdaq_100 表。

返回股票代码、股票简称、指数、价值代码和 mt5代码；异常不会写库。

实现：`yuhanbolh/wencai.py:222`。
