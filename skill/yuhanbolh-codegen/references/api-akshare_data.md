# akshare_data API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## akshare_convertible_bond

```python
akshare_convertible_bond()
```

返回 AKShare 集思录可转债强赎原始表；网络异常向调用方传播。

实现：`yuhanbolh/akshare_data.py:7`。

## filter_bond_cb_redeem_data_and_save_to_db

```python
filter_bond_cb_redeem_data_and_save_to_db(db_path=None)
```

筛选已公告/即将强赎可转债，返回七列数据。

强赎状态与强赎天计数分别保留；可转债代码加 .SH/.SZ 后缀。
db_path 默认为 None，仅取数；显式传入时替换“满足赎回可转债”表。
网络或缺列异常会终止，不把错误当空表写入数据库。

实现：`yuhanbolh/akshare_data.py:14`。
