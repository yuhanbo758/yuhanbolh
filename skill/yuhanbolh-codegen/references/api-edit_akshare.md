# edit_akshare API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## stock_zh_a_spot_em

```python
stock_zh_a_spot_em()
```

获取沪深京A股行情，返回原有23列；超时/分页失败抛出 RuntimeError。

实现：`yuhanbolh/edit_akshare.py:78`。

## bond_cov_comparison

```python
bond_cov_comparison()
```

获取可转债比价表，返回原有20列；代码为字符串，数值缺失为NaN。

实现：`yuhanbolh/edit_akshare.py:88`。
