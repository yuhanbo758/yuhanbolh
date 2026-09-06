# 示例与数据契约

## 技术指标：合成数据离线验证

MA(data, n) 与 RSI(data, n) 读取 close，返回带原始索引的 Series，可能删除预热空值。合并按索引对齐；其他指标可能要求 high/low/volume，须分别查实现。

```python
import pandas as pd
from yuhanbolh import MA, RSI

# 合成日线只用于离线验证调用。
data = pd.DataFrame(
    {"close": [float(i) for i in range(1, 41)]},
    index=pd.date_range("2025-01-01", periods=40, freq="D"),
)
result = data.join(MA(data, 5)).join(RSI(data, 14))
assert result["MA_5"].iloc[-1] == 38.0
print(result.tail())
```

## 问财：失败与空结果分开处理

事先安全配置进程的 IWENCAI_API_KEY。此示例请求网络，不作为离线测试执行。

```python
from yuhanbolh import WencaiError, get_wencai

try:
    data = get_wencai("沪深300成分股", query_type="stock", loop=True, timeout=60)
except WencaiError:
    # 失败必须终止，不能伪造空表交给下游写库或交易。
    raise
if data.empty:
    print("查询成功，但没有匹配记录")
else:
    print(data.head())
```

filter_bond_cb_redeem_data_and_save_to_db(db_path=None) 默认仅返回强赎表，传路径会替换“满足赎回可转债”表。wencai_conditional_query_nz100(query, db_path=None) 默认也不保存，写库前核对表名与覆盖行为。

## 桥接：只读调用

```python
import os
from yuhanbolh import BridgeClient

# 环境变量由此调用脚本约定，不是库自动读取的配置项。
client = BridgeClient(
    base_url=os.environ.get("QMT_BRIDGE_URL", "http://127.0.0.1:1693"),
    token=os.environ.get("QMT_BRIDGE_TOKEN", ""),
    timeout=75,
)
response = client.market("600000.SH")
market = response["data"]
# 按实际响应检查时间与字段，不假设已经是 DataFrame。
print(market)
```

用户要求订单示例时，按 api-qmt_bridge.md 生成 order(..., dry_run=True)。不要为了演示启动终端或启用账户。
