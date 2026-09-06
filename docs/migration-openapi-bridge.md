# 问财、AKShare与大QMT桥接迁移说明

## 兼容性变化

- 最低Python从3.9提高到3.11，AKShare >=1.18.94。旧Python用户须先升级环境；大QMT服务端仍为独立Python 3.6单文件。
- 删除pywencai依赖及get_pywencai入口，改用get_wencai(question, query_type="stock", loop=True, *, page_size=100, timeout=60)。其他问财业务入口保留名称，失败现在抛出WencaiError，不再吞错返回空表/None。
- 删除akshare_index_analysis和index_value_name_funddb；上游已移除原接口，本次不提供不同数据口径的替代品。
- wencai_conditional_query_nz100(query, db_path=None)和filter_bond_cb_redeem_data_and_save_to_db(db_path=None)默认只返回数据；显式传路径才替换各自数据表。
- 强赎保存函数返回强赎状态与强赎天计数两个独立字段；旧版强赎天计数实际存状态文字，调用方须改用强赎状态筛选。
- 东方财富本地适配保留列序，按原始字段名映射；不完整分页抛出异常，不能将失败解释为零持仓或无候选。

## OpenAPI配置

配置IWENCAI_API_KEY和可选IWENCAI_BASE_URL后重启调用进程。地址支持服务根地址或/v1/query2data完整路径。不要把密钥放入文档、源码、命令行和日志。支持stock、fund、conbond、usstock自然语言范围；不向接口传旧query_type参数。

## 桥接使用

```python
from yuhanbolh import BridgeClient, export_qmt_bridge
# 仅向新建或空目录生成文件；不会启动服务。
# output = export_qmt_bridge("./qmt_bridge")
client = BridgeClient()
# 部署服务后才显式请求：client.health()
```

导出服务端为真实GBK，其他文件UTF-8；编辑部署服务端须保持GBK。账号、令牌、数据库路径默认为空，账户及真实报单关闭。订单默认dry_run=True，真实报单须显式确认及服务端许可。未知结果先查原请求号，不自动重报；事件缺口用委托、成交和账本对账。模板不包含撤单、算法单、网页或个人策略。

## 上游依据

- [AKShare债券接口文档](https://akshare.akfamily.xyz/data/bond/bond.html)：强赎接口仍为bond_cb_redeem_jsl()，状态和天计数分列。
- [AKShare股票接口文档](https://akshare.akfamily.xyz/data/stock/stock.html)：核对A股实时行情字段。
- [指数估值接口移除记录](https://github.com/akfamily/akshare/issues/5508)。

## 发布与同步

本次只修改本地源码和文档，未提交、推送或发布；后续发布沿用既有版本工作流。本地知识库函数文档需要用户自行同步远端。
