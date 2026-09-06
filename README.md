
股票量化代码，包括通过qmt、同花顺问题和东财api等获取金融数据，以及处理量化这些数据。

查看库中的函数文档，请前往：[yuhanbolh 在线文档](https://docs.sanrenjz.com/yuhanbolh/about)

GitHub 文档下载地址：[article-code/docs/yuhanbolh](https://github.com/yuhanbo758/article-code/tree/main/docs/yuhanbolh)

代码是从多个渠道获取金融数据，在运行过程中提示没有什么模块，就用pip安装什么模块。
但迅投的xtdata和xttrade是不能通过pip安装的，需要把官网http://dict.thinktrader.net/nativeApi/download_xtquant.html 下载，然后放到python的...\Lib\site-packages路径中。

schedule
akshare
scipy
yfinance
baostock
pytdx
MetaTrader5
yuhanbolh

## 安装

本次源码要求 Python >=3.11、AKShare >=1.18.94。大QMT独立服务端仍使用其内置 Python 3.6。

```powershell
python -m pip install --upgrade yuhanbolh
```

项目不再锁死 NumPy 与 pandas 的补丁版本，当前兼容范围为
`numpy>=1.26.4`、`pandas>=2.2.2`，并通过 NumPy 2.x 与 pandas 3.x
回归测试。需要在 Jupyter 中使用时，可安装 Notebook 可选依赖：

```powershell
python -m pip install --upgrade "yuhanbolh[notebook]"
```

`ipykernel 7` 与 Spyder 自带的 `spyder-kernels 2.x/3.x` 当前存在上游约束冲突
（后者要求 `ipykernel<7`）。请把最新版 Jupyter 内核放在独立虚拟环境中；若必须
使用 Spyder 控制台，则应遵循 Spyder 的环境要求保留 `ipykernel 6.x`。这不是
`yuhanbolh` 的运行依赖冲突。

QMT 的 `xtquant` 仍需按迅投官方方式安装。包入口已改为按需加载，因此只使用
技术指标、SQLite 等功能时，不会因为缺少 `xtquant` 而无法导入 `yuhanbolh`。

## 自动发布到 PyPI

仓库的 `.github/workflows/publish.yml` 会在代码推送到 `main` 后自动完成以下操作：

1. 查询 PyPI 上 `yuhanbolh` 的当前版本。
2. 按每位逢十进一的规则生成下一版本，例如 `0.6.8 → 0.6.9 → 0.7.0`。
3. 构建并校验 wheel 和源码发行包。
4. 自动提交 `setup.py` 中的新版本、创建 `v版本号` 标签并发布到 PyPI。

首次使用前，需要在 GitHub 仓库中完成一次配置：

1. 打开 `Settings → Secrets and variables → Actions`，新增仓库 Secret：`PYPI_API_TOKEN`。
2. Secret 的值填写从 PyPI `Account settings → API tokens` 创建的 API Token（建议限定到 `yuhanbolh` 项目）。
3. 打开 `Settings → Actions → General → Workflow permissions`，选择 `Read and write permissions`。

自动生成的版本提交带有 `[skip ci]`，不会递归触发下一次发布。


## 问财 OpenAPI 与标准大QMT桥接

问财凭据仅从环境变量 `IWENCAI_API_KEY` 读取，可选 `IWENCAI_BASE_URL`。
配置后重新启动 Python 进程；不要将密钥写入源码或提交到仓库。

```python
import yuhanbolh as lh

data = lh.get_wencai("可转债；债券余额", query_type="conbond")
bonds = lh.wencai_conditional_query("可转债；债券余额；转股溢价率")
client = lh.BridgeClient()  # 构造不联网；业务调用前自动校验服务健康和协议。
# 显式调用时只生成文件，不启动QMT；目标须为新建或空目录。
# directory = lh.export_qmt_bridge("./qmt_bridge")
```

迁移、异常行为、数据库参数及文档变更见 [本次迁移说明](docs/migration-openapi-bridge.md)。
桥接订单默认 `dry_run=True`，模板账户及真实报单默认关闭；不包含撤单、条件单或算法单。

## AI 代码生成 Skill

仓库新增了可下载的 [`yuhanbolh-codegen`](skill/README.md) Skill，供 Codex 等支持
`SKILL.md` 的 AI 工具根据真实公开 API 生成、修改和排查 `yuhanbolh` 调用代码。
它覆盖行情与问财选股、可转债、技术指标、SQLite 数据处理、标准大QMT桥接以及
xtquant/MT5 辅助程序，并会提示凭据、数据库写入和真实交易等副作用边界。

### 下载与安装

1. 下载 [`yuhanbolh-codegen.zip`](skill/yuhanbolh-codegen.zip)，并可使用
   [`SHA-256 校验文件`](skill/yuhanbolh-codegen.zip.sha256) 验证完整性。
2. 解压后保留完整的 `yuhanbolh-codegen` 文件夹，确保其根目录直接包含
   `SKILL.md`、`references/`、`scripts/` 和 `agents/`。
3. Codex 用户将该文件夹复制到 `$CODEX_HOME/skills/`；未设置 `CODEX_HOME` 时，
   使用用户目录下的 `.codex/skills/`。复制后开启新会话确认 Skill 已被发现。

调用示例：

```text
使用 $yuhanbolh-codegen，读取本地 CSV 的日线数据，调用 MA 和 RSI，按日期对齐结果并保存到新的 CSV，添加中文注释。
```

Skill 的 API 参考基线为 `yuhanbolh 0.6.5`。实际生成代码前应优先核对目标机器的
已安装版本；Skill 不会自动安装依赖、配置账户、写入业务数据库或执行真实交易。
完整安装说明、文件结构和更多调用示例见 [Skill 使用说明](skill/README.md)。

## 👨‍💻 作者信息

**余汉波** - 编程爱好者-量化交易和效率工具开发

- **GitHub**: [@yuhanbo758](https://github.com/yuhanbo758)

- **Email**: yuhanbo@sanrenjz.com

- **Website**: [三人聚智](https://www.sanrenjz.com)

## 🌐 相关链接

- 🏠 [项目主页](https://www.sanrenjz.com)

- 📚 [在线文档](https://docs.sanrenjz.com/yuhanbolh/about)（yuhanbolh 函数及使用文档）

- 📥 [GitHub 文档下载](https://github.com/yuhanbo758/article-code/tree/main/docs/yuhanbolh)

- 🛒 [插件商店](https://shop.sanrenjz.com)（个人开发的所有程序，包括开源和不开源）


## 联系我们

[联系我们 - 三人聚智-余汉波](https://www.sanrenjz.com/contact_us/)

python 程序管理工具下载：[sanrenjz - 三人聚智-余汉波](https://www.sanrenjz.com/sanrenjz/)

效率工具程序管理下载：[sanrenjz-tools - 三人聚智-余汉波](https://www.sanrenjz.com/sanrenjz-tools/)

智能codebot下载：[sanrenjz-codebot - 三人聚智-余汉波](https://www.sanrenjz.com/sanrenjz-codebot/)

![三码合一](https://gdsx.sanrenjz.com/image/sanrenjz_yuhanbolh_yuhanbo758.png?imageSlim&t=1ab9b82c-e220-8022-beff-e265a194292a)

![余汉波打赏码](https://gdsx.sanrenjz.com/image/%E6%89%93%E8%B5%8F%E7%A0%81%E5%90%88%E4%B8%80.png?imageSlim)

## 🙏 致谢

感谢所有为本项目贡献代码和想法的开发者们！

---
**⭐ 如果这个项目对您有帮助，请给它一个 Star！**
