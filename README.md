
股票量化代码，包括通过qmt、同花顺问题和东财api等获取金融数据，以及处理量化这些数据。

查看库中的函数文档，请前往：https://wd.sanrenjz.com/yuhanbolh/about

代码是从多个渠道获取金融数据，在运行过程中提示没有什么模块，就用pip安装什么模块。
但迅投的xtdata和xttrade是不能通过pip安装的，需要把官网http://dict.thinktrader.net/nativeApi/download_xtquant.html 下载，然后放到python的...\Lib\site-packages路径中。

# 需要证券开户开QMT的可以联系我，微信：yuhanbo758


# 联系我们

主站：[三人聚智-余汉波 - QMT量化、效率工具和财经知识的搬运工](https://www.sanrenjz.com/)

程序小店（个人开发的所有程序，包括开源和不开源）：[首页 | 三人聚智-余汉波程序小店](https://jy.sanrenjz.com/)

文档站点（财经、代码和库文档等）：[余汉波 文档 | 财经、python与效率工具的知识搬运工](https://wd.sanrenjz.com/)

python 程序管理工具下载：[sanrenjz - 三人聚智-余汉波](https://www.sanrenjz.com/sanrenjz/)

![三码合一](https://gdsx.sanrenjz.com/image/sanrenjz_yuhanbolh_yuhanbo758.png?imageSlim&t=1ab9b82c-e220-8022-beff-e265a194292a)

![余汉波打赏码](https://gdsx.sanrenjz.com/PicGo/%E6%89%93%E8%B5%8F%E7%A0%81500.png)



schedule
akshare
scipy
yfinance
pywencai
baostock
pytdx
MetaTrader5
yuhanbolh

## 安装

```powershell
pip install --upgrade yuhanbolh
```

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
