"""AKShare 可转债适配；终端交易依赖不参与导入。"""

import sqlite3
from contextlib import closing


def akshare_convertible_bond():
    """返回 AKShare 集思录可转债强赎原始表；网络异常向调用方传播。"""
    import akshare as ak

    return ak.bond_cb_redeem_jsl()


def filter_bond_cb_redeem_data_and_save_to_db(db_path=None):
    """筛选已公告/即将强赎可转债，返回七列数据。

    强赎状态与强赎天计数分别保留；可转债代码加 .SH/.SZ 后缀。
    db_path 默认为 None，仅取数；显式传入时替换“满足赎回可转债”表。
    网络或缺列异常会终止，不把错误当空表写入数据库。
    """
    data = akshare_convertible_bond()
    mapping = {
        "代码": "可转债代码",
        "名称": "可转债简称",
        "现价": "最新价",
        "正股代码": "正股代码",
        "正股名称": "正股简称",
        "强赎状态": "强赎状态",
        "强赎天计数": "强赎天计数",
    }
    missing = set(mapping) - set(data.columns)
    if missing:
        raise RuntimeError("AKShare 强赎数据缺少必要列：" + "、".join(sorted(missing)))
    result = data[list(mapping)].rename(columns=mapping)
    result = result[
        result["强赎状态"].astype("string").str.contains("已公告强赎|公告要强赎", na=False)
    ].copy()
    codes = result["可转债代码"].astype("string").str.replace(r"\.0$", "", regex=True)
    result["可转债代码"] = codes.mask(
        codes.str.fullmatch(r"11\d{4}", na=False), codes + ".SH"
    ).mask(codes.str.fullmatch(r"12\d{4}", na=False), codes + ".SZ")
    if db_path is not None:
        with closing(sqlite3.connect(db_path)) as conn, conn:
            result.to_sql("满足赎回可转债", conn, if_exists="replace", index=False)
    return result.reset_index(drop=True)
