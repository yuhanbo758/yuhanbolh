"""东方财富行情适配：按字段键映射，只有完整分页成功才返回数据。"""

import json

import pandas as pd
import requests


def _fetch(url, market, sort, mapping):
    """以 total 校验完整性；空响应、重复页、字段结构错误均终止。"""
    rows, seen = [], set()
    total = None
    for page in range(1, 10001):
        params = {
            "pn": str(page),
            "pz": "100",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": sort,
            "fs": market,
            "fields": ",".join(mapping),
        }
        try:
            response = requests.get(url, timeout=15, params=params)
            response.raise_for_status()
            payload = response.json()
            data = payload["data"]
            count = int(data["total"])
            batch = data["diff"]
            if isinstance(batch, dict):
                batch = list(batch.values())
            if (
                not isinstance(batch, list)
                or count < 0
                or any(not isinstance(row, dict) for row in batch)
            ):
                raise ValueError
        except (requests.RequestException, KeyError, TypeError, ValueError):
            raise RuntimeError("东方财富请求失败或响应结构无效，未返回部分数据") from None
        if total is not None and count != total:
            raise RuntimeError("东方财富分页总数变化，请重新查询")
        total = count
        fingerprint = json.dumps(batch, sort_keys=True)
        if batch and fingerprint in seen:
            raise RuntimeError("东方财富重复分页，数据不完整")
        seen.add(fingerprint)
        if any(set(mapping) - set(row) for row in batch):
            raise RuntimeError("东方财富响应缺少预期字段")
        rows.extend(batch)
        if len(rows) > total:
            raise RuntimeError("东方财富行数超过声明总数")
        if len(rows) == total:
            frame = pd.DataFrame(rows, columns=list(mapping)).rename(columns=mapping)
            # 行内空值允许保留；证券代码必须是可用字符串，且保留前导零。
            code = mapping["f12"]
            if not frame.empty and frame[code].isna().any():
                raise RuntimeError("东方财富包含空证券代码")
            strings = {
                c
                for c in mapping.values()
                if any(word in c for word in ("代码", "名称", "日期", "转股日"))
            }
            for column in frame:
                if column in strings:
                    frame[column] = frame[column].astype("string")
                else:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame.insert(0, "序号", range(1, len(frame) + 1))
            return frame
        if not batch:
            raise RuntimeError("东方财富分页提前结束，数据不完整")
    raise RuntimeError("东方财富分页超过安全上限")


def stock_zh_a_spot_em() -> pd.DataFrame:
    """获取沪深京A股行情，返回原有23列；超时/分页失败抛出 RuntimeError。"""
    return _fetch(
        "https://82.push2.eastmoney.com/api/qt/clist/get",
        "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "f3",
        STOCK_FIELDS,
    )


def bond_cov_comparison() -> pd.DataFrame:
    """获取可转债比价表，返回原有20列；代码为字符串，数值缺失为NaN。"""
    return _fetch(
        "https://16.push2.eastmoney.com/api/qt/clist/get", "b:MK0354", "f243", BOND_FIELDS
    )


STOCK_FIELDS = {
    "f12": "代码",
    "f14": "名称",
    "f2": "最新价",
    "f3": "涨跌幅",
    "f4": "涨跌额",
    "f5": "成交量",
    "f6": "成交额",
    "f7": "振幅",
    "f15": "最高",
    "f16": "最低",
    "f17": "今开",
    "f18": "昨收",
    "f10": "量比",
    "f8": "换手率",
    "f9": "市盈率-动态",
    "f23": "市净率",
    "f20": "总市值",
    "f21": "流通市值",
    "f22": "涨速",
    "f11": "5分钟涨跌",
    "f24": "60日涨跌幅",
    "f25": "年初至今涨跌幅",
}
BOND_FIELDS = {
    "f12": "转债代码",
    "f14": "转债名称",
    "f2": "转债最新价",
    "f3": "转债涨跌幅",
    "f232": "正股代码",
    "f234": "正股名称",
    "f229": "正股最新价",
    "f230": "正股涨跌幅",
    "f235": "转股价",
    "f236": "转股价值",
    "f237": "转股溢价率",
    "f238": "纯债溢价率",
    "f239": "回售触发价",
    "f240": "强赎触发价",
    "f241": "到期赎回价",
    "f227": "纯债价值",
    "f242": "开始转股日",
    "f26": "上市日期",
    "f243": "申购日期",
}
