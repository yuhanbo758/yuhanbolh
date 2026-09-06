"""问财 OpenAPI 查询；导入不联网，配置仅从环境变量读取。"""

import json
import math
import os
import re
import secrets
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing

import pandas as pd

API_URL = "https://openapi.iwencai.com/v1/query2data"
QUERY_TYPES = {"stock": "A股", "fund": "基金", "conbond": "可转债", "usstock": "美股"}
BOND_COLUMNS = [
    "可转债代码",
    "可转债简称",
    "最新价",
    "正股代码",
    "正股简称",
    "纯债价值",
    "期权价值",
    "最新变动后余额",
    "转股溢价率",
    "转股价值",
]
ALIASES = {
    "可转债代码": ("可转债代码", "债券代码", "证券代码"),
    "可转债简称": ("可转债简称", "债券简称", "转债名称", "证券简称"),
    "股票代码": ("股票代码", "证券代码"),
    "股票简称": ("股票简称", "证券简称"),
    "最新价": ("可转债最新价", "债券最新价", "最新价", "现价", "债现价"),
    "正股简称": ("正股简称", "正股名称"),
    "最新变动后余额": ("最新变动后余额", "债券余额", "剩余规模"),
}
NUMERIC_COLUMNS = {
    "最新价",
    "涨跌幅",
    "纯债价值",
    "期权价值",
    "最新变动后余额",
    "转股溢价率",
    "转股价值",
}


class WencaiError(RuntimeError):
    """查询失败或数据不完整；不得将此异常作为空选股结果执行交易。"""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 携带认证头的请求不跟随跳转，防止密钥被转发至其他站点。
        return None


def get_wencai(question, query_type="stock", loop=True, *, page_size=100, timeout=60):
    """按自然语言查询股票/基金/可转债/美股，返回原始业务字段 DataFrame。

    loop=False 只取第一页。成功无匹配返回含证券代码、证券简称的空表；
    配置、HTTP、协议或分页失败抛出 WencaiError，不返回部分数据。
    IWENCAI_API_KEY 必填，IWENCAI_BASE_URL 可指定服务根地址或完整接口。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("查询语句不能为空")
    if query_type not in QUERY_TYPES:
        raise ValueError("query_type 必须为 stock、fund、conbond 或 usstock")
    if type(loop) is not bool or type(page_size) is not int or page_size <= 0:
        raise ValueError("loop 必须为布尔值，page_size 必须为正整数")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout 必须为有限正数")
    key = os.environ.get("IWENCAI_API_KEY", "").strip()
    if not key:
        raise WencaiError("请配置 IWENCAI_API_KEY 后重新启动调用进程")
    url = os.environ.get("IWENCAI_BASE_URL", "").strip().rstrip("/") or API_URL
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise WencaiError("IWENCAI_BASE_URL 必须为无凭据和查询参数的HTTP(S)地址")
    if not url.endswith("/v1/query2data"):
        url += "/v1/query2data"
    query = QUERY_TYPES[query_type] + "；" + question.strip()
    if query_type == "conbond":
        query = query.replace("最新变动后余额", "债券余额")
    opener = urllib.request.build_opener(_NoRedirect())
    rows, fingerprints = [], set()
    total = None
    for page in range(1, 10001):
        trace = secrets.token_hex(32)
        payload = {
            "query": query,
            "page": str(page),
            "limit": str(page_size),
            "is_cache": "1",
            "expand_index": "true",
        }
        headers = {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "X-Claw-Call-Type": "normal",
            "X-Claw-Skill-Id": "hithink-astock-selector",
            "X-Claw-Skill-Version": "1.0.0",
            "X-Claw-Plugin-Id": "none",
            "X-Claw-Plugin-Version": "none",
            "X-Claw-Trace-Id": trace,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            raise WencaiError(f"问财HTTP错误 {status}，trace_id={trace}") from None
        except (OSError, ValueError, urllib.error.URLError):
            # 不转述底层异常、响应正文或 URL，避免服务端回显凭据进入日志。
            raise WencaiError("问财请求或 JSON 解析失败，trace_id=" + trace) from None
        if not isinstance(result, dict) or not isinstance(result.get("datas"), list):
            raise WencaiError("问财响应缺少 datas，trace_id=" + trace)
        if (
            result.get("error")
            or result.get("success") is False
            or result.get("status") in ("error", "failed")
        ):
            raise WencaiError("问财网关报告失败，trace_id=" + trace)
        batch = result["datas"]
        if any(not isinstance(row, dict) for row in batch):
            raise WencaiError("问财 datas 行结构错误，trace_id=" + trace)
        try:
            count = int(result["code_count"])
            if count < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise WencaiError("问财响应缺少有效 code_count，无法确认完整性") from None
        if total is not None and count != total:
            raise WencaiError("问财分页总数变化，请重新查询")
        total = count
        fingerprint = json.dumps(batch, sort_keys=True, ensure_ascii=False)
        if batch and fingerprint in fingerprints:
            raise WencaiError("问财返回重复分页，停止获取")
        fingerprints.add(fingerprint)
        rows.extend(batch)
        if len(rows) > total:
            raise WencaiError("问财返回行数超过声明总数")
        if not loop or len(rows) == total:
            return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["证券代码", "证券简称"])
        if not batch:
            raise WencaiError("问财分页提前结束，数据不完整")
    raise WencaiError("问财分页超过安全上限")


def _normalize(data, columns):
    """按精确别名归一日期字段；不使用模糊包含匹配，避免误取正股字段。"""
    normalized = [
        (c, re.sub(r"\[[^\]]*\]", "", str(c)).split("@")[-1].strip()) for c in data.columns
    ]
    result = pd.DataFrame(index=data.index)
    for target in columns:
        matches = [
            c for alias in ALIASES.get(target, (target,)) for c, name in normalized if name == alias
        ]
        result[target] = (
            data[matches[0]] if matches else pd.Series(index=data.index, dtype="object")
        )
        if target in NUMERIC_COLUMNS:
            # 余额统一以亿元计；其他数值的百分号保留百分数口径，不除以100。
            values = result[target].astype("string").str.replace(",", "", regex=False)
            if target == "最新变动后余额":
                factors = values.map(lambda x: 0.0001 if pd.notna(x) and "万" in x else 1.0)
            else:
                factors = 1.0
            values = values.str.replace(r"[%亿元万]", "", regex=True)
            result[target] = pd.to_numeric(values, errors="coerce") * factors
    # 身份字段不可缺失；指标缺失可以保留NaN，但不能生成无法识别证券的数据。
    for code_column in ("可转债代码", "股票代码"):
        if code_column in result and not result.empty:
            if (
                result[code_column].isna().any()
                or result[code_column].astype("string").str.strip().eq("").any()
            ):
                raise WencaiError("问财结果缺少有效" + code_column)
    return result


def wencai_conditional_query(query, fetch_all=True, page_size=100, timeout=60):
    """获取可转债策略十列数据，余额单位亿元；无匹配返回固定列空表。"""
    return _normalize(
        get_wencai(query, "conbond", fetch_all, page_size=page_size, timeout=timeout), BOND_COLUMNS
    )


def get_satisfy_redemption(query):
    """获取强赎条件查询的六列数据；查询失败抛出 WencaiError。"""
    return _normalize(get_wencai(query, "conbond"), BOND_COLUMNS[:5] + ["强赎天计数"])


def get_clean_data(question):
    """获取并清洗可转债十二列数据，保留满足强赎及强赎天计数。"""
    columns = BOND_COLUMNS[:2] + ["涨跌幅"] + BOND_COLUMNS[2:9] + ["满足强赎", "强赎天计数"]
    return _normalize(get_wencai(question, "conbond"), columns)


def wencai_conditional_query_nz100(query, db_path=None):
    """查询美股成分股；仅显式传入 db_path 时替换 nasdaq_100 表。

    返回股票代码、股票简称、指数、价值代码和 mt5代码；异常不会写库。
    """
    data = _normalize(get_wencai(query, "usstock"), ["股票代码", "股票简称", "指数"])
    codes = data["股票代码"].astype("string")
    data["价值代码"] = codes.str.replace(r"\.O$", "", regex=True)
    data["mt5代码"] = codes.str.replace(r"\.O$", ".NAS", regex=True)
    if db_path is not None:
        with closing(sqlite3.connect(db_path)) as conn, conn:
            data.to_sql("nasdaq_100", conn, if_exists="replace", index=False)
    return data
