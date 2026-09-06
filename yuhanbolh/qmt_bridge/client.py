"""通用大QMT桥接标准库客户端；不自动重试，不在导入时访问网络。"""
import json
import math
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:1693"
DEFAULT_TOKEN = ""  # 与服务端一致；不要放入URL、命令行或日志。
DEFAULT_TIMEOUT = 75.0  # 大于服务端60秒RPC等待时间。


class BridgeError(RuntimeError):
    """HTTP、业务或协议错误；订单报错不代表没有提交。"""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁止跳转，避免令牌泄露及POST被隐式改为GET。"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BridgeClient:
    """方法返回完整JSON信封，调用者通过response['data']取得业务内容。"""
    def __init__(self, base_url=DEFAULT_BASE_URL, token=DEFAULT_TOKEN,
                 timeout=DEFAULT_TIMEOUT):
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc
                or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError("base_url必须是无凭据、查询和路径的HTTP地址")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 60:
            raise ValueError("客户端timeout必须大于服务端RPC的60秒")
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _open(self, path, params=None, payload=None, timeout=None):
        # 不实现自动重试；尤其POST超时后必须先核对原请求号。
        params = params or {}
        query = {}
        for key, value in params.items():
            if value is not None:
                if isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, (list, tuple)):
                    value = ",".join(str(item) for item in value)
                query[key] = value
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"Accept": "application/json"}
        if self.token:
            headers["X-QMT-Bridge-Token"] = self.token
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(url, data=body, headers=headers)
        try:
            return self._opener.open(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            code = exc.code
            message = ""
            try:
                error = json.loads(exc.read(65536).decode("utf-8"))
                # 健康校验前不信任目标身份；已确认产品的业务错误只取message。
                if path != "/health" and isinstance(error, dict):
                    message = str(error.get("message", ""))[:1000]
                    if self.token:
                        message = message.replace(self.token, "[redacted]")
            except (OSError, ValueError):
                pass
            finally:
                exc.close()
            raise BridgeError(f"HTTP错误 {code}：{message}；若为报单，请核对原请求号") from None
        except (OSError, urllib.error.URLError):
            raise BridgeError("连接失败或超时；若为报单，结果未知，禁止盲目重报") from None

    def _request(self, path, params=None, payload=None, timeout=None):
        # 每个业务请求先确认目标产品及所需能力，不能只依赖调用者记得调用health。
        if path != "/health":
            capability = {"/tick": "market_data", "/market": "market_data",
                          "/history_data": "market_data", "/trade/order": "direct_order",
                          "/trade/ledger": "trade_ledger", "/events": "events_long_poll"}.get(path, "account_query")
            self.health([capability])
        try:
            with self._open(path, params, payload, timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (ValueError, UnicodeError):
            raise BridgeError("响应不是有效UTF-8 JSON") from None
        except OSError:
            raise BridgeError("读取响应中断；报单结果可能未知") from None
        if not isinstance(result, dict) or result.get("status") != "success":
            raise BridgeError("服务端返回业务错误或无效状态；请检查服务端记录")
        return result

    def health(self, required_capabilities=(), require_ready=True):
        """验证产品/API及所需能力；require_ready=False可仅检查服务身份。"""
        result = self._request("/health")
        if result.get("product") != "qmt_bridge_standard" or result.get("api_version") != 1:
            raise BridgeError("产品或API版本不匹配")
        capabilities = result.get("capabilities", [])
        for capability in required_capabilities:
            if capability not in capabilities or (isinstance(capabilities, dict)
                                                   and not capabilities[capability]):
                raise BridgeError("服务缺少所需能力：" + capability)
        if require_ready and result.get("runtime_ready") is not True:
            raise BridgeError("大QMT运行环境尚未就绪")
        if (require_ready and result.get("rpc_ready") is False
                and set(required_capabilities).intersection(("market_data", "account_query", "direct_order"))):
            raise BridgeError("HTTP已启动，但QMT定时回调尚未就绪或已停止，请检查timer_count和last_timer_at")
        return result

    def tick(self, stock_code):
        """查询指定证券Tick，返回完整JSON信封；不订阅、不下单。"""
        return self._request("/tick", {"stock_code": stock_code})

    def market(self, stock_code):
        """查询证券标准行情，返回完整JSON信封及服务端行情时间。"""
        return self._request("/market", {"stock_code": stock_code})

    def history_data(self, stock_code, period="1d", count=60, fields=None,
                     start_time=None, end_time=None, dividend_type="none",
                     fill_data=False, subscribe=False):
        """读取历史K线；默认不补数据、不订阅，缺失或陈旧数据由调用者核对。"""
        return self._request("/history_data", dict(
            stock_code=stock_code, period=period, count=count, fields=fields,
            start_time=start_time, end_time=end_time, dividend_type=dividend_type,
            fill_data=fill_data, subscribe=subscribe))

    def assets(self, account_type="all"):
        """查询已启用账户资产；账户只能用normal/credit/all别名。"""
        return self._request("/assets", {"account_type": account_type})

    def positions(self, account_type="all"):
        """查询已启用账户持仓，不修改账户或策略归属。"""
        return self._request("/positions", {"account_type": account_type})

    def orders(self, account_type="all", request_id=None):
        """有request_id时查询持久化请求，否则查询柜台委托；用于未知结果对账。"""
        return self._request("/orders", {"account_type": account_type, "request_id": request_id})

    def trades(self, account_type="all"):
        """查询柜台成交记录，查询恢复不等于收到原始成交回调。"""
        return self._request("/trades", {"account_type": account_type})

    def order(self, stock_code, account_type, action, price, volume,
              user_order_id, strategy_name="http_bridge", op_type=None,
              dry_run=True, confirm_live_order=None):
        """默认只预演；真实报单需LIVE_ORDER及服务端许可，超时不得换号重报。

        user_order_id必须稳定唯一；信用账户必须显式指定op_type。
        返回提交信封不等于柜台成交，应查询orders/trades/ledger核对。
        """
        if not isinstance(user_order_id, str) or not user_order_id.strip():
            raise ValueError("必须提供稳定且非空的user_order_id")
        if account_type not in ("normal", "credit") or action not in ("BUY", "SELL"):
            raise ValueError("账户或方向无效")
        if account_type == "credit" and op_type is None:
            raise ValueError("信用账户必须显式提供op_type")
        if not strategy_name or len(strategy_name.encode("gbk")) > 20:
            raise ValueError("策略名称必须非空且不超过20个GBK字节")
        if type(dry_run) is not bool:
            raise ValueError("dry_run必须是布尔值")
        if not dry_run and confirm_live_order != "LIVE_ORDER":
            raise ValueError("真实下单必须确认LIVE_ORDER")
        payload = dict(stock_code=stock_code, account_type=account_type,
                       action=action, price=price, volume=volume,
                       user_order_id=user_order_id, strategy_name=strategy_name,
                       dry_run=dry_run)
        if op_type is not None:
            payload["op_type"] = op_type
        if not dry_run:
            payload["confirm_live_order"] = confirm_live_order
        return self._request("/trade/order", payload=payload)

    def ledger(self, request_id=None, account_type="all"):
        """按请求号/账户查询桥接SQLite账本，不创建或迁移外部业务库。"""
        return self._request("/trade/ledger", dict(
            request_id=request_id, account_type=account_type))

    def events(self, after_id=0, event_types=None, timeout=20, stream_id=None):
        """长轮询事件；游标须连同stream_id保存，缺口需通过查询对账。"""
        wait = float(timeout)
        if not math.isfinite(wait) or wait < 0:
            raise ValueError("长轮询timeout必须是有限非负数")
        return self._request("/events", dict(after_id=after_id,
            event_types=event_types, timeout=timeout, stream_id=stream_id),
            timeout=max(self.timeout, wait + 15))

    def events_sse(self, after_id=0, event_types=None, stream_id=None,
                   max_seconds=60, heartbeat=10):
        """有界SSE迭代器；返回原始事件帧，不自动提交订单或持久化消费游标。"""
        """逐条返回{event, id, data}；保留bridge_gap，不自动重连。"""
        duration, beat = float(max_seconds), float(heartbeat)
        if not all(math.isfinite(v) and v > 0 for v in (duration, beat)):
            raise ValueError("max_seconds及heartbeat必须是有限正数")
        self.health(["events_sse"])
        params = dict(after_id=after_id, event_types=event_types,
                      stream_id=stream_id, max_seconds=max_seconds, heartbeat=heartbeat)
        try:
            with self._open("/events/stream", params,
                            timeout=max(self.timeout, duration + 15, beat + 15)) as response:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise BridgeError("响应不是SSE事件流")
                event, event_id, lines = "message", None, []
                for raw_line in response:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line:
                        if lines:
                            yield {"event": event, "id": event_id,
                                   "data": json.loads("\n".join(lines))}
                        event, event_id, lines = "message", None, []
                    elif not line.startswith(":"):
                        key, separator, value = line.partition(":")
                        if value.startswith(" "):
                            value = value[1:]
                        if key == "data":
                            lines.append(value)
                        elif key == "event":
                            event = value
                        elif key == "id":
                            event_id = value
                # 连接结束时丢弃未以空行提交的残帧，避免消费不完整事件。
        except (ValueError, UnicodeError):
            raise BridgeError("SSE包含无效UTF-8或JSON") from None
        except OSError:
            raise BridgeError("SSE连接中断；请用已消费游标重新订阅并检查缺口") from None
