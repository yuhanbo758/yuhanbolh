"""问财协议、字段映射和轮动失败边界；全部HTTP为模拟响应。"""

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from yuhanbolh import wencai as w


def response(rows, count=None):
    return io.BytesIO(
        json.dumps({"datas": rows, "code_count": len(rows) if count is None else count}).encode()
    )


class WencaiTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ, {"IWENCAI_API_KEY": "fake-secret", "IWENCAI_BASE_URL": ""}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.opener = Mock()
        self.patch = patch.object(w.urllib.request, "build_opener", return_value=self.opener)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_types_and_protocol(self):
        for kind, label in w.QUERY_TYPES.items():
            self.opener.open.return_value = response([])
            result = w.get_wencai("测试", kind)
            self.assertEqual(list(result.columns), ["证券代码", "证券简称"])
            req = self.opener.open.call_args.args[0]
            body = json.loads(req.data)
            self.assertTrue(body["query"].startswith(label + "；"))
            self.assertNotIn("query_type", body)
            self.assertEqual(len(req.get_header("X-claw-trace-id")), 64)

    def test_parameter_gateway_and_identity_failures(self):
        for kwargs in (
            {"query_type": "invalid"},
            {"page_size": 0},
            {"timeout": float("inf")},
            {"loop": 1},
        ):
            with self.assertRaises(ValueError):
                w.get_wencai("测试", **kwargs)
        self.opener.open.return_value = io.BytesIO(
            b'{"datas": [], "code_count": 0, "success": false}'
        )
        with self.assertRaises(w.WencaiError):
            w.get_wencai("测试")
        self.opener.open.return_value = response([{"正股代码": "600001", "最新价": 10}])
        with self.assertRaises(w.WencaiError):
            w.wencai_conditional_query("可转债")
        with patch.dict(os.environ, {"IWENCAI_BASE_URL": "https://fake-secret@example.com"}):
            with self.assertRaises(w.WencaiError) as error:
                w.get_wencai("测试")
            self.assertNotIn("fake-secret", str(error.exception))

    def test_pagination_and_loop_false(self):
        self.opener.open.side_effect = [
            response([{"证券代码": "001"}], 2),
            response([{"证券代码": "002"}], 2),
        ]
        self.assertEqual(len(w.get_wencai("股票", page_size=1)), 2)
        self.opener.open.side_effect = [response([{"证券代码": "001"}], 2)]
        self.assertEqual(len(w.get_wencai("股票", loop=False)), 1)

    def test_incomplete_repeated_changed_and_invalid_pages(self):
        for pages in (
            [response([{"证券代码": "1"}], 2), response([], 2)],
            [response([{"证券代码": "1"}], 2), response([{"证券代码": "1"}], 2)],
            [response([{"证券代码": "1"}], 2), response([{"证券代码": "2"}], 3)],
            [io.BytesIO(b'{"datas": []}')],
            [io.BytesIO(b"{}")],
            [io.BytesIO(b"invalid")],
        ):
            self.opener.open.side_effect = pages
            with self.assertRaises(w.WencaiError):
                w.get_wencai("测试")

    def test_errors_do_not_leak_credentials(self):
        self.opener.open.side_effect = OSError("fake-secret")
        with self.assertRaises(w.WencaiError) as error:
            w.get_wencai("测试")
        self.assertNotIn("fake-secret", str(error.exception))
        with patch.dict(os.environ, {"IWENCAI_API_KEY": ""}):
            with self.assertRaises(w.WencaiError):
                w.get_wencai("测试")

    def test_bond_aliases_numbers_and_underlying_exclusion(self):
        self.opener.open.return_value = response(
            [
                {
                    "证券代码": "110001",
                    "证券简称": "示例债",
                    "正股代码": "600001",
                    "正股最新价": "999",
                    "最新价[20260904]": "101.5",
                    "债券余额": "20,000万元",
                    "转股溢价率": "15%",
                }
            ]
        )
        result = w.wencai_conditional_query("最新变动后余额")
        self.assertEqual(list(result.columns), w.BOND_COLUMNS)
        self.assertEqual(result.iloc[0]["可转债代码"], "110001")
        self.assertEqual(result.iloc[0]["最新价"], 101.5)
        self.assertEqual(result.iloc[0]["最新变动后余额"], 2)
        self.assertEqual(result.iloc[0]["转股溢价率"], 15)
        body = json.loads(self.opener.open.call_args.args[0].data)
        self.assertNotIn("最新变动后余额", body["query"])

    def test_nasdaq_no_implicit_database_and_explicit_save(self):
        with patch.object(
            w, "get_wencai", return_value=pd.DataFrame([{"证券代码": "AAPL.O", "证券简称": "示例"}])
        ):
            with patch.object(w.sqlite3, "connect") as connect:
                data = w.wencai_conditional_query_nz100("纳斯达克100")
                connect.assert_not_called()
            self.assertEqual(data.iloc[0]["mt5代码"], "AAPL.NAS")
            with tempfile.TemporaryDirectory() as directory:
                db = Path(directory) / "test.db"
                w.wencai_conditional_query_nz100("纳斯达克100", db)
                with closing(sqlite3.connect(db)) as conn:
                    self.assertEqual(
                        conn.execute("select count(*) from nasdaq_100").fetchone()[0], 1
                    )

    def test_rotation_uses_named_code_and_failure_skips_writes(self):
        from yuhanbolh.process_data import portfolio_rotation, process_scheduled_tasks

        with patch.object(
            w,
            "get_wencai",
            return_value=pd.DataFrame([{"正股代码": "wrong", "证券代码": "110001.SH"}]),
        ):
            self.assertEqual(
                portfolio_rotation("转债", 10, "测试").iloc[0]["证券代码"], "110001.SH"
            )
        for value in (w.WencaiError("失败"), pd.DataFrame(columns=["证券代码"])):
            cursor, conn = Mock(), Mock()
            kwargs = (
                {"side_effect": value} if isinstance(value, Exception) else {"return_value": value}
            )
            with patch.object(w, "get_wencai", **kwargs):
                process_scheduled_tasks([(1, "", 0, 10, 1, "策略", "转债", "")], cursor, conn)
            cursor.execute.assert_not_called()
            conn.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
