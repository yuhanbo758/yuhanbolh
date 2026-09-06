"""字段顺序、分页完整性、强赎状态及显式数据库边界回归。"""

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from yuhanbolh import akshare_data as a
from yuhanbolh import edit_akshare as e


class AdapterTests(unittest.TestCase):
    def test_reordered_fields_and_missing_values(self):
        for method, fields in (
            (e.stock_zh_a_spot_em, e.STOCK_FIELDS),
            (e.bond_cov_comparison, e.BOND_FIELDS),
        ):
            row = {key: "1" for key in reversed(fields)}
            row.update(f12="001234", f2=None, f14="示例")
            response = Mock()
            response.json.return_value = {"data": {"total": 1, "diff": [row]}}
            with patch.object(e.requests, "get", return_value=response):
                data = method()
            self.assertEqual(list(data.columns), ["序号", *fields.values()])
            self.assertEqual(data.iloc[0][fields["f12"]], "001234")
            self.assertTrue(pd.isna(data.iloc[0][fields["f2"]]))

    def test_pagination_failure_never_returns_partial_data(self):
        first = Mock()
        first.json.return_value = {
            "data": {"total": 2, "diff": [{key: "1" for key in e.STOCK_FIELDS}]}
        }
        with patch.object(e.requests, "get", side_effect=[first, e.requests.Timeout()]):
            with self.assertRaises(RuntimeError):
                e.stock_zh_a_spot_em()

    def test_empty_and_invalid_schema(self):
        for payload, valid in (
            ({"data": {"total": 0, "diff": []}}, True),
            ({"data": None}, False),
            ({"data": {"total": 1, "diff": [{"f12": "1"}]}}, False),
        ):
            r = Mock()
            r.json.return_value = payload
            with patch.object(e.requests, "get", return_value=r):
                if valid:
                    self.assertTrue(e.stock_zh_a_spot_em().empty)
                else:
                    with self.assertRaises(RuntimeError):
                        e.stock_zh_a_spot_em()

    def test_redemption_status_count_and_no_default_write(self):
        data = pd.DataFrame(
            {
                "代码": [110001, "123001.SZ", "110002"],
                "名称": ["a", "b", "c"],
                "现价": [101, 102, 103],
                "正股代码": ["1"] * 3,
                "正股名称": ["x"] * 3,
                "强赎状态": ["已公告强赎", "公告要强赎", None],
                "强赎天计数": ["15/30", "16/30", None],
            }
        )
        with (
            patch.object(a, "akshare_convertible_bond", return_value=data),
            patch.object(a.sqlite3, "connect") as connect,
        ):
            result = a.filter_bond_cb_redeem_data_and_save_to_db()
            connect.assert_not_called()
        self.assertEqual(result["可转债代码"].tolist(), ["110001.SH", "123001.SZ"])
        self.assertEqual(result.iloc[0]["强赎天计数"], "15/30")
        self.assertEqual(result.iloc[0]["强赎状态"], "已公告强赎")


if __name__ == "__main__":
    unittest.main()
