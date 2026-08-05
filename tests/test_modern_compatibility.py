"""NumPy 2、pandas 3 与包级延迟导入的回归测试。"""

from __future__ import annotations

import sys
import unittest

import numpy as np
import pandas as pd


class PublicApiTests(unittest.TestCase):
    """验证普通数据分析用户不会被可选交易终端依赖阻断。"""

    def test_package_import_does_not_eagerly_load_xtquant(self) -> None:
        # 清除当前进程可能由其他测试加载的包，确保验证的是首次导入边界。
        for module_name in list(sys.modules):
            if module_name == "yuhanbolh" or module_name.startswith("yuhanbolh."):
                del sys.modules[module_name]

        import yuhanbolh

        self.assertNotIn("xtquant", sys.modules)
        self.assertEqual(set(yuhanbolh.__all__), set(yuhanbolh._EXPORT_MODULE))

    def test_core_indicator_is_available_from_package_root(self) -> None:
        from yuhanbolh import MA

        data = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        result = MA(data, 2)

        self.assertEqual(result.tolist(), [1.5, 2.5])


class PandasCompatibilityTests(unittest.TestCase):
    """覆盖 pandas 3 默认 Copy-on-Write 和严格索引对齐相关逻辑。"""

    def setUp(self) -> None:
        self.market_data = pd.DataFrame(
            {
                "high": np.linspace(10.0, 30.0, 40),
                "low": np.linspace(9.0, 29.0, 40),
                "close": np.linspace(9.5, 29.5, 40),
                "volume": np.arange(1, 41, dtype=float),
            },
            index=pd.date_range("2026-01-01", periods=40, freq="D"),
        )

    def test_adx_preserves_datetime_index(self) -> None:
        from yuhanbolh import ADX

        result = ADX(self.market_data, 14)

        self.assertFalse(result.empty)
        self.assertTrue(result.index.isin(self.market_data.index).all())
        self.assertFalse(result.isna().any())

    def test_common_indicators_return_results(self) -> None:
        from yuhanbolh import EMA, RSI, STOK, VWMA

        self.assertFalse(EMA(self.market_data, 10).empty)
        self.assertFalse(RSI(self.market_data, 14).empty)
        self.assertFalse(STOK(self.market_data, 14, 3, 3).empty)
        self.assertFalse(VWMA(self.market_data, 10).empty)


if __name__ == "__main__":
    unittest.main()
