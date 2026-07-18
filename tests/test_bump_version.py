"""自动版本递增脚本的单元测试。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


# 直接从脚本路径加载模块，避免把发布辅助脚本打进 yuhanbolh 安装包。
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bump_version.py"
SPEC = importlib.util.spec_from_file_location("bump_version", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载版本脚本：{SCRIPT_PATH}")
BUMP_VERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUMP_VERSION)


class IncrementVersionTests(unittest.TestCase):
    """验证普通递增、补丁进位及主版本进位。"""

    def test_regular_increment(self) -> None:
        self.assertEqual(BUMP_VERSION.increment_version("0.6.8"), "0.6.9")

    def test_patch_carry(self) -> None:
        self.assertEqual(BUMP_VERSION.increment_version("0.6.9"), "0.7.0")

    def test_major_carry(self) -> None:
        self.assertEqual(BUMP_VERSION.increment_version("0.9.9"), "1.0.0")

    def test_invalid_version(self) -> None:
        with self.assertRaises(ValueError):
            BUMP_VERSION.increment_version("0.6")


class SetupFileTests(unittest.TestCase):
    """验证脚本只修改 setup.py 中的目标版本字段。"""

    def test_read_and_write_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            setup_file = Path(temporary_directory) / "setup.py"
            setup_file.write_text(
                "from setuptools import setup\nsetup(name='demo', version='0.6.1')\n",
                encoding="utf-8",
            )

            self.assertEqual(BUMP_VERSION.read_version(setup_file), "0.6.1")
            BUMP_VERSION.write_version(setup_file, "0.6.2")
            self.assertEqual(BUMP_VERSION.read_version(setup_file), "0.6.2")


if __name__ == "__main__":
    unittest.main()

