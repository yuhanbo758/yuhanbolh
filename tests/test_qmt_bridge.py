"""从实际打包资源导出桥接并执行来源隔离测试，不连接真实QMT。"""

import ast
import subprocess
import sys
import tempfile
import unittest

from yuhanbolh.qmt_bridge import export_qmt_bridge


class BridgePackageTests(unittest.TestCase):
    def test_export_encoding_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = export_qmt_bridge(directory)
            raw = (output / "qmt_bridge_server.py").read_bytes()
            source = raw.decode("gbk")
            self.assertEqual(raw, source.encode("gbk"))
            ast.parse(source, feature_version=(3, 6))
            self.assertIn("BRIDGE_ALLOW_LIVE_ORDERS = False", source)
            with self.assertRaises(FileExistsError):
                export_qmt_bridge(directory)

    def test_exported_offline_suite_and_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            output = export_qmt_bridge(directory)
            for command in (
                ["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
                ["validate_source.py"],
            ):
                result = subprocess.run(
                    [sys.executable, "-B", *command], cwd=output, capture_output=True, timeout=180
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stdout.decode("utf-8", errors="replace")
                    + result.stderr.decode("utf-8", errors="replace"),
                )


if __name__ == "__main__":
    unittest.main()
