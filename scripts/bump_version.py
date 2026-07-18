"""根据指定版本计算下一个十进制版本，并更新 setup.py。

项目采用“每一位逢十进一”的三段式版本规则，例如：
0.6.8 -> 0.6.9 -> 0.7.0，0.9.9 -> 1.0.0。
GitHub Actions 会把 PyPI 当前版本传给本脚本，确保线上版本是发布基准。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# 只匹配 setup() 中的标准三段式版本，避免意外修改其他数字或依赖版本。
VERSION_PATTERN = re.compile(
    r"(?P<prefix>\bversion\s*=\s*['\"])(?P<version>\d+\.\d+\.\d+)(?P<suffix>['\"])",
    re.MULTILINE,
)


def increment_version(version: str) -> str:
    """按每位十进制规则将版本号增加 0.0.1。"""

    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"版本号必须是非负整数三段式格式，实际为：{version}")

    major, minor, patch = map(int, parts)

    # 补丁位先加一，再依次把进位传递给次版本位和主版本位。
    patch += 1
    minor += patch // 10
    patch %= 10
    major += minor // 10
    minor %= 10

    return f"{major}.{minor}.{patch}"


def read_version(setup_file: Path) -> str:
    """读取 setup.py 中唯一的版本号。"""

    content = setup_file.read_text(encoding="utf-8")
    matches = list(VERSION_PATTERN.finditer(content))
    if len(matches) != 1:
        raise RuntimeError(
            f"{setup_file} 中应当恰好存在一个 version='x.y.z'，实际找到 {len(matches)} 个。"
        )
    return matches[0].group("version")


def write_version(setup_file: Path, version: str) -> None:
    """仅替换 setup.py 的版本字段，并保留文件中的其他内容。"""

    content = setup_file.read_text(encoding="utf-8")
    updated, count = VERSION_PATTERN.subn(
        lambda match: f"{match.group('prefix')}{version}{match.group('suffix')}",
        content,
    )
    if count != 1:
        raise RuntimeError(
            f"{setup_file} 中应当恰好更新一个版本字段，实际更新 {count} 个。"
        )
    setup_file.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="递增并写入 yuhanbolh 的版本号")
    parser.add_argument(
        "--base-version",
        help="作为递增基准的版本；GitHub Actions 中应传入 PyPI 当前版本。",
    )
    parser.add_argument(
        "--setup-file",
        type=Path,
        default=project_root / "setup.py",
        help="需要更新的 setup.py 路径。",
    )
    return parser.parse_args()


def main() -> None:
    """计算新版本、更新文件，并把新版本输出给工作流。"""

    args = parse_args()
    setup_file = args.setup_file.resolve()
    base_version = args.base_version or read_version(setup_file)
    next_version = increment_version(base_version)
    write_version(setup_file, next_version)

    # 标准输出只保留版本号，便于 Shell 直接捕获为 NEW_VERSION。
    print(next_version)


if __name__ == "__main__":
    main()

