"""只读解析 yuhanbolh 公开 API；不导入包或执行网络、数据库、交易代码。"""

import argparse
import ast
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def read_api(package):
    """按入口导出映射提取签名、文档和实现位置，避免同名函数误路由。"""
    entry = ast.parse((package / "__init__.py").read_text(encoding="utf-8-sig"))
    exports = None
    for node in entry.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_MODULE_EXPORTS"
            for target in node.targets
        ):
            exports = ast.literal_eval(node.value)
    if not isinstance(exports, dict):
        raise ValueError("当前版本没有可静态解析的 _MODULE_EXPORTS；请人工检查包入口")
    records = []
    for module, names in sorted(exports.items()):
        paths = [package / (module + ".py")]
        if module == "qmt_bridge":
            paths = [package / module / "__init__.py", package / module / "client.py"]
        found = set()
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name not in names:
                    continue
                found.add(node.name)
                members = [(node.name, node)]
                if isinstance(node, ast.ClassDef):
                    members += [
                        (node.name + "." + item.name, item)
                        for item in node.body
                        if isinstance(item, ast.FunctionDef)
                        and (not item.name.startswith("_") or item.name == "__init__")
                    ]
                for name, member in members:
                    signature = name
                    if isinstance(member, ast.FunctionDef):
                        signature += "(" + ast.unparse(member.args) + ")"
                    records.append({
                        "module": module, "name": name, "signature": signature,
                        "doc": ast.get_docstring(member) or "无 docstring；需检查源码实现与配套文档。",
                        "source": str(path), "line": member.lineno,
                    })
        missing = set(names) - found
        if missing:
            raise ValueError(f"未解析到公开 API {module}: {sorted(missing)}")
    return records


def main():
    """优先解析显式仓库，否则定位当前解释器的发行包；不修改安装环境。"""
    # Windows 重定向输出时也使用 UTF-8，保持中文接口文档可被下游可靠读取。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="包含 yuhanbolh 子目录的源码仓库")
    parser.add_argument("--name", help="精确公开名称，例如 MA 或 BridgeClient.order")
    parser.add_argument("--module", help="仅输出指定模块")
    args = parser.parse_args()
    try:
        package = args.repo.resolve() / "yuhanbolh" if args.repo else Path(
            distribution("yuhanbolh").locate_file("yuhanbolh")
        )
        records = read_api(package)
        records = [r for r in records if (not args.name or r["name"] == args.name)
                   and (not args.module or r["module"] == args.module)]
        if not records:
            parser.error("未找到匹配的公开 API，请检查名称或版本")
        print(json.dumps(records, ensure_ascii=False, indent=2))
    except (OSError, ValueError, SyntaxError, PackageNotFoundError) as exc:
        parser.exit(1, f"只读 API 检查失败：{exc}\n")


if __name__ == "__main__":
    main()
