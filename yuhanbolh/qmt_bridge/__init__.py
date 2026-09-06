"""大QMT标准桥接客户端与独立部署模板；导入无网络和交易副作用。"""

from importlib.resources import files
from pathlib import Path

from .client import BridgeClient, BridgeError

__all__ = ["BridgeClient", "BridgeError", "export_qmt_bridge"]


def export_qmt_bridge(output_dir):
    """导出独立桥接到新建或空目录，返回绝对 Path。

    服务端输出为真实GBK字节，其他源码和文档为UTF-8。只生成文件，
    不启动服务、不配置账户、不创建业务数据库；非空目录拒绝覆盖。
    """
    target = Path(output_dir).expanduser().absolute()
    if target.is_symlink():
        raise ValueError("导出目录不能是符号链接")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError("导出目标必须为新建或空目录")
    resources = files(__package__)
    contents = {"bridge_client.py": resources.joinpath("client.py").read_bytes()}
    for resource in resources.joinpath("templates").iterdir():
        if not resource.name.endswith(".tmpl"):
            continue
        name = resource.name[:-5]
        text = resource.read_text(encoding="utf-8")
        encoding = "gbk" if name == "qmt_bridge_server.py" else "utf-8"
        raw = text.encode(encoding)
        if raw.decode(encoding) != text:
            raise ValueError("桥接模板编码往返校验失败")
        if name.endswith(".py"):
            compile(raw, name, "exec")  # 静态检查，不执行服务端生命周期。
        contents[name] = raw
    target.mkdir(parents=True, exist_ok=True)
    # 使用独占创建，即使校验后发生并发写入，也不会覆盖已有同名文件。
    for name, raw in contents.items():
        with (target / name).open("xb") as handle:
            handle.write(raw)
    return target.resolve()
