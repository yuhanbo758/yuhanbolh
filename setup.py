from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent

setup(
    name="yuhanbolh",
    version="0.6.4",
    packages=find_packages(),
    description="量化投资，数据获取和处理",
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/yuhanbo758/yuhanbolh",
    author="余汉波",
    author_email="yuhanbo@sanrenjz.com",
    license="MIT",
    install_requires=[
        "akshare>=1.12.10",
        "baostock>=0.7.5",
        "beautifulsoup4>=4.12.0",
        "MetaTrader5>=5.0.45",
        # 仅声明已验证的最低版本，不锁死补丁版本；这样 pip 可以安装
        # NumPy 2.x、pandas 3.x 以及后续兼容版本，避免与用户环境冲突。
        # numexpr 与 Bottleneck 是 pandas 的可选加速依赖；显式声明 pandas 3
        # 要求的最低版本，避免旧 Anaconda 环境残留版本在导入时产生警告。
        "bottleneck>=1.4.2",
        "numpy>=1.26.4",
        "numexpr>=2.10.2",
        "pandas>=2.2.2",
        "pymysql>=1.1.0",
        "pytdx>=1.72",
        "pywencai>=0.3.7",
        "requests>=2.31.0",
        "schedule>=1.2.0",
        "scipy>=1.12.0",
    ],
    extras_require={
        # ipykernel 不是运行本库所必需的依赖，因此仅作为 Notebook 可选项。
        # Spyder 自带的 spyder-kernels 可能要求 ipykernel<7，应使用独立环境。
        "notebook": ["ipykernel>=7.0.0"],
        "dev": ["build>=1.2.0", "ruff>=0.12.0", "twine>=6.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
