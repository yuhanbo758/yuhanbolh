from setuptools import setup, find_packages

setup(
    name='yuhanbolh',
    version='0.2.1',
    packages=find_packages(),
    description='量化投资，数据获取和处理',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yuhanbo758/yuhanbolh',
    author='余汉波',
    author_email='yuhanbo@sanrenjz.com',
    license='MIT',
    install_requires=[
        'BeautifulSoup4',
        'requests',
        'numpy',
        'pandas',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # 根据您的开发状态选择：Alpha/Beta/Stable
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 根据需要修改
)
