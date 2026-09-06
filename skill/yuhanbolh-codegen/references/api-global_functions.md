# global_functions API

来源：yuhanbolh 0.6.5 公开导出映射及源码。仅加载所需接口；文档不足时检查对应实现。

## list_dir_contents

```python
list_dir_contents(directory_path)
```

列出指定文件夹中的所有文件和子文件夹。

参数:
    directory_path: 要列出内容的文件夹路径

返回:
    包含三个列表的字典：
    - 'files': 文件名列表
    - 'folders': 子文件夹名列表
    - 'full_paths': 所有项目的完整路径列表

实现：`yuhanbolh/global_functions.py:13`。

## create_account_database

```python
create_account_database(db_data_path)
```

创建账号密码数据库及表，参数为数据库路径

实现：`yuhanbolh/global_functions.py:57`。

## check_account

```python
check_account(column_name, project_name)
```

读取mm.db，查询账号密码，参数分别为列名（指定为username，password），项目名称（mm）。

实现：`yuhanbolh/global_functions.py:86`。

## add_account

```python
add_account(project_name, username, password)
```

向数据库添加账号密码，参数为项目名称，用户名，密码

实现：`yuhanbolh/global_functions.py:120`。

## copy_table_to_mysql

```python
copy_table_to_mysql(sqlite_db_path: str, table_names: list)
```

将数据上传到远程数据库，参数分别是：数据库路径、数据表名称列表

实现：`yuhanbolh/global_functions.py:152`。

## calculate_unhedged_transactions

```python
calculate_unhedged_transactions(db_path, table_names)
```

先买先卖，用于基本技术止盈止损。获取未平仓的持仓数据，即未对冲的买入交易，参数分别是：数据库路径、表名列表

实现：`yuhanbolh/global_functions.py:204`。

## calculate_unhedged_transactions_sbb

```python
calculate_unhedged_transactions_sbb(db_path, table_names)
```

后买先卖，用于网格止盈止损。获取未平仓的持仓数据，即未对冲的买入交易，参数分别是：数据库路径、表名列表
calculate_unhedged_transactions_sbb(db_path, ['实测交易数据'], '实测持仓')

实现：`yuhanbolh/global_functions.py:267`。

## open_positions

```python
open_positions(db_path)
```

策略持仓函数，参数是数据库路径

实现：`yuhanbolh/global_functions.py:559`。

## calculate_stop_profit_loss

```python
calculate_stop_profit_loss(target_strategies, profit_multipliers, loss_multipliers, db_path=None)
```

计算各策略的止盈止损价格并保存到指定数据库。

``db_path`` 过去被误写成未定义的全局变量，函数一经调用就可能触发
``NameError``。现在显式接收路径；保留可选参数形式是为了给旧调用方提供
清晰的迁移错误，而不是继续隐式使用个人电脑路径。

实现：`yuhanbolh/global_functions.py:593`。

## generate_mole_strategy

```python
generate_mole_strategy(db_path, strategies)
```

获取打地鼠策略的持仓数据，参数是数据库路径和策略名称列表

实现：`yuhanbolh/global_functions.py:631`。

## copy_tables

```python
copy_tables(table_names, source_db_path='<USER_PATH>', target_db_path='<USER_PATH>')
```

复制数据库的多个数据表到另一个数据库中，参数为需要复制的表名列表，源数据库路径，目标数据库路径

实现：`yuhanbolh/global_functions.py:660`。

## sync_folders

```python
sync_folders(source, destination)
```

同步文件夹——将源文件夹（量化电脑的共享文件夹）中的文件和子文件夹复制到目标文件夹（本机电脑文件夹）中

实现：`yuhanbolh/global_functions.py:701`。

## get_decimal_places

```python
get_decimal_places(number)
```

获取小数点后的位数

实现：`yuhanbolh/global_functions.py:727`。

## save_to_database

```python
save_to_database(data, db_path, table_name)
```

保存到数据库

实现：`yuhanbolh/global_functions.py:737`。

## get_existing_data

```python
get_existing_data(db_path, table_name)
```

读取数据库数据

实现：`yuhanbolh/global_functions.py:748`。

## process_data

```python
process_data(data, db_path, data_table, top_3_table, price_w, prem_w, size_w, conv_w, opt_w)
```

对可转债数据进行加权

实现：`yuhanbolh/global_functions.py:761`。

## list_dir_recursive

```python
list_dir_recursive(directory_path, file_extension=None, include_subdirs=True)
```

递归列出指定文件夹及其所有子文件夹中的文件。

参数:
    directory_path: 要列出内容的文件夹路径
    file_extension: 可选，文件扩展名过滤器（例如 '.py'）
    include_subdirs: 是否包含子文件夹，默认为True

返回:
    包含所有满足条件文件信息的列表，每个元素是一个字典，包含：
    - 'file_name': 文件名
    - 'file_path': 文件的完整路径
    - 'file_size': 文件大小（字节）
    - 'modified_time': 文件最后修改时间

实现：`yuhanbolh/global_functions.py:797`。

## search_files

```python
search_files(directory_path, search_text, file_extension=None, case_sensitive=False, max_results=100)
```

在指定文件夹及其子文件夹中搜索包含特定文本内容的文件。

参数:
    directory_path: 要搜索的文件夹路径
    search_text: 要搜索的文本内容
    file_extension: 可选，文件扩展名过滤器（例如 '.py'）
    case_sensitive: 是否区分大小写，默认为False
    max_results: 最大结果数，默认为100

返回:
    包含搜索结果的列表，每个元素是一个字典，包含：
    - 'file_path': 文件的完整路径
    - 'line_number': 匹配行的行号
    - 'line_content': 匹配行的内容

实现：`yuhanbolh/global_functions.py:858`。
