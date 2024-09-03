import sqlite3
import pymysql
import pandas as pd
import os
import shutil
import requests

# 全局函数



# 读取mm.db，查询账号密码
def check_account(column_name, project_name):
    db_path = r"D:\wenjian\python\data\data\mm.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 使用 column_name 参数构建查询语句
        query = f"""
        SELECT {column_name}
        FROM connect_account_password
        WHERE project_name = ?
        """

        cursor.execute(query, (project_name,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            return result[0]  # 返回查询结果而不是列表
        else:
            return None

    except Exception as e:
        print(f"数据库操作错误：{e}")
        return None
    

# 将数据上传到远程数据库
def copy_table_to_mysql(sqlite_db_path: str, table_names: list):

    host='111.229.252.56'
    user = check_account('username', host)
    password = check_account('password', host)
    print(user, password)

    # 连接到MySQL数据库
    mysql_conn = pymysql.connect(host=host,
                                 user=user,
                                 password=password,
                                 database='financial_data')
    mysql_cursor = mysql_conn.cursor()

    for table_name in table_names:
        try:
            # 连接到SQLite数据库
            sqlite_conn = sqlite3.connect(sqlite_db_path)
            sqlite_cursor = sqlite_conn.cursor()

            # 从SQLite数据库读取数据
            sqlite_cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = sqlite_cursor.fetchall()
            columns = [f"`{column[1]}` {column[2]}" for column in columns_info]

            columns_str = ', '.join(columns)

            sqlite_cursor.execute(f"SELECT * FROM {table_name}")
            sqlite_data = sqlite_cursor.fetchall()

            # 创建MySQL表格（如果不存在）
            create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({columns_str});"
            mysql_cursor.execute(create_table_query)

            # 删除MySQL表中的所有现有记录
            mysql_cursor.execute(f"TRUNCATE TABLE `{table_name}`")

            # 将数据插入到MySQL数据库
            insert_columns_str = ', '.join([f"`{column[1]}`" for column in columns_info])
            insert_query = f"INSERT INTO `{table_name}` ({insert_columns_str}) VALUES ({', '.join(['%s'] * len(columns_info))})"
            mysql_cursor.executemany(insert_query, sqlite_data)

            # 关闭SQLite连接
            sqlite_conn.close()

        except Exception as e:
            print(f"复制{table_name}到MySQL时出错：{str(e)}")
            continue  # 出现错误时继续下一个循环

    # 提交更改并关闭MySQL连接
    mysql_conn.commit()
    mysql_conn.close()


# 先买先卖，用于基本技术止盈止损。获取未平仓的持仓数据，即未对冲的买入交易，参数是：表名列表
def calculate_unhedged_transactions(db_path, table_names):
    try:
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        
        all_remaining_buys = pd.DataFrame()  # 初始化一个空的DataFrame来存储所有表的未对冲买入交易

        for table_name in table_names:
            # 从数据库读取数据到DataFrame
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

            # 获取所有不同的策略名称
            strategies = df['策略名称'].unique()

            # 遍历每个策略
            for strategy_name in strategies:
                # 筛选当前策略的交易记录
                strategy_df = df[df['策略名称'] == strategy_name]

                # 初始化未对冲买入成交数量
                unhedged_transactions = pd.DataFrame()

                # 遍历交易记录
                for _, row in strategy_df.iterrows():
                    if row['买卖'] == 1:  # 买入
                        unhedged_transactions = pd.concat([unhedged_transactions, pd.DataFrame([row])], ignore_index=True)
                    elif row['买卖'] == -1:  # 卖出
                        sell_shares = row['成交数量']
                        for i in unhedged_transactions.index:
                            buy_txn = unhedged_transactions.loc[i]
                            if buy_txn['证券代码'] == row['证券代码']:
                                if buy_txn['成交数量'] <= sell_shares:
                                    sell_shares -= buy_txn['成交数量']
                                    unhedged_transactions.at[i, '成交数量'] = 0  # 对冲完全
                                else:
                                    unhedged_transactions.at[i, '成交数量'] -= sell_shares
                                    break

                # 筛选出未完全对冲的买入交易
                remaining_buys = unhedged_transactions[(unhedged_transactions['成交数量'] > 0) & (unhedged_transactions['买卖'] == 1)]
                
                # 累积所有策略的未对冲买入交易
                all_remaining_buys = pd.concat([all_remaining_buys, remaining_buys], ignore_index=True)

        # 返回数据
        return all_remaining_buys

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 关闭数据库连接
        conn.close()


# 后买先卖，用于网格止盈止损。获取未平仓的持仓数据，即未对冲的买入交易，参数是：表名列表
# calculate_unhedged_transactions_sbb(db_path, ['实测交易数据'], '实测持仓')
def calculate_unhedged_transactions_sbb(db_path, table_names):
    try:
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        
        all_remaining_buys = pd.DataFrame()  # 初始化一个空的DataFrame来存储所有表的未对冲买入交易

        for table_name in table_names:
            # 从数据库读取数据到DataFrame
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

            # 获取所有不同的策略名称
            strategies = df['策略名称'].unique()

            # 遍历每个策略
            for strategy_name in strategies:
                # 筛选当前策略的交易记录
                strategy_df = df[df['策略名称'] == strategy_name]

                # 初始化未对冲买入成交数量
                unhedged_transactions = pd.DataFrame()

                # 遍历交易记录
                for _, row in strategy_df.iterrows():
                    if row['买卖'] == 1:  # 买入
                        unhedged_transactions = pd.concat([unhedged_transactions, pd.DataFrame([row])], ignore_index=True)
                    elif row['买卖'] == -1:  # 卖出
                        sell_shares = row['成交数量']
                        # 逆序遍历未对冲买入交易，实现后买先卖
                        for i in reversed(unhedged_transactions.index):
                            buy_txn = unhedged_transactions.loc[i]
                            if buy_txn['证券代码'] == row['证券代码']:
                                if buy_txn['成交数量'] <= sell_shares:
                                    sell_shares -= buy_txn['成交数量']
                                    unhedged_transactions.at[i, '成交数量'] = 0  # 对冲完全
                                else:
                                    unhedged_transactions.at[i, '成交数量'] -= sell_shares
                                    break

                # 筛选出未完全对冲的买入交易
                remaining_buys = unhedged_transactions[(unhedged_transactions['成交数量'] > 0) & (unhedged_transactions['买卖'] == 1)]
                
                # 累积所有策略的未对冲买入交易
                all_remaining_buys = pd.concat([all_remaining_buys, remaining_buys], ignore_index=True)

        # 返回数据
        return all_remaining_buys

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 关闭数据库连接
        conn.close()


# 策略持仓函数，参数是数据库路径
def open_positions(db_path):
    try:
        # 调用函数并将结果存储在变量中
        # 先买先卖，用于基本技术止盈止损
        df1 = calculate_unhedged_transactions(db_path, ['execute_fund_basics_technical_trade'])

        # 后买先卖，用于网格止盈止损
        df2 = calculate_unhedged_transactions_sbb(db_path, ['execute_fund_grid_trade', 'execute_calmar_ratio_trade', 'execute_sortino_ratio_trade'])

        # 使用 pd.concat 合并两个 DataFrame
        df = pd.concat([df1, df2])

        # 创建数据库连接
        conn = sqlite3.connect(db_path)
        # 将合并后的 DataFrame 保存到数据库中的 open_positions 表中
        df.to_sql('open_positions', conn, if_exists='replace', index=False)
        
        # 返回pd表
        return df
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()



# 读取open_positions，计算止盈和止损价格，并保存到数据表stop_profit_loss，参数是：策略名称列表、止盈倍数列表和止损倍数列表
def calculate_stop_profit_loss(target_strategies, profit_multipliers, loss_multipliers):
    conn = sqlite3.connect(db_path)
    try:
        all_filtered_df = open_positions(db_path)

        for strategy, profit_multiplier, loss_multiplier in zip(target_strategies, profit_multipliers, loss_multipliers):
            df = pd.read_sql(f"SELECT * FROM open_positions WHERE 策略名称 = '{strategy}'", conn)

            df['止盈'] = df['成交均价'].apply(lambda x: round(x * profit_multiplier, get_decimal_places(x)))
            df['止损'] = df['成交均价'].apply(lambda x: round(x * loss_multiplier, get_decimal_places(x)))

            all_filtered_df = pd.concat([all_filtered_df, df])

    
        all_filtered_df.to_sql('stop_profit_loss', conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()


# 获取打地鼠策略的持仓数据
def generate_mole_strategy(db_path, strategies):
    try:
        df = open_positions(db_path)[['策略名称', '委托类型', '证券代码', '成交均价', '成交数量', '成交金额', '买卖']]
        filtered_df = df[df['策略名称'].isin(strategies)]
        grouped_df = filtered_df.groupby('证券代码').agg({
            '策略名称': 'first',
            '委托类型': 'first',
            '成交均价': 'mean',
            '成交数量': 'sum',
            '成交金额': 'sum',
            '买卖': 'sum'
        }).reset_index()
        grouped_df = grouped_df.rename(columns={'成交数量': '持仓数量'})
        # 使用 with 语句管理数据库连接
        with sqlite3.connect(db_path) as conn:
            grouped_df.to_sql('generate_mole_strategy', conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"An error occurred: {e}")


# 复制数据库的多个数据表到另一个数据库中，参数为需要复制的表名列表
def copy_tables(table_names):
    # 连接到 guojin_account.db 数据库
    conn1 = sqlite3.connect('D:\\wenjian\\python\\smart\\data\\guojin_account.db')
    cursor1 = conn1.cursor()

    # 连接到 sync_database.db 数据库
    conn2 = sqlite3.connect(r"D:\wenjian\synkdy\data\sync_database.db")
    cursor2 = conn2.cursor()

    for table_name in table_names:
        # 获取表的创建语句
        cursor1.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_table_query = cursor1.fetchone()[0]

        # 获取表的数据
        cursor1.execute(f"SELECT * FROM {table_name}")
        data = cursor1.fetchall()

        # 如果 sync_database.db 数据库中已经存在同名表，先删除它
        cursor2.execute(f"DROP TABLE IF EXISTS {table_name}")

        # 在 sync_database.db 数据库中创建表
        cursor2.execute(create_table_query)

        # 将数据插入到 sync_database.db 数据库中
        for row in data:
            cursor2.execute(f"INSERT INTO {table_name} VALUES (" + ",".join(["?"]*len(row)) + ")", row)

    # 提交更改并关闭连接
    conn2.commit()
    conn1.close()
    conn2.close()




# 同步文件夹——将源文件夹（量化电脑的共享文件夹）中的文件和子文件夹复制到目标文件夹（本机电脑文件夹）中
def sync_folders(source, destination):
    # 确保目标文件夹存在
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source):
        # 计算相对路径
        rel_path = os.path.relpath(root, source)
        dest_path = os.path.join(destination, rel_path)
        
        # 确保目标文件夹中有相应的子文件夹
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        # 复制文件
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            if not os.path.exists(dest_file) or os.path.getmtime(src_file) > os.path.getmtime(dest_file):
                shutil.copy2(src_file, dest_file)


# 上传文件到 Lsky Pro 并返回 Markdown 链接，策略ID为2时，返回的链接为minlo图床的链接
def upload_to_lsky_pro(file_path):
    # 上传信息
    upload_url = "http://192.168.31.143:7791/api/v1/upload"
    token = check_account("password", upload_url)
    policy_id = 2  # minlo图床

    # 读取文件
    files = {'file': open(file_path, 'rb')}
    # 设置请求头
    headers = {
        'Authorization': f'Bearer {token}'
    }

    # 设置请求体
    data = {
        'strategy_id': policy_id
    }

    # 发起请求
    response = requests.post(upload_url, headers=headers, files=files, data=data)

    # 处理响应
    if response.status_code == 200:
        response_data = response.json()
        if response_data['status']:
            return response_data['data']['links']['markdown']
        else:
            raise Exception(f"上传失败: {response_data['message']}")
    else:
        raise Exception(f"上传失败: {response.status_code} {response.text}")

# 获取小数点后的位数
def get_decimal_places(number):
    num_str = f"{number}".rstrip('0')  # 移除末尾的0
    parts = num_str.split('.')
    if len(parts) > 1:
        return len(parts[1])
    else:
        return 0
    
# 保存到数据库
def save_to_database(data, db_path, table_name):
    try:
        # 使用with语句管理数据库连接和光标
        with sqlite3.connect(db_path) as conn:
            # 将数据保存到指定的表名
            data.to_sql(table_name, conn, if_exists='replace', index=False)
    except sqlite3.DatabaseError as e:
        print(f"保存到数据库时出错: {e}")


# 读取数据库数据
def get_existing_data(db_path, table_name):
    try:
        # 使用with语句管理数据库连接
        with sqlite3.connect(db_path) as conn:
            query = f"SELECT * FROM {table_name}"
            existing_data = pd.read_sql_query(query, conn)
            return existing_data
    except sqlite3.DatabaseError as e:
        print(f"读取数据库数据时出错: {e}")
        return pd.DataFrame()


# 对可转债数据进行加权
def process_data(data, db_path, data_table, top_3_table, price_w, prem_w, size_w, conv_w, opt_w):
    try:
        # 对于转股溢价率、最新变动后余额、最新价越高排名越低，设置 ascending=False
        data['最新价_rank'] = data['最新价'].rank(ascending=False)
        data['转股溢价率_rank'] = data['转股溢价率'].rank(ascending=False)
        data['最新变动后余额_rank'] = data['最新变动后余额'].rank(ascending=False)
        data['转股价值_rank'] = data['转股价值'].rank(ascending=False)
        
        # 对于期权价值越高排名越高，设置 ascending=True
        data['期权价值_rank'] = data['期权价值'].rank(ascending=True) 

        # 计算总得分
        data['总得分'] = (
            data['最新价_rank'] * price_w +
            data['转股溢价率_rank'] * prem_w +
            data['最新变动后余额_rank'] * size_w +
            data['转股价值_rank'] * conv_w +
            data['期权价值_rank'] * opt_w
        )

        # 获取总得分最高的前3名
        top_3 = data.nlargest(3, '总得分')

        # 将处理好的数据保存到数据库
        save_to_database(data, db_path, data_table)

        # 将总得分最高的3只保存到数据库
        save_to_database(top_3, db_path, top_3_table)

        return top_3
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return pd.DataFrame()




if __name__ == "__main__":
    db_path = r'D:\wenjian\python\smart\data\mt5.db'
