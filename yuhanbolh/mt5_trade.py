import pandas as pd
import time
import MetaTrader5 as mt5
import schedule
import sqlite3
from datetime import datetime, timezone, timedelta


# mt5交易委托文件


# 将持仓数据保存到数据库，参数分别是：数据库路径、表名
def export_positions_to_db(db_path, table_name):
    try:
        # 尝试获取所有持仓数据
        positions = mt5.positions_get()
        if positions is None:
            print("没有持仓, error code =", mt5.last_error())
        else:
            # 创建持仓数据的DataFrame
            df_positions = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            
            # 创建时区信息，北京时间为 UTC+8
            tz_beijing = timezone(timedelta(hours=8))

            # 转换时间戳为北京时间并更新DataFrame
            df_positions['time'] = pd.to_datetime(df_positions['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(tz_beijing)

            # 连接到SQLite数据库
            conn = sqlite3.connect(db_path)
            
            # 导出数据到数据库
            df_positions.to_sql(table_name, conn, if_exists='replace', index=False)
            
            print(f"导出数据到数据表{table_name} =", len(positions))

            # 提交事务并关闭连接
            conn.commit()

    except Exception as e:
        print("发生错误:", e)

    finally:
        # 确保在任何情况下都关闭数据库连接
        conn.close()

# 将委托后数据插入到成交历史数据到数据库，参数分别是：数据库路径、结果
def insert_into_db(db_path, result):
    try:
        # 检查是否成功执行订单
        if result.retcode != 10009:
            print("Order not executed successfully, retcode:", result.retcode)
            return
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 提取交易请求的信息，并使用返回的订单号更新
        request = result._asdict()['request']._asdict()
        request['order'] = result.order  # 使用返回的订单号更新请求字典

        # 将信息插入到forex_trades表中
        cursor.execute("INSERT INTO forex_trades (交易类型, EA_id, 订单号, 品种名称, 交易量, 价格, Limit挂单, 止损, 止盈, 价格偏差, 订单类型, 成交类型, 订单有效期, 订单到期, 订单注释, 持仓单号, 反向持仓单号) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (request['action'], request['magic'], request['order'], request['symbol'], request['volume'], request['price'], request['stoplimit'], request['sl'], request['tp'], request['deviation'], request['type'], request['type_filling'], request['type_time'], request['expiration'], request['comment'], request['position'], request['position_by']))

        # 提交更改
        conn.commit()

    except Exception as e:
        print("发生错误:", e)

    finally:
        # 关闭数据库连接
        if conn:
            conn.close()


# 将未成交的委托保存到数据库中，参数分别是：数据库路径、表名
def save_unsettled_orders_to_db(db_path, table_name):
    try:
        # 获取所有未成交的订单
        orders = mt5.orders_get()

        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
        
        if orders is None or len(orders) == 0:
            print("No unsettled orders")
            # 使用 SQL 删除命令来清空表中的数据，但保留列结构
            conn.execute(f"DELETE FROM {table_name}")
            print(f"表 '{table_name}' 的数据已被清除。")
        else:
            # 创建 DataFrame
            df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())

            # 将数据存储到 SQL 数据库
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"数据成功保存到表'{table_name}' 在数据库 '{db_path}'.")

        # 提交事务并关闭连接
        conn.commit()

    except Exception as e:
        print("发生错误:", e)

    finally:
        # 确保在任何情况下都关闭数据库连接
        conn.close()





# 读取数据库forex_order的委托订单，进行批量委托，参数有两个，一个是数据库路径，一个是表名。例如execute_order_from_db(db_path, "forex_order")
def execute_order_from_db(db_path, table_name):
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
    
        # 读取数据库中的订单数据
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        print(f"数据库读取错误: {e}")
        return
    finally:
        # 确保在任何情况下都关闭数据库连接
        conn.close()

    # 检查数据框是否为空
    if df.empty:
        print("委托数据表为空")
        return

    # 遍历DataFrame中的每一行
    for index, order_info in df.iterrows():
        try:
            symbol = order_info['品种名称']
            order_type = int(order_info['订单类型'])
            requested_price = float(order_info['价格'])

            # 获取当前市场价格
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"无法获取品种信息：{symbol}")
                continue

            current_price = symbol_info.bid if order_type in [0, 2] else symbol_info.ask

            # 调整订单类型和行动
            if order_type == 2 and requested_price > current_price:
                action = 1  # 市价委托
                order_type = 0  # 市价买单
            elif order_type == 3 and requested_price < current_price:
                action = 1  # 市价委托
                order_type = 1  # 市价卖单
            else:
                # action = 5  # 限价委托
                action = order_info['交易类型']  # 假设你已经在订单信息中存储了原始的动作
                order_type = order_info['订单类型']  # 保持原始的订单类型

            # 检查是否已有相同的委托单
            existing_orders = mt5.orders_get(symbol=symbol, magic=int(order_info['EA_id']), type=order_type)
            if existing_orders:
                for order in existing_orders:
                    # 撤销已有的委托单
                    request_cancel = {
                        "action": mt5.TRADE_ACTION_REMOVE,   # 相当于8
                        "order": order.ticket
                    }
                    mt5.order_send(request_cancel)

            # 设置交易请求字典
            request = {
                "action": action,
                "magic": int(order_info['EA_id']),
                "order": int(order_info['订单号']),
                "symbol": symbol,
                "volume": float(order_info['交易量']),
                "price": requested_price if action == 5 else current_price,
                "stoplimit": float(order_info['Limit挂单']),

                "sl": float(order_info['止损']),
                "tp": float(order_info['止盈']),

                "deviation": int(order_info['价格偏差']),
                "type": order_type,
                "type_filling": int(order_info['成交类型']),
                "type_time": int(order_info['订单有效期']),  # 设置订单的有效期
                "expiration": int(order_info['订单到期']),    # 订单到期 
                "comment": order_info['订单注释'],
                "position": int(order_info['持仓单号']),  # 持仓单号
                "position_by": int(order_info['反向持仓单号'])    # 反向持仓单号
            }

            print("交易请求：", request)
            # 发送交易请求
            result = mt5.order_send(request)


            # 检查执行结果，并输出相关信息
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print("Order executed successfully:", result)
                insert_into_db(db_path, result)
            else:
                print("Order execution failed:", result)
        except Exception as e:
            print(f"处理订单时出错: {e}")
    # 断开MetaTrader 5连接
    # mt5.shutdown()

# 插入市价委托，参数：conn为数据库连接对象，magic为EA的magic number，symbol为交易品种，volume为交易量，sl为止损价，tp为止盈价，deviation为价格偏差，type为订单类型，comment为订单注释。例如market_order_fn(conn, magic, symbol, volume, sl, tp, deviation, type, comment)
def market_order_fn(conn, magic, symbol, volume, sl, tp, deviation, type, comment):
    try:
        insert_query = """
        INSERT INTO forex_order (交易类型, EA_id, 订单号, 品种名称, 交易量, 价格, Limit挂单, 止损, 止盈, 价格偏差, 订单类型, 成交类型, 订单有效期, 订单到期, 订单注释, 持仓单号, 反向持仓单号) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # 插入的值
        values = (1, magic, 0, symbol, volume, 0, 0, sl, tp, deviation, type, 1, 0, 0, comment, 0, 0)

        # 执行插入操作
        conn.execute(insert_query, values)
        conn.commit()

    except Exception as e:
        print("An error occurred:", e)



# 插入限价委托，参数：conn为数据库连接对象，magic为EA的magic number，symbol为交易品种，volume为交易量，price为价格，sl为止损价，tp为止盈价，deviation为价格偏差，type为订单类型，comment为订单注释。例如limit_order_fn(conn, magic, symbol, volume, price, sl, tp, deviation, type, comment)
def limit_order_fn(conn, magic, symbol, volume, price, sl, tp, deviation, type, comment):
    try:
        insert_query = """
        INSERT INTO forex_order (交易类型, EA_id, 订单号, 品种名称, 交易量, 价格, Limit挂单, 止损, 止盈, 价格偏差, 订单类型, 成交类型, 订单有效期, 订单到期, 订单注释, 持仓单号, 反向持仓单号)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # 插入的值
        values = (5, magic, 0, symbol, volume, price, 0, sl, tp, deviation, type, 1, 0, 0, comment, 0, 0)

        # 执行插入操作
        conn.execute(insert_query, values)
        conn.commit()

    except Exception as e:
        print("An error occurred:", e)




# 插入平仓委托，参数：conn为数据库连接对象，magic为EA的magic number，symbol为交易品种，volume为交易量，deviation为价格偏差，type为订单类型，comment为订单注释，position为持仓单号。例如close_position_fn(conn, magic, symbol, volume, deviation, type, comment, position)
def close_position_fn(conn, magic, symbol, volume, deviation, type, comment, position):
    try:
        insert_query = """
        INSERT INTO forex_order (交易类型, EA_id, 订单号, 品种名称, 交易量, 价格, Limit挂单, 止损, 止盈, 价格偏差, 订单类型, 成交类型, 订单有效期, 订单到期, 订单注释, 持仓单号, 反向持仓单号)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # 插入的值
        values = (1, magic, 0, symbol, volume, 0, 0, 0, 0, deviation, type, 1, 0, 0, comment, position, 0)

        # 执行插入操作
        conn.execute(insert_query, values)
        conn.commit()

    except Exception as e:
        print("An error occurred:", e)



# 插入撤单委托，参数：conn为数据库连接对象，magic为EA的magic number，order为订单号。例如cancel_order_fn(conn, magic, order)
def cancel_order_fn(conn, magic, order):
    try:
        insert_query = """
        INSERT INTO forex_order (交易类型, EA_id, 订单号, 品种名称, 交易量, 价格, Limit挂单, 止损, 止盈, 价格偏差, 订单类型, 成交类型, 订单有效期, 订单到期, 订单注释, 持仓单号, 反向持仓单号)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # 插入的值
        values = (8, magic, order, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # 执行插入操作
        conn.execute(insert_query, values)
        conn.commit()

    except Exception as e:
        print("An error occurred:", e)


# 插入撤销未成交的订单，参数：数据库路径、magic。例如cancel_pending_order(db_path, magic)
def cancel_pending_order(db_path, magic):
    """
    从 unsettled_orders 表中读取特定 magic 的数据，并将其插入到 forex_order 表中
    :param db_path: 数据库文件路径
    :param magic: 用于筛选的 magic 值
    """
    # 连接到数据库
    conn = sqlite3.connect(db_path)

    try:
        # 查询指定 magic 值的 unsettled_orders 表中的数据
        select_query = "SELECT ticket FROM unsettled_orders WHERE magic = ?"
        cursor = conn.execute(select_query, (magic,))
        rows = cursor.fetchall()

        # 遍历查询结果并插入到 forex_order 表中
        for row in rows:
            ticket = row[0]
            cancel_order_fn(conn, magic, ticket)

    except sqlite3.OperationalError as e:
        print(f"查询或插入时出错: {e}")

    finally:
        # 关闭数据库连接
        conn.close()



# 从成分股中清除不能在mt5交易的产品，参数分别是需要清除的数据库路径和表名。例如remove_unavailable_products_mt5(db_path, "成分股")
def remove_unavailable_products_mt5(db_path, table_name):
    try:
        # 获取所有交易品种的名称
        symbols = mt5.symbols_get()
        if symbols is None:
            raise Exception("无法获取交易品种，可能是MetaTrader 5未初始化")

        symbol_names = {symbol.name for symbol in symbols}

        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 读取指定表中的所有列数据
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # 获取表的列名
        column_names = [description[0] for description in cursor.description]

        # 遍历数据库中的代码，与交易品种对比
        deleted_rows = []
        for row in rows:
            mt5_code = row[column_names.index('mt5代码')]
            if mt5_code not in symbol_names:
                # 如果代码不在交易品种中，删除该行数据
                cursor.execute(f"DELETE FROM {table_name} WHERE mt5代码 = ?", (mt5_code,))
                deleted_rows.append(row)

        # 提交更改
        conn.commit()

        # 打印被删除的行
        print("以下行已被删除：")
        for deleted_row in deleted_rows:
            print(dict(zip(column_names, deleted_row)))

    except sqlite3.Error as e:
        print(f"数据库操作出错: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保关闭数据库连接
        if conn:
            conn.close()



# 处理并保存“非策略持仓”，筛选出不是所有策略的持仓，即不在策略的成分股中，已经被剔除，每月运行一次。参数：数据库路径，策略表名，策略magic值（即策略值）。例如export_non_strategy_positions(db_path, tables, magic_values)
def export_non_strategy_positions(db_path, tables, magic_values):
    # 连接到数据库
    with sqlite3.connect(db_path) as conn:
        # 读取`position`表中符合`magic`值的所有行
        query = f"SELECT * FROM position WHERE magic IN ({','.join('?'*len(magic_values))})"
        df_position = pd.read_sql_query(query, conn, params=magic_values)
        symbols_from_position = set(df_position['symbol'])

        # 读取多个表的`mt5代码`列数据并合并
        symbols_from_other_tables = set()
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT `mt5代码` FROM {table}", conn)
                symbols_from_other_tables.update(df['mt5代码'])
            except sqlite3.Error as e:
                print(f"Error reading from {table}: {e}")

        # 找出不在指定表`mt5代码`列中的`symbol`数据
        symbols_not_in_others = symbols_from_position - symbols_from_other_tables

        # 首先清空 '非策略持仓' 表中的内容，但保留表结构
        conn.execute('DELETE FROM 非策略持仓')

        # 如果有不在其他表中的symbols，则查询对应的完整行数据
        if symbols_not_in_others:
            params = tuple(symbols_not_in_others)
            query = f"SELECT * FROM position WHERE symbol IN ({','.join('?'*len(params))})"
            df_not_in_others = pd.read_sql_query(query, conn, params=params)

            # 将结果导出到新表“非策略持仓”中
            df_not_in_others.to_sql('非策略持仓', conn, if_exists='replace', index=False)

            # 使用Pandas打印表格形式的结果
            # print(df_not_in_others)

# 将“非策略持仓”表中的数据插入到forex_order表中，进行平仓持仓，每月运行一次。参数：数据库路径。例如process_non_strategy_positions(db_path)  
def process_non_strategy_positions(db_path):
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    
    # 清空forex_order表
    cursor = conn.cursor()
    cursor.execute("DELETE FROM forex_order")
    conn.commit()
    
    # 读取“非策略持仓”表中的数据
    df_non_strategy_positions = pd.read_sql_query("SELECT * FROM `非策略持仓`", conn)
    
    # 遍历“非策略持仓”表中的每一行数据，并插入到forex_order表中
    for index, position in df_non_strategy_positions.iterrows():
        # 从数据表中提取相应的值
        magic = position['magic']
        symbol = position['symbol']
        volume = position['volume']
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:  # 确保symbol信息存在
            continue
        deviation = symbol_info.point * 10

        type = 0 if position['type'] == 1 else 1
        comment = position['comment']
        ticket = position['ticket']
        
        close_position_fn(conn, magic, symbol, volume, deviation, type, comment, ticket)
    
    # 关闭数据库连接
    conn.close()




# 主程序，将要执行的任务函数放到下面。例如schedule_jobs()
def schedule_jobs():

    # 计算从周一早上7点到周六凌晨4点，每50分钟的时间点
    hours = list(range(7, 24)) + list(range(0, 4))  # 周一到周五全天，周六到凌晨4点
    minutes = ["05", "35"]
    times = [f"{hour:02d}:{minute}" for hour in hours for minute in minutes]

    while True:
        schedule.run_pending()
        current_time = time.strftime("%H:%M", time.localtime())
        current_weekday = datetime.now().weekday()

        # 检查当前是否为周六且时间已经达到或超过上午7点
        if current_weekday == 5 and current_time >= "07:00":
            print("已经是周六上午07:00，委托程序退出。")
            break

        time.sleep(30)  # 每30秒检查一次



if __name__ == '__main__':
    db_path = r'D:\wenjian\python\smart\data\mt5_demo.db'
    # MT5连接，需要修改的内容主要包括：MT5安装路径、MT5账号、MT5密码、MT5服务器
    while True:
        # 获取当前星期几
        current_time = datetime.now()

        if 0 <= current_time.weekday() < 6 and (7 <= current_time.hour < 24 or current_time.weekday() == 5 and current_time.hour < 4):
            # 初始化MetaTrader 5连接
            if not mt5.initialize(path=r"D:\jiaoyi\IC-MT5-Demo\terminal64.exe", login=515822, password="5228854", server="ICMarketsSC-Demo"):
                print("initialize()失败，错误代码=", mt5.last_error())
            else:
                print("MT5 initialized")

                schedule_jobs()

                while True:
                    schedule.run_pending()
                    current_time = time.strftime("%H:%M", time.localtime())
                    current_weekday = datetime.now().weekday()

                    # 检查当前是否为周六且时间已经达到或超过上午7点
                    if current_weekday == 5 and current_time >= "07:00":
                        print("已经是周六上午07:00，委托程序退出。")
                        mt5.shutdown()
                        break

                    time.sleep(30)  # 每30秒检查一次

            time.sleep(3600)  # 每小时检查一次，等待下一个交易时段
        else:
            print("当前不在运行时间内，等待下周一...")
            time.sleep(3600)  # 每小时检查一次


    

    