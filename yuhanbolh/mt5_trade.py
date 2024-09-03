import pandas as pd
import numpy as np
import math
import time
import MetaTrader5 as mt5
from . import send_message as sm
import schedule
import sqlite3
from datetime import datetime, timezone, timedelta, date
# mt5交易委托文件


# 将持仓数据保存到数据库
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

# 将委托后数据插入到成交历史数据到数据库
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


# 将未成交的委托保存到数据库中
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





# 读取数据库forex_order的委托订单，进行批量委托，参数有两个，一个是数据库路径，一个是表名
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
        print("No orders found in the database.")
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
                print(f"Failed to get symbol info for {symbol}")
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
                action = 5  # 限价委托

            # 检查是否已有相同的委托单
            existing_orders = mt5.orders_get(symbol=symbol, magic=int(order_info['EA_id']), type=order_type)
            if existing_orders:
                for order in existing_orders:
                    # 撤销已有的委托单
                    request_cancel = {
                        "action": mt5.TRADE_ACTION_REMOVE,
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

# 插入市价委托
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





# 插入限价委托
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



# 插入平仓委托
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


# 插入撤单委托
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



# 将查询结果转换为字典格式，以便可以通过列名访问数据。
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# 获得美股网格交易委托数据，有4个参数，分别是数据库路径、总评价表名、EA策略代码、交易金额
# 例如get_stock_grid_orders(db_path, "纳指100总评价", 8, 500)
def get_stock_grid_orders(db_path, table_name, magic_number, volume_base):
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory

    try:
        cursor = conn.cursor()

        # 清空forex_order表
        cursor.execute("DELETE FROM forex_order")
        conn.commit()

        # 查询“纳指100总评价”表中的所有记录
        cursor.execute(f"SELECT * FROM `{table_name}`")
        nasdaq_evaluations = cursor.fetchall()

        for eval in nasdaq_evaluations:
            symbol = eval['mt5代码']
            close = eval['close']
            std_residuals_lyear = eval['std_residuals_1year']
            grid = eval['网格']
            comment = f'{table_name}网格'

            if grid is None:
                continue

            # 查找“position”表中指定magic且对应的symbol的记录
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ?", (symbol, magic_number))
            positions = cursor.fetchall()

            # 查询策略内多单的数据，用于计算持仓个数
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ? AND type = ?", (symbol, magic_number, 0))
            positions_type_0 = cursor.fetchall()

            # 查询策略内空单的数据，用于计算持仓个数
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ? AND type = ?", (symbol, magic_number, 1))
            positions_type_1 = cursor.fetchall()

            # 通过magic和symbol查询position表中的记录数量
            position_count = len(positions)

            # 如果position记录数超过10，跳过插入委托数据
            if position_count > 10:
                continue

            # 查询“汇率换算”表中“USDCNH”的汇率
            cursor.execute("SELECT 汇率 FROM `汇率换算` WHERE 货币对 = 'USDCNH'")
            exchange_rate_result = cursor.fetchone()
            if exchange_rate_result is None:  # 确保汇率信息存在
                continue
            exchange_rate = exchange_rate_result['汇率']

            # 计算volume
            volume = volume_base / exchange_rate / close
            # 向上取至最接近的0.1
            volume = math.ceil(volume * 10) / 10

            magic = magic_number
            # 获取symbol的价格偏差
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:  # 确保symbol信息存在
                continue
            deviation = symbol_info.point * 10

            if grid >= 3:
                # 处理多头订单逻辑
                tp = close + std_residuals_lyear
                type = 2  # 多头订单类型
                if not positions:
                    # 插入新的多头订单
                    limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)
                else:
                    # 检查现有的多头订单是否需要操作
                    latest_position = max(positions, key=lambda x: x['time'])
                    price_open = latest_position['price_open']
                    price_current = latest_position['price_current']
                    if price_open - std_residuals_lyear > price_current:
                        # 插入新的多头订单
                        limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)


            # 清仓所有订单
            elif grid >= 2 and grid <= 3 and len(positions_type_1) > 0:
                # 平所有空头订单
                type = 0  # 平空头订单
                for position in positions:
                    ticket = position['ticket']
                    volume = position['volume']
                    close_position_fn(conn, magic, symbol, volume, deviation, type, comment, ticket)

            elif grid <= -2 and grid >= -3 and len(positions_type_0) > 0:
                # 平所有多头订单
                type = 1  # 平多头订单
                for position in positions:
                    ticket = position['ticket']
                    volume = position['volume']
                    close_position_fn(conn, magic, symbol, volume, deviation, type, comment, ticket)

            


            elif grid <= -4:
                # 处理空头订单逻辑
                tp = close - std_residuals_lyear
                type = 3  # 空头订单类型
                if not positions:
                    # 插入新的空头订单
                    limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)
                # 如果有其他处理逻辑，例如检查现有的空头订单是否需要操作，可以在这里添加
                else:
                    # 检查现有的空头订单是否需要操作
                    latest_position = max(positions, key=lambda x: x['time'])
                    price_open = latest_position['price_open']
                    price_current = latest_position['price_current']
                    if price_open + std_residuals_lyear < price_current:
                        # 插入新的空头订单
                        limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)

    except Exception as e:
        print("发生错误:", e)
    finally:
        # 确保在函数结束前关闭数据库连接
        conn.close()

# 获得美股基本技术交易委托数据，有4个参数，分别是数据库路径、总评价表名、EA策略代码、交易金额
# 例如get_stock_basic_tech_orders(db_path, "纳指100总评价", 9, 1000)
def get_stock_basic_tech_orders(db_path, table_name, magic_number, volume_base):
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory

    try:
        cursor = conn.cursor()

        # 清空forex_order表
        cursor.execute("DELETE FROM forex_order")
        conn.commit()

        # 查询“纳指100总评价”表中的所有记录
        cursor.execute(f"SELECT * FROM `{table_name}`")
        nasdaq_evaluations = cursor.fetchall()

        for eval in nasdaq_evaluations:
            symbol = eval['mt5代码']
            close = eval['close']
            std_residuals_lyear = eval['std_residuals_1year']
            grid = eval['基本技术']
            comment = f'{table_name}基本技术'
            
            if grid is None:
                continue

            # 查找“position”表中指定magic且对应的symbol的记录
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ? ", (symbol, magic_number))
            positions = cursor.fetchall()

            # 通过magic和symbol查询position表中的记录数量
            position_count = len(positions)

            # 查询策略内多单的数据，用于计算持仓个数
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ? AND type = ?", (symbol, magic_number, 0))
            positions_type_0 = cursor.fetchall()

            # 查询策略内空单的数据，用于计算持仓个数
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ? AND type = ?", (symbol, magic_number, 1))
            positions_type_1 = cursor.fetchall()

            # 如果position记录数超过10，跳过插入委托数据
            if position_count > 10:
                continue

            # 查询“汇率换算”表中“USDCNH”的汇率
            cursor.execute("SELECT 汇率 FROM `汇率换算` WHERE 货币对 = 'USDCNH'")
            exchange_rate_result = cursor.fetchone()
            if exchange_rate_result is None:  # 确保汇率信息存在
                continue
            exchange_rate = exchange_rate_result['汇率']

            # 计算volume
            volume = volume_base / exchange_rate / close
            # 向上取至最接近的0.1
            volume = math.ceil(volume * 10) / 10

            magic = magic_number
            # 获取symbol的价格偏差
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:  # 确保symbol信息存在
                continue
            deviation = symbol_info.point * 10

            if grid >= 4 and grid <= 6:
                # 处理多头订单逻辑
                tp = close * 1.15
                type = 2  # 多头订单类型
                if (position_count == 0 and grid >= 4) or \
                   (position_count == 1 and grid >= 5) or \
                   (position_count == 2 and grid == 6):
                    # 插入新的多头订单
                    limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)

            elif grid >= 1 and grid <= 3 and len(positions_type_1) > 0:
                # 平所有空头订单
                type = 0  # 平空头订单
                for position in positions:
                    ticket = position['ticket']
                    volume = position['volume']
                    close_position_fn(conn, magic, symbol, volume, deviation, type, comment, ticket)

            elif grid <= -1 and grid >= -3 and len(positions_type_0) > 0:
                # 平所有多头订单
                type = 1  # 平多头订单
                for position in positions:
                    ticket = position['ticket']
                    volume = position['volume']
                    close_position_fn(conn, magic, symbol, volume, deviation, type, comment, ticket)

            elif grid <= -4 and grid >= -6:
                # 处理空头订单逻辑
                tp = close * 0.85
                type = 3  # 空头订单类型
                if (position_count == 0 and grid <= -5) or \
                   (position_count == 1 and grid <= -6):
                    # 插入新的空头订单
                    limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)
    except Exception as e:
        print("发生错误:", e)
    finally:
        # 确保在函数结束前关闭数据库连接
        conn.close()



# 获得指数外汇网格交易委托数据，有3个参数，分别是数据库路径、总评价表名、EA策略代码
# 例如get_index_forex_grid_orders(db_path, "指数外汇总评价", 1)
def get_index_forex_grid_orders(db_path, table_name, magic_number):
    conn = sqlite3.connect(db_path)
    conn.row_factory = dict_factory

    try:
        cursor = conn.cursor()

        # 清空forex_order表
        cursor.execute("DELETE FROM forex_order")
        conn.commit()

        # 查询“纳指100总评价”表中的所有记录
        cursor.execute(f"SELECT * FROM `{table_name}`")
        nasdaq_evaluations = cursor.fetchall()

        for eval in nasdaq_evaluations:
            symbol = eval['mt5代码']
            close = eval['close']
            std_residuals_lyear = eval['std_residuals_1year']
            grid = eval['网格']
            volume = eval['手数']
            comment = f'{table_name}网格'

            # 查找“position”表中指定magic且对应的symbol的记录
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ?", (symbol, magic_number))
            positions = cursor.fetchall()



            # 查询策略内多单的数据，用于计算持仓个数
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ? AND type = ?", (symbol, magic_number, 0))
            positions_type_0 = cursor.fetchall()

            # 查询策略内空单的数据，用于计算持仓个数
            cursor.execute("SELECT * FROM position WHERE symbol = ? AND magic = ? AND type = ?", (symbol, magic_number, 1))
            positions_type_1 = cursor.fetchall()




            # 通过magic和symbol查询position表中的记录数量
            position_count = len(positions)

            # 如果position记录数超过10，跳过插入委托数据
            if position_count > 10:
                continue

            magic = magic_number
            # 获取symbol的价格偏差
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:  # 确保symbol信息存在
                continue
            deviation = symbol_info.point * 10

            if grid >= 3:
                # 处理多头订单逻辑
                tp = close + std_residuals_lyear
                type = 2  # 多头订单类型
                if not positions:
                    # 插入新的多头订单
                    limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)
                else:
                    # 检查现有的多头订单是否需要操作
                    latest_position = max(positions, key=lambda x: x['time'])
                    price_open = latest_position['price_open']
                    price_current = latest_position['price_current']
                    if price_open - std_residuals_lyear > price_current:
                        # 插入新的多头订单
                        limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)
            

            
            # 清仓所有订单
            elif grid >= 2 and grid <= 3 and len(positions_type_1) > 0:
                # 平所有空头订单
                type = 0  # 平空头订单
                for position in positions:
                    ticket = position['ticket']
                    volume = position['volume']
                    close_position_fn(conn, magic, symbol, volume, deviation, type, comment, ticket)

            elif grid <= -2 and grid >= -3 and len(positions_type_0) > 0:
                # 平所有多头订单
                type = 1  # 平多头订单
                for position in positions:
                    ticket = position['ticket']
                    volume = position['volume']
                    close_position_fn(conn, magic, symbol, volume, deviation, type, comment, ticket)




            elif grid <= -3:
                # 处理空头订单逻辑
                tp = close - std_residuals_lyear
                type = 3  # 空头订单类型
                if not positions:
                    # 插入新的空头订单
                    limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)
                # 如果有其他处理逻辑，例如检查现有的空头订单是否需要操作，可以在这里添加
                else:
                    # 检查现有的空头订单是否需要操作
                    latest_position = max(positions, key=lambda x: x['time'])
                    price_open = latest_position['price_open']
                    price_current = latest_position['price_current']
                    if price_open + std_residuals_lyear < price_current:
                        # 插入新的空头订单
                        limit_order_fn(conn, magic, symbol, volume, close, 0, tp, deviation, type, comment)

    except Exception as e:
        print("发生错误:", e)
    finally:
        # 确保在函数结束前关闭数据库连接
        conn.close()

# 处理并保存“非策略持仓”，筛选出不是所有策略的持仓，即不在策略的成分股中，已经被剔除，每月运行一次。参数：数据库路径，策略表名，策略magic值（即策略值）
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

# 将“非策略持仓”表中的数据插入到forex_order表中，进行平仓持仓，每月运行一次。参数：数据库路径
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


# 定义装饰器函数，用于处理任务函数中的异常
# 需修改的内容：sender_email、sender_password、receiver_email
def se_send_email_on_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"在运行 {func.__name__} 函数时发生错误：{str(e)}"
            sm.send_email('sender_email', 'sender_password', 'receiver_email', error_message)
    return wrapper



# 主程序，将要执行的任务函数放到下面
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


    

    