#coding=utf-8
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from xtquant import xtdata
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from typing import List
import random
import os
import schedule
import time
import threading

# qmt的委托、交易和推送文件


# 时间的转换和格式化函数，参数：时间戳-'%Y-%m-%d %H:%M:%S'
def convert_time(unix_timestamp):
    traded_time_utc = datetime.utcfromtimestamp(unix_timestamp)  # 注意这里改为 datetime.utcfromtimestamp
    traded_time_beijing = traded_time_utc + timedelta(hours=8)  # 注意这里改为 timedelta
    return traded_time_beijing.strftime('%Y-%m-%d %H:%M:%S')




# 证券资产查询并保存到数据库manage_assets，参数：资产对象（固定）、数据库路径
def save_stock_asset(asset, db_path):
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)

        # 创建DataFrame
        data = {
            '账户类型': [asset.account_type],
            '资金账号': [asset.account_id],
            '现金': [asset.cash],
            '冻结现金': [asset.frozen_cash],
            '市值': [asset.market_value],
            '总资产': [asset.total_asset]
        }
        df = pd.DataFrame(data)

        # 将DataFrame写入数据库，替换现有数据
        df.to_sql('manage_assets', conn, if_exists='replace', index=False)
        conn.close()

        # 打印资金变动推送
        # print("资金变动推送")
        # print(asset.account_type, asset.account_id, asset.cash, asset.frozen_cash, asset.market_value, asset.total_asset)
    except Exception as e:
        print("An error occurred:", e)

# 获取持仓数据并保存到数据库account_holdings，参数：持仓对象（固定）、数据库路径
def save_positions(positions, db_path):
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)

        # 创建DataFrame
        data = {
            '账户类型': [position.account_type for position in positions],
            '资金账号': [position.account_id for position in positions],
            '证券代码': [position.stock_code for position in positions],
            '持仓数量': [position.volume for position in positions],
            '可用数量': [position.can_use_volume for position in positions],
            '平均建仓成本': [position.open_price for position in positions],
            '市值': [position.market_value for position in positions]
        }
        df = pd.DataFrame(data)

        # 将DataFrame写入数据库，替换现有数据
        df.to_sql('account_holdings', conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        print("An error occurred:", e)

# 查询当日委托并保存到数据库daily_orders，参数：委托对象（固定）、数据库路径
def save_daily_orders(orders, db_path):
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)

        # 创建DataFrame
        data = {
            '账户类型': [order.account_type for order in orders],
            '资金账号': [order.account_id for order in orders],
            '证券代码': [order.stock_code for order in orders],
            '委托类型': [order.order_type for order in orders],
            '订单编号': [order.order_id for order in orders],
            '报单时间': [convert_time(order.order_time) for order in orders],
            '委托价格': [order.price for order in orders],
            '委托数量': [order.order_volume for order in orders],
            '报价类型': [order.price_type for order in orders],
            '委托状态': [order.order_status for order in orders],
            '柜台合同编号': [order.order_sysid for order in orders],
            '策略名称': [order.strategy_name for order in orders],
            '委托备注': [order.order_remark for order in orders]
        }
        
        df = pd.DataFrame(data)
        
        # 将DataFrame写入数据库，替换现有数据
        df.to_sql('daily_orders', conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        print("An error occurred:", e)

    # 假设 orders 是一个包含所有订单信息的列表
    # save_daily_orders(orders, db_path)

# 查询当日成交并保存到数据库daily_trades，参数：成交对象（固定）、数据库路径
def save_daily_trades(trades, db_path):
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
        

        # 创建DataFrame
        data = {
            '账户类型': [trade.account_type for trade in trades],
            '资金账号': [trade.account_id for trade in trades],
            '证券代码': [trade.stock_code for trade in trades],
            '委托类型': [trade.order_type for trade in trades],
            '成交编号': [trade.traded_id for trade in trades],
            '成交时间': [convert_time(trade.traded_time) for trade in trades],
            '成交均价': [trade.traded_price for trade in trades],
            '成交数量': [trade.traded_volume for trade in trades],
            '成交金额': [trade.traded_amount for trade in trades],
            '订单编号': [trade.order_id for trade in trades],
            '柜台合同编号': [trade.order_sysid for trade in trades],
            '策略名称': [trade.strategy_name for trade in trades],
            '委托备注': [trade.order_remark for trade in trades]
        }
        
        df = pd.DataFrame(data)
        
        # 将DataFrame写入数据库，替换现有数据
        df.to_sql('daily_trades', conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        print("An error occurred:", e)

    # 假设 trades 是一个包含所有交易信息的列表
    # save_daily_trades(trades, db_path)

# 查询除策略外的其他持仓，并将它保存到数据表other_positions，参数：数据库路径、策略表名称列表
def calculate_remaining_holdings(db_path, strategy_tables):
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)

        # 从account_holdings获取所有的持仓，并确保数据类型正确
        account_holdings = pd.read_sql('SELECT * FROM account_holdings', conn).set_index('证券代码')
        account_holdings['持仓数量'] = account_holdings['持仓数量'].astype(int)

        # 对于每个策略的成交数量表，计算每种证券的总成交数量（考虑买卖方向）并从account_holdings中减去
        for table in strategy_tables:
            strategy_data = pd.read_sql(f'SELECT 证券代码, SUM(成交数量 * 买卖) as total FROM {table} GROUP BY 证券代码', conn).set_index('证券代码')
            strategy_data['total'] = strategy_data['total'].astype(int)
            account_holdings['持仓数量'] -= account_holdings.join(strategy_data, how='left')['total'].fillna(0).astype(int)

        # 只保留持仓数量不等于0的记录
        account_holdings = account_holdings[account_holdings['持仓数量'] != 0].reset_index()

        # 将结果存储到新的数据表other_positions
        account_holdings.to_sql('other_positions', conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"出现错误: {e}")
    finally:
        # 无论是否出现异常，都关闭数据库连接
        conn.close()


# 当为卖出时插入的数据，参数包括数据表名、证券代码、委托价格、委托数量、买卖方向、策略名称、委托备注
def insert_buy_sell_data(place_order_table, security_code, order_price, order_volume, trade_direction, strategy_name, order_remark):
    try:
        # 连接到 SQLite 数据库
        conn = sqlite3.connect(db_path)

        # 获取当前时间，精确到分钟
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

        insert_query = f"""
        INSERT INTO {place_order_table} (证券代码, 委托价格, 委托数量, 买卖, 策略名称, 委托备注, 日期时间)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            security_code,
            order_price,
            order_volume,
            trade_direction,  # 1 表示买入，-1 表示卖出
            strategy_name,
            order_remark,
            current_time,
        )
        conn.execute(insert_query, values)
        conn.commit()
        print(f"数据插入成功: {values}")
    except Exception as e:
        print(f"插入数据时出错: {e}")
    finally:
        # 无论是否出现异常，都关闭数据库连接
        conn.close()



# 证券委托，参数分别是：数据库路径、委托数据表名称、成交数据表名称、判断处函数前缀（不改动）、账号（不改动）
# 注：报价类型主要有xtconstant.FIX_PRICE（限价）、xtconstant.LATEST_PRICE（最新介）、xtconstant.MARKET_PEER_PRICE_FIRST（对手方最优）、xtconstant.MARKET_MINE_PRICE_FIRST（本方最优）
def place_orders(db_path, table_name, trade_table_name, xt_trader, acc):
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)

        # 从数据库读取交易数据
        frame = pd.read_sql(f'SELECT * FROM {table_name}', conn)

        # 如果frame为空，跳过后续所有操作
        if frame.empty:
            print("没有需要处理的交易数据")
            return  # 退出函数


        # 检查连接结果
        connect_result = xt_trader.connect()
        if connect_result == 0:
            print('连接成功')

            # 查询可撤单的订单，连接成功后执行
            orders = xt_trader.query_stock_orders(acc, True)
            
            # 遍历交易数据，执行交易操作
            for i, row in frame.iterrows():
                stock_code = row['证券代码']
                order_type = xtconstant.STOCK_BUY if row['买卖'] == 1 else xtconstant.STOCK_SELL
                order_volume = row['委托数量']
                price = row['委托价格']

                # 根据'委托价格'设置price_type，当委托价格为0时用最新价交易，非0以限价交易
                if price == 0:
                    price_type = xtconstant.LATEST_PRICE
                else:
                    price_type = xtconstant.FIX_PRICE

                strategy_name = row['策略名称']
                order_remark = row['委托备注']



                # 检查是否需要撤单
                for order in orders:
                    if order.stock_code == stock_code:
                        # 撤单
                        cancel_result = xt_trader.cancel_order_stock(acc, order.order_id)
                        if cancel_result == 0:
                            print(f"撤单成功：{order.order_id}")
                        else:
                            print(f"撤单失败：{order.order_id}")


                # 买入操作
                if order_type == xtconstant.STOCK_BUY:
                    asset = xt_trader.query_stock_asset(acc)
                    # 针对特定表名增加额外的现金保留条件
                    if table_name in ['place_fund_grid_order', 'place_fund_basics_technical_order'] and asset.cash <= 3000:
                        print("基金交易，保存现金3000，金额不足，跳过买入")
                        continue
                    
                    # 原有的现金检查条件
                    if asset.cash < order_volume * price:
                        print(f"跳过买入 {stock_code}，现金不足")
                        continue

                # 卖出操作
                if order_type == xtconstant.STOCK_SELL:
                    # 从指定的trade_table_name表中获取该股票的可用数量，并求和
                    query = f"SELECT SUM(成交数量 * 买卖) FROM {trade_table_name} WHERE 证券代码 = '{stock_code}'"
                    available_volume = pd.read_sql_query(query, conn).iloc[0, 0]

                    if available_volume is None or available_volume <= 0:
                        print(f"跳过卖出 {stock_code}，没有持仓或可用数量不足")
                        continue

                    if available_volume < order_volume:
                        print(f"{stock_code} 可用数量不足，将卖出剩余 {available_volume} 股")
                        order_volume = available_volume
                    elif available_volume >= order_volume:
                        print(f"{stock_code} 可用数量充足，将卖出 {order_volume} 股")
                
                # 尝试同步报单，异步报单需要改成xt_trader.order_stock_async
                try:
                    order_id = xt_trader.order_stock(acc, stock_code, order_type, order_volume, price_type, price, strategy_name, order_remark)
                    print(order_id, stock_code, order_type, order_volume, price)
                except Exception as e:
                    error_message = f"报单操作失败，原因：{str(e)}"
                    print(f"报单操作失败，原因：{e}")
                    continue
        else:
            print('连接失败')
    except Exception as e:
        print(f"执行过程中出现错误：{e}")
    finally:
        # 无论是否出现异常，都关闭数据库连接
        conn.close()


# 比较沪深两市的一天期的买一国债逆回购，选择值大的进行卖出，参数分别是：交易对象（固定）、账号（固定）、数据对象（固定）
def place_order_based_on_asset(xt_trader, acc, xtdata):
    # 检查连接结果
    connect_result = xt_trader.connect()
    if connect_result == 0:
        print('连接成功')
        try:
            # 查询证券资产
            asset = xt_trader.query_stock_asset(acc)
            print("证券资产查询保存成功, 可用资金：", asset.cash)

            # 判断可用资金是否足够
            if asset.cash >= 1000:
                # 根据资产计算订单量
                order_volume = int(asset.cash / 1000) * 10

                # 获取市场数据以确定股票代码和价格
                xtdata.subscribe_quote('131810.SZ', period='tick', start_time='', end_time='', count=1, callback=None)
                xtdata.subscribe_quote('204001.SH', period='tick', start_time='', end_time='', count=1, callback=None)
                generate_func = xtdata.get_market_data(field_list=['bidPrice'], stock_list=['131810.SZ','204001.SH'], period='tick', start_time='', end_time='', count=-1, dividend_type='none', fill_data=True)

                # 提取 '131810.SZ' 和 '204001.SH' 的最后一个 bidPrice
                price_131810 = generate_func['131810.SZ']['bidPrice'][-1][0]
                price_204001 = generate_func['204001.SH']['bidPrice'][-1][0]

                # 根据价格选择股票代码和价格
                if price_131810 >= price_204001:
                    stock_code, price = '131810.SZ', price_131810
                else:
                    stock_code, price = '204001.SH', price_204001

                # 下达股票订单
                xt_trader.order_stock(
                    acc, 
                    stock_code, 
                    xtconstant.STOCK_SELL, 
                    order_volume, 
                    xtconstant.FIX_PRICE, 
                    price, 
                    '国债逆回购策略', 
                    ''
                )
                print(f"成功下达订单，股票代码：{stock_code}，价格：{price}，订单量：{order_volume}。")
            else:
                print("可用资金不足，不进行交易")
        except Exception as e:
            print("下达订单时出现错误:", e)
    else:
        print('连接失败')

# 对委托数据表进行排序并更新，先卖后买，先评分（操作）高后评分低。参数是表名和数据库路径
def sort_and_update_table(table_name, db_path=r"D:\wenjian\python\smart\data\guojin_account.db"):
    conn = None
    try:
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 按照买卖和操作列排序查询数据
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY 买卖 ASC, 操作 DESC")
        sorted_rows = cursor.fetchall()

        # 如果查询结果为空，则不进行后续操作
        if not sorted_rows:
            print("没有数据进行排序和更新。")
            return

        # 如果查询结果不为空，则删除所有现有数据
        cursor.execute(f"DELETE FROM {table_name}")

        # 为排序后的数据准备插入语句
        columns_count = len(sorted_rows[0])
        placeholders = ', '.join('?' * columns_count)
        insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"

        # 将排序后的数据重新插入到表中
        cursor.executemany(insert_query, sorted_rows)

        # 提交事务
        conn.commit()

    except sqlite3.Error as error:
        print("SQLite数据库错误:", error)

    finally:
        # 确保关闭数据库连接
        if conn:
            conn.close()


# 定义qmt推送的类
class MyXtQuantTraderCallback(XtQuantTraderCallback):
    
    def __init__(self, db_path=r'D:\wenjian\python\smart\data\guojin_account.db'):
        self.db_path = db_path
    
    # 资金变动推送  注意，该回调函数目前不生效
    def on_stock_asset(self, asset):
        try:
            # 使用上下文管理器连接到SQLite数据库
            with sqlite3.connect(self.db_path) as conn:

                # 创建DataFrame
                data = {
                    '账户类型': [asset.account_type],
                    '资金账号': [asset.account_id],
                    '现金': [asset.cash],
                    '冻结现金': [asset.frozen_cash],
                    '市值': [asset.market_value],
                    '总资产': [asset.total_asset]
                }
                df = pd.DataFrame(data)

                # 将DataFrame写入数据库，替换现有数据
                df.to_sql('manage_assets', conn, if_exists='replace', index=False)

            # 打印资金变动推送
            print("资金变动推送")
            print(asset.account_type, asset.account_id, asset.cash, asset.frozen_cash, asset.market_value, asset.total_asset)
        except Exception as e:
            print("出现错误:", e)

    # 成交变动推送，每增加一个策略都要往里增加保存数据表的名称
    def on_stock_trade(self, trade):
        try:
            # 使用上下文管理器连接到SQLite数据库
            with sqlite3.connect(self.db_path) as conn:
            
                # 成交变动推送
                buy_sell = 1 if trade.order_type == 23 else -1 if trade.order_type == 24 else 0
                values = (
                    trade.account_id,        # 资金账号
                    trade.order_id,          # 订单编号
                    trade.strategy_name,     # 策略名称
                    trade.order_remark,      # 委托备注
                    convert_time(trade.traded_time),       # 成交时间
                    trade.order_type,        # 委托类型
                    trade.stock_code,        # 证券代码
                    trade.traded_price,      # 成交均价
                    trade.traded_volume,     # 成交数量
                    trade.traded_amount,     # 成交金额
                    buy_sell                # 买卖
                )
    
                # 使用策略名称与表名的映射字典简化选择插入的表的代码
                strategy_to_table = {
                    "基金网格": "execute_fund_grid_trade",
                    "基金基本技术": "execute_fund_basics_technical_trade",
                    "索提诺比率策略": "execute_sortino_ratio_trade",
                    "卡玛比率策略": "execute_calmar_ratio_trade"
                }
    
                table_name = strategy_to_table.get(trade.strategy_name, "unallocated_transaction")
                if table_name == "unallocated_transaction":
                    return
    
                insert_query = f"""
                INSERT INTO {table_name} (资金账号, 订单编号, 策略名称, 委托备注, 成交时间, 委托类型, 证券代码, 成交均价, 成交数量, 成交金额, 买卖)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                conn.execute(insert_query, values)
                conn.commit()
    
        except Exception as e:
            print("出现错误:", e)

    # 持仓变动推送  注意，该回调函数目前不生效
    def on_stock_position(self, positions):
        try:
            # 使用上下文管理器连接到SQLite数据库
            with sqlite3.connect(self.db_path) as conn:

                # 准备插入的数据
                values = [(position.account_type,         # 账户类型
                           position.account_id,           # 资金账号
                           position.stock_code,           # 证券代码
                           position.volume,               # 持仓数量
                           position.can_use_volume,       # 可用数量
                           position.open_price,           # 平均建仓成本
                           position.market_value          # 市值
                          ) for position in positions]

                # 插入到account_holdings表
                insert_query = """
                INSERT INTO account_holdings (账户类型, 资金账号, 证券代码, 持仓数量, 可用数量, 平均建仓成本, 市值)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                conn.executemany(insert_query, values)
                conn.commit()

        except Exception as e:
            print("出现错误:", e)


    def on_disconnected(self):
        """
        连接断开
        :return:
        """
        print("connection lost, 交易接口断开，即将重连")

        global schedule_thread
        schedule_thread = None

    def on_stock_order(self, order):
        """
        委托回报推送
        :param order: XtOrder对象
        :return:
        """
        print("委托回报推送")
        print(order.stock_code, order.order_status, order.order_sysid)

    def on_order_error(self, order_error):
        """
        委托失败推送
        :param order_error:XtOrderError 对象
        :return:
        """
        print("委托失败推送")
        print(order_error.order_id, order_error.error_id, order_error.error_msg)

    def on_cancel_error(self, cancel_error):
        """
        撤单失败推送
        :param cancel_error: XtCancelError 对象
        :return:
        """
        print("撤单失败推送")
        print(cancel_error.order_id, cancel_error.error_id, cancel_error.error_msg)

    def on_order_stock_async_response(self, response):
        """
        异步下单回报推送
        :param response: XtOrderResponse 对象
        :return:
        """
        print("异步下单回报推送")
        print(response.account_id, response.order_id, response.seq)

    # 账户状态
    def on_account_status(self, status):
        """
        :param response: XtAccountStatus 对象
        :return:
        """
        print("账户状态，类型2为\"证券账户\"，状态：-1-无效；0-正常；4-初始化")   # http://docs.thinktrader.net/vip/pages/198696/#%E8%B4%A6%E5%8F%B7%E7%8A%B6%E6%80%81-account-status
        print(status.account_id, status.account_type, status.status)


# 保存证券资产、委托、成交和持仓数据到数据库，参数分别是：交易对象（固定）、账号（固定）、数据库路径
def save_daily_data(xt_trader, acc, db_path):
    """
    保存当日的持仓、委托和成交数据到数据库
    :param xt_trader: 交易对象
    :param acc: 账户信息
    :param db_path: 数据库路径
    """
    try:
        # 查询证券资产
        asset = xt_trader.query_stock_asset(acc)
        if asset:
            save_stock_asset(asset, db_path)
            print("证券资产查询保存成功,可用资金：", asset.cash)
    except Exception as e:
        print("保存证券资产时出现错误:", e)

    try:
        # 查询当日所有的持仓
        positions = xt_trader.query_stock_positions(acc)
        if len(positions) != 0:
            save_positions(positions, db_path)
            print(f"持仓保存成功, 持仓数据： {len(positions)}")
    except Exception as e:
        print("保存持仓数据时出现错误:", e)

    try:
        # 查询当日所有的委托
        orders = xt_trader.query_stock_orders(acc)
        if len(orders) != 0:
            save_daily_orders(orders, db_path)
            print(f"当日委托保存成功, 委托数据： {len(orders)}")
    except Exception as e:
        print("保存委托数据时出现错误:", e)

    try:
        # 查询当日所有的成交
        trades = xt_trader.query_stock_trades(acc)
        if len(trades) != 0:
            save_daily_trades(trades, db_path)
            print(f"当日成交保存成功, 成交数据： {len(trades)}")
    except Exception as e:
        print("保存成交数据时出现错误:", e)



# 查询未成交的委托，然后进行逐一撤单，参数分别是：交易对象（固定）、账号（固定）
def cancel_all_orders(xt_trader, acc):
    try:
        orders = xt_trader.query_stock_orders(acc, True)
        
        # 提取订单ID
        order_ids_strings = [order.order_id for order in orders]
        
        # 如果有订单，则尝试撤销
        if len(orders) != 0:
            print(f"查询到的订单ID： {order_ids_strings}")

            # 遍历订单ID并撤销
            for order in order_ids_strings:
                xt_trader.cancel_order_stock(acc, int(order))
                print(f"订单ID {order} 已成功撤销。")
            print("所有订单已成功撤销。")
        else:
            print("没有订单需要撤销。")
            
    except Exception as e:
        print("撤销订单时出现错误:", e)



# 周一到周五运行函数
def run_weekdays_at(time_str, function):
    schedule.every().monday.at(time_str).do(function)
    schedule.every().tuesday.at(time_str).do(function)
    schedule.every().wednesday.at(time_str).do(function)
    schedule.every().thursday.at(time_str).do(function)
    schedule.every().friday.at(time_str).do(function)

# 计划任务
def schedule_jobs():
    # 全局变量，用于判断计划任务是否在运行
    global schedule_thread
    try:
        print("开始执行计划任务")
        '''
        # 在以下时间点执行onshore_fund_grid_strategy任务，运行场内基金网格策略
        for time_str in ["09:35", "09:55", "10:15", "10:35", "10:55", "11:15", "13:05", "13:25", "13:45", "14:05", "14:25", "14:45"]:
            run_weekdays_at(time_str, onshore_fund_grid_strategy)

        # 在收盘的前4分钟对委托进行撤单
        for time_str in ["11:28", "14:56"]:
            run_weekdays_at(time_str, lambda: cancel_all_orders(xt_trader, acc))
        '''
        # 循环，用于持续执行计划任务
        while True:
            schedule.run_pending()

            current_time = time.strftime("%H:%M", time.localtime())

            # 直接在这里检查当前时间是否在指定的时间区间内
            if ("09:30" <= current_time <= "11:30") or ("13:00" <= current_time <= "15:00"):
                print('测试')

            # 检查是否到达或超过15:20，如果是，则退出循环
            if current_time >= "15:10":
                print("已经15:20，退出委托程序。")
                break

            # 每3秒检查一次
            time.sleep(5)
        
    except Exception as e:
        print(f"发生错误: {e}")
        # 可以在这里添加额外的错误处理逻辑
    finally:
        schedule_thread = None
        # 在这里确保在退出函数前将线程运行状态设置为False




# 下面是运行例子
if __name__ == '__main__':
    global schedule_thread
    schedule_thread = None

    while True:
        try:
            # 获取当前时间和日期
            now = datetime.now()
            current_weekday = now.weekday()  # 周一为0，周二为1，以此类推至周日为6
            current_time = time.strftime("%H:%M", time.localtime())
            if "09:00" <= current_time <= "16:10" and 0 <= current_weekday <= 4:
                # 需传入下面四个参数，分别是QMT交易端路径、账户ID、数据库路径、计划任务线程
                path = r'D:\国金证券QMT交易端\userdata_mini'
                acc = StockAccount('8484555568')
                db_path = r'D:\wenjian\python\smart\data\guojin_account.db'


                session_id = int(random.randint(100000, 999999))
                xt_trader = XtQuantTrader(path, session_id)
                callback = MyXtQuantTraderCallback(db_path)
                xt_trader.register_callback(callback)
                xt_trader.start()
                
                connect_result = xt_trader.connect()
                if connect_result != 0:
                    print('连接失败，程序即将重试')
                else:
                    subscribe_result = xt_trader.subscribe(acc)
                    if subscribe_result != 0:
                        print('账号订阅失败 %d' % subscribe_result)
                    print('连接成功')
                    asset = xt_trader.query_stock_asset(acc)
                    if asset:
                        save_stock_asset(asset, db_path)
                        print("证券资产查询保存成功,可用资金：", asset.cash)


                    # 检查计划任务线程是否活跃，如果不活跃则启动，需要修改主程序
                    if schedule_thread is None or not schedule_thread.is_alive():
                        schedule_thread = threading.Thread(target=schedule_jobs())
                        schedule_thread.start()
                    while xt_trader:
                        time.sleep(10)
                        current_time = datetime.now().strftime("%H:%M")
                        if current_time > "16:10":
                            break
                    print("连接断开，即将重新连接")
            else:
                print("当前时间不在运行时段内")
                time.sleep(1800)  # 睡眠半小时后再次检查
        except Exception as e:
            print(f"运行过程中发生错误: {e}")
            time.sleep(1800)  # 如果遇到异常，休息半小时再试












