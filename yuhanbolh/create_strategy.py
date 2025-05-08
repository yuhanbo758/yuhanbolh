import pandas as pd
from pandas import Series, DataFrame
from typing import List
import pandas as pd
import sqlite3
from xtquant import xtdata
import pandas as pd
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from xtquant import xtdata
from datetime import datetime, timedelta
import sqlite3
import pandas as pd

# 策略文件



# 以下是打地鼠策略的函数，参数分别是：数据库路径、数据表名称列表、输出数据表名称
# 读取需要打地鼠的数据库中的持仓数据，并保存到数据表——与策略一起运行，与打地鼠策略无直接关系。
def get_filtered_data(db_path: str, table_names: list, output_table_name: str):
    """
    从SQLite数据库中读取多个表的数据，并筛选出对于每个'证券代码'，
    所有'成交数量'乘以'买卖'的和大于0的行，并将合并结果保存到指定的数据表中。

    :param db_path: 数据库文件的路径。
    :param table_names: 数据表名称的列表。
    :param output_table_name: 输出数据表的名称。
    """
    try:
        # 创建到SQLite数据库的连接
        conn = sqlite3.connect(db_path)
        
        # 用于存储每个表筛选结果的DataFrame列表
        dfs = []

        for table_name in table_names:
            # 读取每个表的数据
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)

            # 计算每个'证券代码'的'成交数量'乘以'买卖'的总和
            df['product'] = df['成交数量'] * df['买卖']
            grouped = df.groupby('证券代码')['product'].sum().reset_index(name='持仓数量')

            # 筛选出总和大于0的'证券代码'
            filtered_codes = grouped[grouped['持仓数量'] > 0]['证券代码']

            # 从原始DataFrame中筛选出具有这些代码的行，并合并计算的总和
            filtered_df = df[df['证券代码'].isin(filtered_codes)]

            # 只保留需要的列
            final_df = filtered_df[['策略名称', '证券代码']].drop_duplicates().merge(grouped, on='证券代码')
            
            # 将筛选后的DataFrame添加到列表中
            dfs.append(final_df)
        
        # 合并所有表的结果
        merged_df = pd.concat(dfs, ignore_index=True)

        # 将合并后的结果保存到指定的数据表中
        merged_df.to_sql(output_table_name, conn, if_exists='replace', index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 关闭数据库连接
        if conn:
            conn.close()

# 从QMT获得行情数据，筛选出符合条件的标的数据
def get_snapshot(code_list: List[str]):
    # 获取标的快照数据
    df = xtdata.get_full_tick(code_list)
    df = DataFrame.from_dict(df).T.reset_index().rename(columns={'index': '证券代码'})

    # 计算标的均价
    bidPrice_columns = ['bid1', 'bid2', 'bid3', 'bid4', 'bid5']
    askPrice_columns = ['ask1', 'ask2', 'ask3', 'ask4', 'ask5']
    df[bidPrice_columns] = df['bidPrice'].apply(Series, index=bidPrice_columns)
    df[askPrice_columns] = df['askPrice'].apply(Series, index=askPrice_columns)

    # 对可能需要转换为float的列进行转换
    float_columns = ['bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'amount', 'lastClose', 'volume']
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['averagePrice'] = (df['bid1'] + df['ask1']) / 2              # 求买1和卖1的平均价
    df.loc[(df['bid1'] == 0) | (df['ask1'] == 0), 'averagePrice'] = df['bid1'] + df['ask1'] # 涨跌停修正

    df.rename(columns={'averagePrice': 'close', 'lastClose': 'pre_close', 'volume': 'vol'}, inplace=True)
    df['amount'] = df['amount'] / 1e4
    df = df[(df.close != 0) & (df.high != 0)] # 现价大于1的标的

    # 计算衍生指标
    df['pct_chg'] = ((df.close / df.pre_close - 1) * 100)   # 今日涨跌幅（%）
    df['max_pct_chg'] = ((df.high / df.pre_close - 1) * 100)    # 最大涨跌幅（%）

    # 展示列,分别表示：代码、买1和卖1平均价、今日涨跌幅（%）、最大涨跌幅（%）、最高价、最低价、昨收价、成交量、成交额（万元）
    display_columns = ['证券代码', 'close', 'pct_chg', 'max_pct_chg', 'high', 'low', 'pre_close', 'vol', 'amount']
    df = df[display_columns]
    return df


# 读取指定数据库表中的'证券代码'列，获取对应的行情数据和持仓量，最后打印合并后的数据。
# 参数分别是：数据库路径、数据表名称、交易对象、账号
def process_and_merge_data(db_path: str, table_name: str, xt_trader: str, acc: str):
    """
    读取指定数据库表中的'证券代码'列，获取对应的行情数据和持仓量，
    并将行情数据与原始表数据以及持仓数据合并，最后打印合并后的数据。

    :param db_path: 数据库文件的路径。
    :param table_name: 读取证券代码的数据表名称。
    :param acc: 账户标识符。
    """
    try:
        # 连接到数据库
        conn = sqlite3.connect(db_path)

        # 从数据库读取证券代码
        query = f"SELECT 证券代码 FROM {table_name}"
        strategy_data = pd.read_sql_query(query, conn)

        # 重新读取完整的数据表
        complete_strategy_data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        # 获取行情数据
        codes = strategy_data['证券代码'].tolist()
        market_data = get_snapshot(codes)  # 确保这个函数返回一个包含'证券代码'列的DataFrame

        # 获取所有持仓信息
        positions = xt_trader.query_stock_positions(acc)  # 获取所有持仓
        positions_data = pd.DataFrame([{'证券代码': pos.stock_code, '持仓量': pos.volume} for pos in positions])

        # 合并策略数据和行情数据
        merged_data = pd.merge(complete_strategy_data, market_data, on='证券代码', how='left')
        # 再将持仓数据合并到已有的合并数据中
        final_merged_data = pd.merge(merged_data, positions_data, on='证券代码', how='left')

        # 打印合并后的数据
        # print(final_merged_data)
        return final_merged_data

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 关闭数据库连接
        conn.close()


# 条件判断，并进行委托，参数分别是：数据库路径、数据表名称、账号、最大回撤跌幅、激活止盈的最大涨幅阈值、执行止盈的最小涨幅阈值、交易对象
def mole_hunting_delegation(db_path: str, table_name: str, acc: str, drawdown: float, active_thres: float, deal_thres: float, xt_trader: str):
    """
    读取指定数据库表中的'证券代码'列，获取对应的行情数据，
    并根据止盈条件和买入条件处理数据，最后返回处理后的数据。

    :param db_path: 数据库文件的路径。
    :param table_name: 读取证券代码的数据表名称。
    :param acc: 账户标识符。
    :param drawdown: 触发止盈的最大回撤跌幅。
    :param active_thres: 激活止盈的最大涨幅阈值。
    :param deal_thres: 执行止盈的最小涨幅阈值。
    :return: 处理后的DataFrame。
    """
    try:
        # 执行函数并获得DataFrame
        df = process_and_merge_data(db_path, table_name, acc)

        # 确保df不是None
        if df is None:
            raise ValueError("Returned DataFrame is None")

        # 判断是否触发止盈条件
        sell_cond = (df['max_pct_chg'] - df['pct_chg']) > drawdown   # 最大涨幅减去今日涨幅大于最大回撤跌幅,即触发上涨止盈后回撤多少止盈
        sell_cond &= df['max_pct_chg'] > active_thres   # 最大涨幅>多少点后，触发移动止盈
        sell_cond &= df['pct_chg'] > deal_thres  # 今日涨幅>多少点后，才可以止盈
        sell_cond &= df['持仓数量'] == df['持仓量']   # 假设只有满仓时才考虑止盈，持仓数量是计划持仓数量，持仓量是实际持仓数量
        sell_df = df[sell_cond]

        # 判断是否触发买入条件（这里需要您定义买入条件）
        buy_cond = (df['max_pct_chg'] - df['pct_chg']) > (drawdown+1.0)   # 最大涨幅减去今日涨幅大于最大回撤跌幅再加1%，以此来确保卖出后买入有1%的收益率
        buy_cond &= df['max_pct_chg'] > active_thres   # 最大涨幅>多少点后，触发移动止盈，只有触发移动止盈，即有了打地鼠之后再会有买回的考虑
        buy_cond &= (df['high'] + df['low'] + df['pre_close'])/3 > df['close']    # （最高价+最低价+昨收价）的平均价大于现价，即买入的价格在平均价上下
        buy_cond &= df['持仓数量'] > df['持仓量']   # 假设无持仓时才考虑买入
        buy_df = df[buy_cond]

        # 执行卖出操作
        if not sell_df.empty:
            sell_df['target_price'] = sell_df['close'] * 0.98
            sell_df['order_volume'] = sell_df['持仓数量']/2   # 假设卖出一半
            sell_df['remark'] = '动态止盈单'
            for index, row in sell_df.iterrows():
                xt_trader.order_stock(acc, row['证券代码'], xtconstant.STOCK_SELL, row['order_volume'], xtconstant.FIX_PRICE, row['target_price'], row['策略名称'], row['remark'])

        # 执行买入操作
        if not buy_df.empty:
            buy_df['target_price'] = buy_df['close'] * 1.02  # 假设以当前价格的102%买入
            buy_df['order_volume'] = buy_df['持仓数量'] - buy_df['持仓量']   # 假设卖出一半
            buy_df['remark'] = '动态买入单'
            for index, row in buy_df.iterrows():
                xt_trader.order_stock(acc, row['证券代码'], xtconstant.STOCK_BUY, row['order_volume'], xtconstant.FIX_PRICE, row['target_price'], row['策略名称'], row['remark'])
        print('打地鼠策略执行完毕')
    except Exception as e:
        print(f"发生错误: {e}")
# 以上是打地鼠策略的函数


# 以下是国债逆回购策略的函数，参数分别是：交易对象、账号
# 比较沪深两市的一天期的买一国债逆回购，选择值大的进行卖出
def place_order_based_on_asset(xt_trader: str, acc: str):
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

if __name__ == "__main__":
    db_path = r'D:\wenjian\python\smart\data\guojin_account.db'


