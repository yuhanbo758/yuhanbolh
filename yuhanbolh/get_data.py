# 获取各种金融数据
import akshare as ak
import pandas as pd
import sqlite3
import os
import numpy as np
from xtquant import xtdata
import requests
import re
import pywencai
from bs4 import BeautifulSoup
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import baostock as bs
from pytdx.hq import TdxHq_API
import configparser
from pandas import Series, DataFrame
from typing import List

xtdata.enable_hello = False


# 定义全局变量
global_tdx_ip = None
global_tdx_port = None




def init_global_address(cfg_file):
    """初始化全局服务器地址"""
    global global_tdx_ip, global_tdx_port
    global_tdx_ip, global_tdx_port = get_tdx_market_address(cfg_file)
    return global_tdx_ip, global_tdx_port

# 从通达信配置文件获取行情IP地址
def get_tdx_market_address(cfg_file):
    try:

        if os.path.exists(cfg_file):
            # 读取配置文件
            config = configparser.ConfigParser()
            with open(cfg_file, 'r', encoding='gb2312') as f:
                config.read_file(f)
            
            # 获取主服务器编号
            primary_host = int(config.get('HQHOST', 'PrimaryHost'))
            
            # 获取对应的服务器信息
            ip_key = f'IPAddress{primary_host:02d}'
            port_key = f'Port{primary_host:02d}'
            
            ip = config.get('HQHOST', ip_key)
            port = int(config.get('HQHOST', port_key))
            print(f"行情服务器地址: {ip}:{port}")
            
            return ip, port
            
    except Exception as e:
        print(f"获取行情地址失败：{str(e)}")
        return None, None

# 处理数据
def get_financial_data(server_ip, server_port, data_function):
    try:
        api = TdxHq_API()

        if api.connect(server_ip, server_port):
            data = data_function(api)  # 使用传入的函数获取数据
            if data is None:
                api.disconnect()
                return "没有收到数据。"

            df = api.to_df(data)  # 转换为DataFrame
            
            api.disconnect()
            return df
        else:
            return "连接失败。"
    except Exception as e:
        return f"发生异常: {str(e)}"

# 获取盘口数据(买卖五档)
def get_security_quotes(market_codes):
    """获取盘口数据(买卖五档)
    Args:
        market_codes: 列表，每个元素为元组(市场代码,股票代码)，如[(0,'000001'),(0,'000002')]
    """
    return get_financial_data(global_tdx_ip, global_tdx_port, lambda api: api.get_security_quotes(market_codes))

# 获取K线数据
def get_security_bars(category, market, code, start, count):
    """获取K线数据
    Args:
        category: K线类型（0 5分钟K线; 1 15分钟K线; 2 30分钟K线; 3 1小时K线; 4 日K线; 5 周K线; 6 月K线; 7 1分钟; 8 1分钟K线; 9 日K线; 10 季K线; 11 年K线）
        market: 市场代码（0:深圳, 1:上海）
        code: 股票代码
        start: 起始位置（0为最新）
        count: 数量，最高800
    """
    return get_financial_data(global_tdx_ip, global_tdx_port, lambda api: api.get_security_bars(category, market, code, start, count))

# 获取市场股票数量
def get_security_count(market):
    """获取市场股票数量
    Args:
        market: 市场代码（0:深圳, 1:上海）
    """
    return get_financial_data(global_tdx_ip, global_tdx_port, lambda api: api.get_security_count(market))

# 获取指数K线数据
def get_index_bars(category, market, code, start, count):
    """获取指数K线数据
    Args:
        category: K线类型（0:分时, 1:1分钟, 2:5分钟, 3:15分钟, 4:30分钟, 5:60分钟）
        market: 市场代码（0:深圳, 1:上海）
        code: 指数代码
        start: 起始位置（0为最新）
        count: 数量
    """
    return get_financial_data(global_tdx_ip, global_tdx_port, lambda api: api.get_index_bars(category, market, code, start, count))

# 获取历史分钟数据
def get_history_minute_time_data(market, code, date):
    """获取历史分钟数据
    Args:
        market: 市场代码（0:深圳, 1:上海）
        code: 股票代码
        date: 日期，格式如20241115
    """
    return get_financial_data(global_tdx_ip, global_tdx_port, lambda api: api.get_history_minute_time_data(market, code, date))

# 获取历史分笔成交
def get_transaction_data(market, code, start, count):
    """获取历史分笔成交
    Args:
        market: 市场代码（0:深圳, 1:上海）
        code: 股票代码
        start: 起始位置（0为最新）
        count: 数量
    """
    return get_financial_data(global_tdx_ip, global_tdx_port, lambda api: api.get_transaction_data(market, code, start, count))

# 获取财务数据
def get_finance_info(market, code):
    """获取财务数据
    Args:
        market: 市场代码（0:深圳, 1:上海）
        code: 股票代码
    """
    return get_financial_data(global_tdx_ip, global_tdx_port, lambda api: api.get_finance_info(market, code))



# 获取金融数据文件

# 通过东方财富api获取K线数据，参数包括股票代码（000001.SZ的代码获取最新数），天数，复权类型，K线类型
# `klt`：K 线周期，可选值包括 5（5 分钟 K 线）、15（15 分钟 K 线）、30（30 分钟 K 线）、60（60 分钟 K 线）、101（日 K 线）、102（周 K 线）、103（月 K 线）等。
# `fqt`：复权类型，可选值包括 0（不复权）、1（前复权）、2（后复权）。
def json_to_dfcf(code, days=1, fqt=1, klt=101):     # 参数参考我的东方财富api文档
    if code.endswith("SH"):
        code = "1." + code[:-3]
    else:
        code = "0." + code[:-3]
    try:
        today = datetime.now().date()
        start_time = (today - timedelta(days=days)).strftime("%Y%m%d")
        end_date = today.strftime('%Y%m%d')
        url = f'http://push2his.eastmoney.com/api/qt/stock/kline/get?&secid={code}&fields1=f1,f3&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt={klt}&&fqt={fqt}&beg={start_time}&end={end_date}'
        response = requests.get(url)
        data = response.json()
        data = [x.split(',') for x in data['data']['klines']]
        column_names = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']
        df = pd.DataFrame(data, columns=column_names)

        # 转换列为浮点数
        float_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 将无法转换的值设为NaN
        
        return df
    except Exception as e:
        print(f"发生异常: {e}")
        return None

# 通过类似000001.SZ的代码获取日线数据（东财api），参数3个，分别是：代码（必要），天数，复权类型
def json_to_dfcf_qmt(code, days=7*365, fqt=1):
    if code.endswith("SH"):
        code = "1." + code[:-3]
    else:
        code = "0." + code[:-3]
    try:
        today = datetime.now().date()
        start_time = (today - timedelta(days=days)).strftime("%Y%m%d")
        end_date = today.strftime('%Y%m%d')
        url = f'http://push2his.eastmoney.com/api/qt/stock/kline/get?&secid={code}&fields1=f1,f3&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt={fqt}&beg={start_time}&end={end_date}'
        response = requests.get(url)
        data = response.json()
        data = [x.split(',') for x in data['data']['klines']]
        column_names = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']
        df = pd.DataFrame(data, columns=column_names)

        # 转换列为浮点数
        float_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 将无法转换的值设为NaN
        
        return df
    except Exception as e:
        print(f"发生异常: {e}")
        return None

# 与上面的函数相同，只是通数days参数为天数，而不是日期。比如100表示100个交易日的数据，而不是日期往前推100天。
def json_to_dfcf_qmt_jyr(code, days=7*250, fqt=1):
    if code.endswith("SH"):
        code = "1." + code[:-3]
    else:
        code = "0." + code[:-3]
    try:
        today = datetime.now().date()
        end_date = today.strftime('%Y%m%d')
        url = f'http://push2his.eastmoney.com/api/qt/stock/kline/get?&secid={code}&fields1=f1,f3&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt={fqt}&end={end_date}&lmt={days}'
        response = requests.get(url)
        data = response.json()
        data = [x.split(',') for x in data['data']['klines']]
        column_names = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']
        df = pd.DataFrame(data, columns=column_names)

        # 转换列为浮点数
        float_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 将无法转换的值设为NaN
        
        return df
    except Exception as e:
        print(f"发生异常: {e}")
        return None

# 从baostock获取股票数据，参数有4个：股票代码，周期，复权类型，指标列表
# adjustflag默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据
# adjustflag复权类型：不复权：3；后复权：1；前复权：2
def query_stock_data(stock_code, days_back=60, frequency="d", adjustflag="2"):
    try:
        # 登陆系统
        lg = bs.login()

        # 计算开始日期和结束日期
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        # 获取沪深A股历史K线数据
        rs = bs.query_history_k_data_plus(stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM",
            start_date=start_date, end_date=end_date,
            frequency=frequency, adjustflag=adjustflag)
        
        # 打印结果集
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        # 登出系统
        bs.logout()

        return result

    except Exception as e:
        return f"An exception occurred: {str(e)}"

# 通过qmt获取证券的7年K线历史数据，不包数据下载补充
def qmt_data_source(stock_code, days=7*365):

    # 计算7年前的日期
    start_time = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    # xtdata.download_history_data2([stock_code], period='1d', start_time=start_time, callback=on_progress)

    field_list = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
    
    # 从新的数据源获取数据
    data = xtdata.get_market_data(field_list, [stock_code], period='1d', start_time=start_time, count=-1, dividend_type='front', fill_data=True)
    
    # 转置每个字段并连接在一起
    data_transposed = pd.concat([data[field].T for field in field_list], axis=1)
    data_transposed.columns = field_list
    data_transposed.reset_index(drop=True, inplace=True) # 重置索引
    
    # 将时间戳转换为日期字符串
    data_transposed['time'] = pd.to_datetime(data_transposed['time'], unit='ms') + pd.Timedelta(hours=8) # 加上时区偏移
    data_transposed['time'] = data_transposed['time'].dt.strftime('%Y-%m-%d')
    
    return data_transposed


# 获取国金qmt行情数据的补充数据
def on_progress(data):
    '''补充历史数据回调函数'''
    print(data) 

# 通过qmt获取证券的7年K线历史数据，包含数据下载补充
def qmt_data_source_download(stock_code, days=7*365):

    # 计算7年前的日期
    start_time = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    xtdata.download_history_data2([stock_code], period='1d', start_time=start_time, callback=on_progress)

    field_list = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
    
    # 从新的数据源获取数据
    data = xtdata.get_market_data(field_list, [stock_code], period='1d', start_time=start_time, count=-1, dividend_type='front', fill_data=True)
    
    # 转置每个字段并连接在一起
    data_transposed = pd.concat([data[field].T for field in field_list], axis=1)
    data_transposed.columns = field_list
    data_transposed.reset_index(drop=True, inplace=True) # 重置索引
    
    # 将时间戳转换为日期字符串
    data_transposed['time'] = pd.to_datetime(data_transposed['time'], unit='ms') + pd.Timedelta(hours=8) # 加上时区偏移
    data_transposed['time'] = data_transposed['time'].dt.strftime('%Y-%m-%d')
    
    return data_transposed

# 从国金qmt中获取指定代码的近7年行情数据
def download_7_years_data(stock_list):
    # 计算7年前的日期
    start_time = (datetime.now() - timedelta(days=7*365)).strftime("%Y%m%d")

    # 下载近7年的历史数据
    xtdata.download_history_data2(stock_list, period='1d', start_time=start_time, callback=on_progress)
    



# 获取akshare指数估值数据，没有参数
def akshare_index_analysis():
    df = ak.index_value_name_funddb()
    return df

# 获取韭圈儿的指数估值数据
def index_value_name_funddb():
    temp_df = ak.index_value_name_funddb()
    return temp_df

# 通过pywencai模块获取问财数据，参数为：问题和查询类型，以及是否循环分页。
def get_pywencai(question, query_type, loop=True):
    res = pywencai.get(question=f'{question}',query_type =f'{query_type}', loop=True)
    return res

# 获取可转债数据，没有参数
def akshare_convertible_bond():
    df = ak.bond_cb_redeem_jsl()
    return df


# 通过问财的问询方式爬取问财的数据——条件：满足强赎
def get_satisfy_redemption(query):
    try:
        data = pywencai.get(query=query, query_type='conbond', loop=True)
        
        # 指定需要查找和重命名的列名
        col_names_to_change = ["可转债代码", "可转债简称", "最新价", "正股代码", "正股简称", "强赎天计数"]
        for name in col_names_to_change:
            col_name = [col for col in data.columns if name in col]
            if col_name:
                # 重命名列名
                data.rename(columns={col_name[0]: name}, inplace=True)
            else:
                # 若未找到，则创建一个新的列，所有值都为空
                data[name] = np.nan
        
        # 只保留指定的列
        data = data[col_names_to_change]
        return data
    except Exception as e:
        print(f"获取满足强赎数据时出错: {e}")
        return pd.DataFrame()
    
# 通过问财的问询方式爬取问财的数据——条件：可转债策略
def wencai_conditional_query(query):
    try:
        data = pywencai.get(query=query, query_type='conbond', loop=True)
        
        # 指定需要查找和重命名的列名
        col_names_to_change = ["可转债代码", "可转债简称", "最新价", "正股代码", "正股简称", "纯债价值", "期权价值", "最新变动后余额", "转股溢价率", "转股价值"]
        for name in col_names_to_change:
            col_name = [col for col in data.columns if name in col]
            if col_name:
                # 重命名列名
                data.rename(columns={col_name[0]: name}, inplace=True)
            else:
                # 若未找到，则创建一个新的列，所有值都为空
                data[name] = np.nan
        
        # 只保留指定的列
        data = data[col_names_to_change]
        return data
    except Exception as e:
        print(f"获取可转债策略数据时出错: {e}")
        return pd.DataFrame()


# 通过问财的问询方式爬取问财的数据
def get_clean_data(question):
    try:
        data = pywencai.get(question=question, query_type='conbond', loop=True)
        
        # 查找列名中包含特定关键字的列并进行重命名
        col_names_to_change = ["可转债代码", "可转债简称", "涨跌幅", "最新价", "正股代码", "正股简称", "纯债价值", "期权价值", "最新变动后余额", "转股溢价率", "满足强赎", "强赎天计数"]
        for name in col_names_to_change:
            col_name = [col for col in data.columns if name in col]
            if col_name:
                # 重命名列名
                data.rename(columns={col_name[0]: name}, inplace=True)
            else:
                # 若未找到，则创建一个新的列，所有值都为空
                data[name] = np.nan
        
        # 只保留指定的列
        data = data[col_names_to_change]
        return data
    except Exception as e:
        print(f"获取清洗后的数据时出错: {e}")
        return pd.DataFrame()




# 获取集思录的可转债强赎数据，并保存到数据库
def filter_bond_cb_redeem_data_and_save_to_db():
    try:
        # 获取数据
        bond_cb_redeem_jsl_df = ak.bond_cb_redeem_jsl()

        # 提取需要的列
        selected_columns = ['代码', '名称', '现价', '正股代码', '正股名称', '强赎状态']
        filtered_df = bond_cb_redeem_jsl_df[selected_columns]

        # 更改列名
        new_column_names = ['可转债代码', '可转债简称', '最新价', '正股代码', '正股简称', '强赎天计数']
        filtered_df.columns = new_column_names

        # 使用str.contains方法筛选出“已公告强赎”和“公告要强赎”的行
        # filtered_rows = filtered_df[filtered_df['强赎天计数'].str.contains('已公告强赎|公告要强赎')]
        filtered_rows = filtered_df[filtered_df['强赎天计数'].str.contains('已公告强赎|公告要强赎')].copy()

        # 修改“可转债代码”列的数据
        filtered_rows.loc[filtered_rows['可转债代码'].str.startswith('11'), '可转债代码'] = filtered_rows['可转债代码'] + '.SH'
        filtered_rows.loc[filtered_rows['可转债代码'].str.startswith('12'), '可转债代码'] = filtered_rows['可转债代码'] + '.SZ'
        
        # 连接到数据库
        db_path = r'D:\wenjian\python\smart\data\guojin_account.db'
        conn = sqlite3.connect(db_path)

        # 将数据保存到数据库表
        table_name = "满足赎回可转债"
        filtered_rows.to_sql(table_name, conn, if_exists="replace", index=False)

        print("数据成功保存到数据库")
    except Exception as e:
        print(f"保存数据到数据库时出错: {e}")
    finally:
        # 无论是否出现异常都关闭数据库连接
        conn.close()

# 获取价值大师网的大师价值数据，参数为：股票代码
def get_valuation_ratios(code):
    url = f'https://www.gurufocus.cn/stock/{code}/term/gf_value'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 获取大师价值线
    element_jzx = soup.select_one('#term-page-title').text.strip()
    element_jzx = re.findall(r"\d+\.\d+", element_jzx)[0]

    # 获取名称
    element_mc = soup.select_one('html body div div main div:nth-child(1) div:nth-child(2) div:nth-child(1) div:nth-child(1) div:nth-child(2) div:nth-child(1) h1 span:nth-child(1)').text.strip()

    # 获取现价
    element_xj = soup.select_one('html body div div main div:nth-child(1) div:nth-child(2) div:nth-child(1) div:nth-child(1) div:nth-child(2) div:nth-child(2)')
    element_xj.span.decompose()  # 去掉span标签
    element_xj = re.findall(r'\d+\.\d+', element_xj.text.strip())[0]  # 只保留数字和小数点

    # 拼接链接
    link = f'https://www.gurufocus.cn/stock/{code}/summary'

    # 创建一个pandas DataFrame来保存比率
    ratios = pd.DataFrame({
        '名称': [element_mc],
        '大师价值': [element_jzx],
        '现价': [element_xj],
    })

    return ratios


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



# 获取持仓和收盘价，从而获得整体的交易日期和现金流，然后保存到数据库到r"D:\wenjian\python\smart\data\guojin_account.db"
def position_close_process_data(table_name, db_path = r"D:\wenjian\python\smart\data\guojin_account.db"):
    
    conn = sqlite3.connect(db_path)

    query = f"SELECT * FROM {table_name}"
    data_from_db = pd.read_sql_query(query, conn)

    conn.close()

    data = calculate_unhedged_transactions(db_path, [table_name])

    for index, row in data.iterrows():
        market_data = json_to_dfcf_qmt(row['证券代码'], 5)
        close_price = market_data['close'].iloc[-1]
        data.at[index, '成交时间'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        data.at[index, '成交均价'] = close_price
        data.at[index, '成交金额'] = close_price * row['成交数量']
        data.at[index, '买卖'] = -1

    merged_data = pd.concat([data_from_db, data], ignore_index=True)

    # 创建新的数据库连接
    new_db_path = r"D:\wenjian\python\smart\data\guojin_account.db"
    new_conn = sqlite3.connect(new_db_path)

    # 新表名为原表名加"_hedge"
    new_table_name = f"{table_name}_hedge"


    # 将merged_data保存到新的数据库
    merged_data.to_sql(new_table_name, new_conn, if_exists='replace', index=False)
    

    # 关闭新的数据库连接
    new_conn.close()

    return merged_data


# 从东方财富网的API获取指定股票代码的汇率信息，并提取汇率数据，常用
def get_exchange_rate(secid):
    """
    从东方财富网的API获取指定股票代码的汇率信息，并提取汇率数据。

    参数:
    secid: str
        股票代码，格式为 "市场代码.股票代码"，例如 "133.USDCNH"。

    返回:
    exchange_rate: float
        提取的汇率数据。
    """
    # 构建 API URL
    api_url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get?" \
              f"&secid={secid}&fields1=f1,f2,f3,f4,f5,f6,f7,f8" \
              f"&fields2=f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61" \
              f"&klt=101&fqt=0&end=20280314&lmt=1"

    try:
        # 发送 GET 请求到 API URL
        response = requests.get(api_url)

        # 检查请求是否成功
        if response.status_code == 200:
            # 解析响应的 JSON 数据
            data = response.json()
            # 导航至 JSON 数据以找到所需数据
            kline_data = data.get("data", {}).get("klines", [])
            if kline_data:
                # 从 kline 字符串中提取第二个值（汇率）
                exchange_rate = kline_data[0].split(',')[2]
                return float(exchange_rate)
            else:
                return "数据未找到"
        else:
            return "请求失败，状态码：" + str(response.status_code)
    except Exception as e:
        return "请求过程中发生异常：" + str(e)

# 获取保存汇率数据到数据库，常用
def save_exchange_rates_to_db(db_path, table_name):
    """
    获取所需的所有汇率，计算兑换到CNH的汇率，并将结果保存到数据库中。

    参数:
    db_path: str
        数据库文件的路径。
    table_name: str
        数据库中的表名，用于存储汇率数据。
    """
    # 从 API 或模拟函数中获取所有需要的汇率
    rate_identifiers = ["119.USDJPY", "119.EURGBP", "119.EURAUD", "119.GBPAUD", "119.GBPUSD", "119.EURUSD", "133.USDCNH"]
    rates = {secid.split('.')[1]: get_exchange_rate(secid) for secid in rate_identifiers}

    # 计算其他货币对人民币的汇率
    rates["JPYCNH"] = (1 / rates["USDJPY"]) * rates["USDCNH"]
    rates["GBPCNH"] = (1 / rates["EURGBP"]) * rates["EURUSD"] * rates["USDCNH"]
    rates["AUDCNH"] = (1 / rates["EURAUD"]) * rates["EURUSD"] * rates["USDCNH"]
    rates["EURCNH"] = rates["EURUSD"] * rates["USDCNH"]

    # 将汇率数据转换为 DataFrame
    df_rates = pd.DataFrame(list(rates.items()), columns=["货币对", "汇率"])

    # 连接到 SQLite 数据库，并将汇率数据保存到指定的表中
    conn = sqlite3.connect(db_path)
    df_rates.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

    # 输出一条信息，确认数据已被保存
    print("汇率数据已经成功保存到数据库表：" + table_name)


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


# 读取指定数据库表中的'证券代码'列，获取对应的行情数据和持仓量，最后打印并合并后的数据。
def process_and_merge_data(db_path: str, table_name: str, acc: str):
    """
    读取指定数据库表中的'证券代码'列，获取对应的行情数据和持仓量，
    并将行情数据与原始表数据以及持仓数据合并，最后打印合并后的数据。

    :param db_path: 数据库文件的路径。
    :param table_name: 读取证券代码的数据表名称。
    :param acc: 账户标识符。
    """
    try:
        xt_trader = None
        # 连接到数据库
        conn = sqlite3.connect(db_path)

        # 从数据库读取证券代码
        query = f"SELECT 证券代码 FROM {table_name}"
        strategy_data = pd.read_sql_query(query, conn)

        if strategy_data is None or strategy_data.empty:
            # print('没有可用数据')
            return

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


# 获取东方财富全球财经快讯，共200条，只保留摘要
def stock_info_global_em() -> pd.DataFrame:
    """
    东方财富-全球财经快讯
    https://kuaixun.eastmoney.com/7_24.html
    :return: 全球财经快讯摘要
    :rtype: pandas.DataFrame
    """
    url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
    params = {
        "client": "web",
        "biz": "web_724",
        "fastColumn": "102",
        "sortEnd": "",
        "pageSize": "200",
        "req_trace": "1710315450384",
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json["data"]["fastNewsList"])
    temp_df = temp_df[["summary"]]
    temp_df.rename(
        columns={
            "summary": "摘要",
        },
        inplace=True,
    )
    return temp_df



