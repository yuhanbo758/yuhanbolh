# 获取各种金融数据
import akshare as ak
import pandas as pd
import sqlite3
import os
import numpy as np
from xtquant import xtdata
from scipy import stats
import math
import requests
import yfinance as yf
import re
import pywencai
from bs4 import BeautifulSoup
import MetaTrader5 as mt5
from datetime import datetime, timedelta


# 通过类似000001.SZ的代码获取最新数据（东财api），参数3个，分别是：代码（必要），天数，复权类型【可选值包括 0（不复权）、1（前复权）、2（后复权）】
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



# 获取乌龟量化的指数估值数据，没有参数
def turtle_quant_analysis():
    # 设置请求头，模仿浏览器发送请求。
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-encoding': 'gzip, deflate, br'
    }
    # 发送HTTP请求，获取网页内容。
    response = requests.get('https://wglh.com/indicestop', headers=headers)
    response.raise_for_status()  # 如果请求返回失败的状态码，将抛出异常。

    # 使用BeautifulSoup解析HTML内容。
    soup = BeautifulSoup(response.text, 'lxml')

    # 用于存储解析数据的列表。
    data = []

    # 遍历HTML中的表格及其行和单元格。
    for table in soup.select('#table1'):
        for row in table.select('tr'):
            row_data = []
            for cell in row.select('th, td'):
                cell_data = cell.text.strip()
                # 将百分比字符串转换为浮点数。
                if '%' in cell_data:
                    cell_data = float(cell_data.rstrip('%')) / 100
                row_data.append(cell_data)
            data.append(row_data)
        break  # 只处理第一个表格。

    # 使用解析到的数据创建Pandas DataFrame。
    return pd.DataFrame(data[1:], columns=data[0])

# 获取akshare指数估值数据，没有参数
def akshare_index_analysis():
    df = ak.index_value_name_funddb()
    return df

# 通过pywencai模块获取问财数据，参数为：问题和查询类型，以及是否循环分页。
def get_pywencai(question, query_type, loop=True):
    res = pywencai.get(question=f'{question}',query_type =f'{query_type}', loop=True)
    return res

# 获取可转债数据，没有参数
def akshare_convertible_bond():
    df = ak.bond_cb_redeem_jsl()
    return df


# 获取价值大师网的大师价值数据，参数为：股票代码
def get_valuation_ratios(code):
    url = f'https://www.gurufocus.cn/stock/{code}/term/gf_value'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 获取大师价值线
    element_jzx = soup.select_one('#term-page-title').text.strip()
    element_jzx = re.findall("\d+\.\d+", element_jzx)[0]

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


# 获取mt5中的行情数据，参数有3个：品种（必要），时间框架，天数。
def get_mt5_data(symbol, timeframe=mt5.TIMEFRAME_D1, days_back=10):
    # 连接到MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    try:
        # 设置时间范围
        current_time = datetime.now()
        time_ago = current_time - timedelta(days=days_back)
        
        # 获取品种从指定时间前到当前时间的数据
        rates = mt5.copy_rates_range(symbol, timeframe, time_ago, current_time)
        
        # 如果成功获取到数据，进行数据转换
        if rates is not None and len(rates) > 0:
            # 将数据转换为Pandas DataFrame
            df = pd.DataFrame(rates)
            # 转换时间格式
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # 重命名 'tick_volume' 列为 'volume'
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        else:
            print(f"No rates data found for {symbol}")
            df = pd.DataFrame()  # 如果没有数据，则返回一个空的DataFrame
        return df
    except Exception as e:
        print(f"在获取数据时发生错误：{e}")
        return pd.DataFrame()  # 发生异常时返回一个空的DataFrame
    
# 获取mt5中的行情数据，参数有3个：品种（必要），时间框架，天数。