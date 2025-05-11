
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os
import time
import numpy as np
import os
import datetime
from scipy import stats
import math
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pywencai
import random




# 从问财网获取纳斯达克100指数成分股数据，参数查询语句，每年的1月1日运行一次，因为纳斯达克100指数成分股每年年底会有变化
def wencai_conditional_query_nz100(query):
    try:
        data = pywencai.get(query=query, query_type='usstock', loop=True)
        
        # 指定需要查找和重命名的列名
        col_names_to_change = ["股票代码", "股票简称", "指数"]
        for name in col_names_to_change:
            col_name = [col for col in data.columns if name in col]
            if col_name:
                # 重命名列名
                data.rename(columns={col_name[0]: name}, inplace=True)
            else:
                # 若未找到，则创建一个新的列，所有值都为空
                data[name] = np.nan

        # 删除 '股票代码' 中的 '.0' 并保存为 '价值代码'
        data['价值代码'] = data['股票代码'].str.replace('.O', '', regex=False)

        # 将 '股票代码' 中的 '.O' 替换为 '.NYSE' 并保存为 'mt5代码'
        data['mt5代码'] = data['股票代码'].str.replace('.O', '.NAS', regex=False)
        
        # 只保留指定的列
        data = data[col_names_to_change + ['价值代码', 'mt5代码']]
        
        # 数据库文件路径
        db_path = "D:/wenjian/python/smart/data/mt5.db"
        
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_path)
        
        # 将DataFrame保存到SQLite数据库中，表名为“nasdaq_100”
        data.to_sql('nasdaq_100', conn, if_exists='replace', index=False)
        
        # 提交事务
        conn.commit()
        
        # 关闭连接
        conn.close()
        
        return data
    except Exception as e:
        print(f"获取纳斯达克100指数成分股数据时出错: {e}")
        return pd.DataFrame()
    

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



# 通过代码，获取价值大师的价值线，参数是代码，例如000001
def get_valuation_ratios(code):
    try:
        url = 'https://www.gurufocus.cn/stock/{}/term/gf_value'.format(code)

        # 设置随机的User-Agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        ]
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document'
        }


        # 添加随机延迟
        time.sleep(random.uniform(1, 5))

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            # 使用BeautifulSoup解析网页内容
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 获取文本内容
            text = soup.get_text()
            # print(text)
            
            # 去掉换行符，保留空格
            text = re.sub(r'\n+', ' ', text)
            # print(text)

            # 尝试使用下面的代码提取数据
            element_mc = None
            element_xj = None
            element_data = None
            element_pj = None
            element_jzxs = None

            # 定义匹配模式，查找“大师价值 为”后面的数值
            # 根据code的前两位决定匹配模式
            if code[:2] == 'HK':
                pattern_master_value = re.compile(r'大师价值 : HK\$ [\d\.]+ \(今日\)[\s\S]*?HK\$ (\d+\.\d+)')
            else:
                pattern_master_value = re.compile(r'大师价值 : [¥$] ([\d\.]+)')
                
            element_jzxs = pattern_master_value.findall(text)   # 大师价值
            if element_jzxs:
                element_jzxs = element_jzxs[0]

            # 正则表达式匹配从"登录/注册价值大师/股票列表/"开始，到"价值大师评分"结束的文本
            pattern = re.compile(r'登录/注册价值大师/股票列表/.*?价值大师评分')
            match = pattern.search(text)

            if match:
                extracted_text = match.group(0)
                # print(extracted_text)
                
                name_pattern = re.compile(r'/([\u4e00-\u9fa5]+)\(')  # 匹配形如 /平安银行( 或 /苹果( 的名称
                name_match = name_pattern.search(extracted_text)
                element_mc = name_match.group(1) if name_match else None
                
                price_pattern = re.compile(r'[¥$]([\d\.]+)')
                price_match = price_pattern.search(extracted_text)
                element_xj = price_match.group(1) if price_match else None
                
                # 提取 element_xj 后面的第三个空格到第四个空格的内容
                data_pattern = re.compile(r'[¥$]{}\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\s+'.format(element_xj))
                data_match = data_pattern.search(extracted_text)
                if data_match:
                    element_data = data_match.group(3)
                    # print(element_data)
                else:
                    element_data = None

                # 提取 element_xj 的第六个到第七个空格的内容
                pj_pattern = re.compile(r'[¥$]{}\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\s+'.format(element_xj))
                pj_match = pj_pattern.search(extracted_text)
                if pj_match:
                    element_pj = pj_match.group(6)
                    # print(element_pj)
                else:
                    element_pj = None

            # 如果下面的代码没有提取到数据，则使用上面的代码进行提取
            if not all([element_mc, element_xj, element_data, element_pj]):
                # 定义匹配模式，查找“当前股价 为”后面的数值
                pattern_current_price = re.compile(r'当前股价 为 [¥$]([\d\.]+)')

                # 定义匹配模式，查找“的评级是：”后面的文字，直到遇到下一个空格或标点符号
                pattern_rating = re.compile(r'评级是： ([^ ，。]+)')

                # 查找所有匹配的数值
                element_xj = pattern_current_price.findall(text)   # 价值现价
                element_pj = pattern_rating.findall(text)   # 评价

                # 如果提取到数据，则更新变量
                if element_xj:
                    element_xj = element_xj[0]
                if element_pj:
                    element_pj = element_pj[0]

            # 计算价值估值
            try:
                if element_pj == "股价被严重高估":
                    Fraction_pj = -2
                elif element_pj == "股价被高估":
                    Fraction_pj = -1
                elif element_pj == "股价被低估":
                    Fraction_pj = 1
                elif element_pj == "股价被严重低估":
                    Fraction_pj = 2
                else:
                    Fraction_pj = 0
            except:
                Fraction_pj = 0

            # 创建DataFrame
            ratios = pd.DataFrame({
                '价值代码': [code],
                '价值名称': [element_mc],
                '价值现价': [element_xj],
                '大师价值': [element_jzxs],
                '评价': [element_pj],
                '价值估值': [Fraction_pj],
                '基本数据': [element_data],
            })

            print(ratios)
            return ratios
        else:
            print('请求失败:', response.status_code)
    except Exception as e:
        print(f'发生错误: {e}')



# 通过读取excel中的列“代码”，从而获取价值大师价格
def get_all_valuation_ratios_db(db_path, table_name):
    """
    从数据库表中读取“价值代码”，然后获取相应的价值大师价格。

    参数:
    db_path: str
        数据库文件的路径。
    table_name: str
        包含“价值代码”的数据库表名。

    返回:
    all_ratios: DataFrame
        包含所有价值大师比率数据的 DataFrame。
    """
    try:
        # 尝试连接到 SQLite 数据库
        conn = sqlite3.connect(db_path)
        
        # 尝试执行 SQL 查询以获取“价值代码”列，并将其转换为列表
        query = f"SELECT `价值代码` FROM `{table_name}`"
        codes = pd.read_sql_query(query, conn)['价值代码'].astype(str).tolist()
    except Exception as e:
        print(f"Database connection or query failed: {e}")
        return pd.DataFrame()  # 返回一个空的 DataFrame

    all_ratios = pd.DataFrame()  # 创建一个空的 pandas DataFrame 来保存所有的比率数据
    for code in codes:  # 遍历所有代码并获取比率数据
        try:
            ratios = get_valuation_ratios(code)  # 假设这是一个自定义函数，用于获取价值大师比率
            all_ratios = pd.concat([all_ratios, ratios])  # 将获取的数据拼接到 DataFrame
        except Exception as e:
            print(f"Error occurred for code {code}: {e}")
            continue  # 发生错误时继续处理下一个代码

    # 关闭数据库连接
    conn.close()
    return all_ratios



# 从MT5获取指定品种的8年历史日线数据
def get_mt5_data(symbol):
    """
    从MT5获取指定品种的历史日线数据，并返回包含所有数据的DataFrame。

    参数:
    symbol: str
        希望获取数据的市场品种的符号。
    
    返回:
    df: DataFrame
        包含历史日线数据的DataFrame。
    """
    try:
        # 设置时间范围
        timezone = mt5.TIMEFRAME_D1  # 日线数据
        current_time = datetime.now()
        one_year_ago = current_time - timedelta(days=365*8)
        
        # 获取品种从8年前到当前时间的日线数据
        rates = mt5.copy_rates_range(symbol, timezone, one_year_ago, current_time)
        
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
    

# 从MT5获取指定品种的历史日线数据，日期长度可自定义，参数分别是交易品种和天数
def get_mt5_data_with_days(symbol, days=365*8):
    """
    从MT5获取指定品种的历史日线数据，日期长度可自定义，并返回包含所有数据的DataFrame。

    参数:
    symbol: str
        希望获取数据的市场品种的符号。
    days: int
        希望获取的历史数据天数，默认为8年(365*8天)。
    
    返回:
    df: DataFrame
        包含历史日线数据的DataFrame。
    """
    try:
        # 设置时间范围
        timezone = mt5.TIMEFRAME_D1  # 日线数据
        current_time = datetime.now()
        start_time = current_time - timedelta(days=days)
        
        # 获取品种从指定天数前到当前时间的日线数据
        rates = mt5.copy_rates_range(symbol, timezone, start_time, current_time)
        
        # 如果成功获取到数据，进行数据转换
        if rates is not None and len(rates) > 0:
            # 将数据转换为Pandas DataFrame
            df = pd.DataFrame(rates)
            # 转换时间格式
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # 重命名 'tick_volume' 列为 'volume'
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        else:
            print(f"未找到 {symbol} 的数据")
            df = pd.DataFrame()  # 如果没有数据，则返回一个空的DataFrame
        return df
    except Exception as e:
        print(f"在获取数据时发生错误：{e}")
        return pd.DataFrame()  # 发生异常时返回一个空的DataFrame


# 获取data数据中的第几行数据
def get_row(data, index):
    return data.iloc[[index]].reset_index(drop=True)


# 获取data数据中的第几行数据
def get_stock_list_from_db():
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 执行查询，获取“代码”列的数据
    query = "SELECT 代码 FROM 指数价值"
    cursor.execute(query)
    results = cursor.fetchall()

    # 将结果放入stock_list中
    stock_list = [row[0] for row in results]

    # 关闭连接
    conn.close()

    return stock_list


def MA_zb(data, n):
    MA = pd.Series(data['close'].rolling(n).mean(), name='MA_' + str(n))
    close = data['close']
    signal = np.where(MA < close, 1, np.where(MA > close, -1, 0))
    return pd.DataFrame(signal, columns=['MA_' + str(n)])  # 修改这行


# 获取指数移动平均线，参数有2个，一个是数据源，一个是日期
def EMA_zb(data, n):
    EMA = pd.Series(data['close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    close = data['close']
    signal = np.where(EMA < close, 1, np.where(EMA > close, -1, 0))
    return pd.DataFrame(signal, columns=['EMA_' + str(n)])


# 获取一目均衡表基准线 (data, conversion_periods, base_periods, lagging_span2_periods, displacement)
# 参数有5个，第一个是数据源，其他4个分别是一目均衡表基准线 (9, 26, 52, 26)，即ichimoku_cloud(data,9, 26, 52, 26)
def ichimoku_cloud_zb(data, conversion_periods, base_periods, lagging_span2_periods, displacement):
    def donchian(length):
        return (data['high'].rolling(length).max() + data['low'].rolling(length).min()) / 2
    
    conversion_line = donchian(conversion_periods)
    base_line = donchian(base_periods)
    lead_line1 = (conversion_line + base_line) / 2
    lead_line2 = donchian(lagging_span2_periods)
    
    lagging_span = data['close'].shift(-displacement + 1).shift(25).ffill()
    leading_span_a = lead_line1.shift(displacement - 1)
    leading_span_b = lead_line2.shift(displacement - 1)
    # 计算买入和卖出信号
    buy_signal = (base_line < data['close']) & (conversion_line > data['close'].shift(1)) & (conversion_line <= data['close']) & (lead_line1 > data['close']) & (lead_line1 > lead_line2)
    sell_signal = (base_line > data['close']) & (conversion_line < data['close'].shift(1)) & (conversion_line >= data['close']) & (lead_line1 < data['close']) & (lead_line1 < lead_line2)
    # 使用np.where生成信号列，买入为1，卖出为-1，中立为0
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame(signal, columns=['Ichimoku'])



# 成交量加权移动平均线 VWMA (data, 20)，参数有2个，1个是数据源，另一个是日期，通过为20
def VWMA_zb(data, n):
    # 计算VWMA
    vwma = (data['close'] * data['volume']).rolling(n).sum() / data['volume'].rolling(n).sum()
    
    # 根据VWMA和收盘价计算买卖信号
    buy_signal = vwma < data['close']
    sell_signal = vwma > data['close']
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame(signal, columns=['VWMA_' + str(n)])


# 计算Hull MA船体移动平均线 Hull MA (data,9)，参数有2，一个是数据源，另一个是日期，一般为9
def HullMA_zb(data, n=9):
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    source = data['close']
    wma1 = wma(source, n // 2) * 2
    wma2 = wma(source, n)
    hullma = wma(wma1 - wma2, int(math.floor(math.sqrt(n))))

    # 根据HullMA和收盘价计算买卖信号
    buy_signal = hullma < source
    sell_signal = hullma > source
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame(signal, columns=['HullMA_' + str(n)])

# 计算RSI指标，参数有2，一个为数据源，另一个为日期，一般为14，即RSI(data, 14)
def RSI_zb(data, n):
    lc = data['close'].shift(1)
    diff = data['close'] - lc
    up = diff.where(diff > 0, 0)
    down = -diff.where(diff < 0, 0)
    ema_up = up.ewm(alpha=1/n, adjust=False).mean()
    ema_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - 100 / (1 + rs)

    rsi_prev = rsi.shift(1)  # 前一个周期的RSI
    buy_signal = (rsi < 30) & (rsi > rsi_prev)
    sell_signal = (rsi > 70) & (rsi < rsi_prev)
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame({'RSI_' + str(n): signal})

# 计算Stochastic，k是主线，d_signal是信号线，参数有4，一个是数据源，另外三个为日期，一般为STOK(data, 14, 3, 3)
def STOK_zb(data, n, m, t):
    # 计算过去n天的最高价
    high = data['high'].rolling(n).max()
    # 计算过去n天的最低价
    low = data['low'].rolling(n).min()
    # 计算%K值
    k = 100 * (data['close'] - low) / (high - low)
    # 使用m天的滚动平均计算%D值
    d = k.rolling(m).mean()
    # 使用t天的滚动平均计算%D_signal值
    d_signal = d.rolling(t).mean()

    main_line = d
    signal_line = d_signal
    main_line_prev = d.shift(1)
    signal_line_prev = d_signal.shift(1)

    buy_signal = (main_line < 20) & (main_line_prev < signal_line_prev) & (main_line > signal_line)
    sell_signal = (main_line > 80) & (main_line_prev > signal_line_prev) & (main_line < signal_line)
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame({'STOK_' + str(n): signal})



# 计算CCI指标，参数有2，一个是数据源，另一个是日期，一般为20，即CCI(data, 20)
def CCI_zb(data, n):
    TP = (data['high'] + data['low'] + data['close']) / 3
    MA = TP.rolling(n).mean()
    MD = TP.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean())
    CCI = (TP - MA) / (0.015 * MD)

    CCI_prev = CCI.shift(1) # 前一个周期的CCI
    
    buy_signal = (CCI < -100) & (CCI > CCI_prev)    # 买入信号
    sell_signal = (CCI > 100) & (CCI < CCI_prev)    # 卖出信号
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame({'CCI_' + str(n): signal})


# 平均趋向指数ADX(14)，参数有2，一个是数据源，另一个是日期，一般为14，即ADX(data,14)
def ADX_zb(data, n):
    # 计算当前最高价与前一天最高价的差值，以及前一天最低价与当前最低价的差值
    up = data['high'] - data['high'].shift(1)
    down = data['low'].shift(1) - data['low']

    # 判断哪些是上涨天，哪些是下跌天
    plusDM = np.where((up > down) & (up > 0), up, 0)
    minusDM = np.where((down > up) & (down > 0), down, 0)

    # 计算真实波幅
    truerange = np.maximum(data['high'] - data['low'], np.maximum(np.abs(data['high'] - data['close'].shift()), np.abs(data['low'] - data['close'].shift())))
    
    # 计算+DI和-DI
    plus = 100 * pd.Series(plusDM).ewm(alpha=1/n, min_periods=n).mean() / pd.Series(truerange).ewm(alpha=1/n, min_periods=n).mean()
    minus = 100 * pd.Series(minusDM).ewm(alpha=1/n, min_periods=n).mean() / pd.Series(truerange).ewm(alpha=1/n, min_periods=n).mean()

    # 计算ADX
    adx = 100 * (np.abs(plus - minus) / (plus + minus)).ewm(alpha=1/n, min_periods=n).mean()

    # 根据您给出的条件定义买入和卖出信号
    adx_plusDI = (adx > 20) & (plus > minus) & (plus.shift(1) < minus.shift(1))
    adx_minusDI = (adx > 20) & (plus < minus) & (plus.shift(1) > minus.shift(1))

    # 计算信号值：买入为1，卖出为-1，中立为0
    signal = np.where(adx_plusDI, 1, np.where(adx_minusDI, -1, 0))

    # 返回包含所有历史数据的DataFrame
    return pd.DataFrame({'ADX_' + str(n): signal})


# 计算动量震荡指标(AO)，参数只有一个，即数据源
def AO_zb(data):
    # 计算Awesome Oscillator（AO）
    AO = (data['high'].rolling(5).mean() + data['low'].rolling(5).mean()) / 2 - (data['high'].rolling(34).mean() + data['low'].rolling(34).mean()) / 2

    # 定义茶碟形买入的条件
    AO_plus_saucer = (AO.shift(2) < AO.shift(1)) & (AO.shift(1) < AO) & (AO > 0)
    # 定义向上穿过零线的条件
    AO_plus_cross = (AO.shift(1) < 0) & (AO > 0)

    # 定义茶碟形卖出的条件
    AO_minus_saucer = (AO.shift(2) > AO.shift(1)) & (AO.shift(1) > AO) & (AO < 0)
    # 定义向下穿过零线的条件
    AO_minus_cross = (AO.shift(1) > 0) & (AO < 0)

    # 根据条件计算买入、卖出和中立的信号
    signal = np.where(AO_plus_saucer | AO_plus_cross, 1, np.where(AO_minus_saucer | AO_minus_cross, -1, 0))

    # 返回包含所有历史数据的DataFrame
    return pd.DataFrame({'AO': signal
    })



# 计算动量指标(10)，参数只有一个，即数据源
def MTM_zb(data):
    # 计算MTM值：当前收盘价与10天前的收盘价之差
    MTM = data['close'] - data['close'].shift(10)
    
    # 根据MTM值的变化确定信号
    # MTM值上升时，信号为1（买入）
    # MTM值下跌时，信号为-1（卖出）
    # MTM值无变化时，信号为0（中立）
    signal = np.where(MTM > MTM.shift(1), 1, np.where(MTM < MTM.shift(1), -1, 0))
    
    # 返回包含所有历史数据的DataFrame
    return pd.DataFrame({'MTM': signal})

# MACD_1是以金叉和死叉进行判断，参数有3个，第一个是数据源，其余两个为日期，一般取12和26，即MACD(data, 12,26)
def MACD_Level_zb(data, n_fast, n_slow):
     # 计算快速EMA
    EMAfast = data['close'].ewm(span=n_fast, min_periods=n_slow).mean()
    
    # 计算慢速EMA
    EMAslow = data['close'].ewm(span=n_slow, min_periods=n_slow).mean()
    
    # 计算MACD值，即快速EMA与慢速EMA之差
    MACD = EMAfast - EMAslow
    
    # 计算MACD的信号线值
    MACDsignal = MACD.ewm(span=9, min_periods=9).mean()
    
    # 根据MACD和其信号线值确定信号
    # 主线值 > 信号线值时，信号为1（买入）
    # 主线值 < 信号线值时，信号为-1（卖出）
    # 主线值等于信号线值时，信号为0（中立）
    signal = np.where(MACD > MACDsignal, 1, np.where(MACD < MACDsignal, -1, 0))
    
    # 返回包含所有历史数据的DataFrame
    return pd.DataFrame({'MACD': signal})

# 计算Stoch_RSI(data,3, 3, 14, 14)，有5个参数，第1个为数据源
def Stoch_RSI_zb(data, smoothK=3, smoothD=3, lengthRSI=14, lengthStoch=14):
    # 计算RSI
    lc = data['close'].shift(1)
    diff = data['close'] - lc
    up = diff.where(diff > 0, 0)
    down = -diff.where(diff < 0, 0)
    ema_up = up.ewm(alpha=1/lengthRSI, adjust=False).mean()
    ema_down = down.ewm(alpha=1/lengthRSI, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - 100 / (1 + rs)
    
    # 计算Stochastic
    stoch = (rsi - rsi.rolling(window=lengthStoch).min()) / (rsi.rolling(window=lengthStoch).max() - rsi.rolling(window=lengthStoch).min())
    k = stoch.rolling(window=smoothK).mean()*100
    d = k.rolling(window=smoothD).mean()
    
    # 判断趋势：1为上升，-1为下降
    trend = (data['close'] > data['close'].rolling(window=10).mean()).astype(int)
    trend[trend == 0] = -1
    
    # 根据Stochastic RSI的K线和D线确定买入、卖出信号
    buy_signal = (trend == -1) & (k < 20) & (d < 20) & (k > d) & (k.shift(1) <= d.shift(1))
    sell_signal = (trend == 1) & (k > 80) & (d > 80) & (k < d) & (k.shift(1) >= d.shift(1))
    
    # 根据信号赋值：1为买入，-1为卖出，0为中立
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame({'Stoch_RSI': signal})

# 计算威廉百分比变动，参数有2，第1是数据源，第二是日期，一般为14，即WPR(data, 14)
def WPR_zb(data, n):
    # 计算WPR（Williams %R）值
    WPR = pd.Series((data['high'].rolling(n).max() - data['close']) / 
                    (data['high'].rolling(n).max() - data['low'].rolling(n).min()) * -100, 
                    name='WPR_' + str(n))
    
    # 设置上下轨
    lower_band = -80
    upper_band = -20
    
    # 根据WPR值和上下轨确定买卖信号
    # 当WPR < 下轨且正在上升时，信号为1（买入）
    # 当WPR > 上轨且正在下跌时，信号为-1（卖出）
    # 否则，信号为0（中立）
    signal = np.where((WPR < lower_band) & (WPR > WPR.shift(1)), 1, 
                      np.where((WPR > upper_band) & (WPR < WPR.shift(1)), -1, 0))

    # 返回包含所有历史数据的DataFrame
    return pd.DataFrame({'WPR_' + str(n):  signal})


# 计算Bull Bear Power牛熊力量(BBP)，参数有2，一个是数据源，另一个是日期，一般为20，但在tradingview取13，即BBP(data, 13)
def BBP_zb(data, n):
    # 计算牛市力量（BullPower）和熊市力量（BearPower）
    bullPower = data['high'] - data['close'].ewm(span=n).mean()
    bearPower = data['low'] - data['close'].ewm(span=n).mean()
    
    # 计算BBP值
    BBP = bullPower + bearPower
    
    # 计算n天的移动平均值
    moving_avg = data['close'].rolling(window=n).mean()

    # 确定股价趋势：1表示上升，-1表示下降，0表示无明显趋势
    trend = np.where(data['close'] > moving_avg, 1, 
                     np.where(data['close'] < moving_avg, -1, 0))

    
    # 根据牛市力量、熊市力量和股价趋势来决定买卖信号
    signal = np.where((trend == 1) & (bearPower < 0) & (bearPower > bearPower.shift(1)), 1, 
                      np.where((trend == -1) & (bullPower > 0) & (bullPower < bullPower.shift(1)), -1, 0))
    
    # 返回BBP值和买卖信号的DataFrame
    return pd.DataFrame({'BBP' : signal})


# 计算Ultimate Oscillator终极震荡指标UO (data,7, 14, 28)，有4个参数，第1个是数据源，其他的是日期
def UO_zb(data, n1, n2, n3):
    # 计算前一天的收盘价和今天的最低价中的最小值
    min_low_or_close = pd.concat([data['low'], data['close'].shift(1)], axis=1).min(axis=1)
    
    # 计算前一天的收盘价和今天的最高价中的最大值
    max_high_or_close = pd.concat([data['high'], data['close'].shift(1)], axis=1).max(axis=1)
    
    # 计算买入压力
    bp = data['close'] - min_low_or_close
    
    # 计算真实范围
    tr_ = max_high_or_close - min_low_or_close
    
    # 计算不同时间范围内的平均买入压力和真实范围的比值
    avg7 = bp.rolling(n1).sum() / tr_.rolling(n1).sum()
    avg14 = bp.rolling(n2).sum() / tr_.rolling(n2).sum()
    avg28 = bp.rolling(n3).sum() / tr_.rolling(n3).sum()
    
    # 计算终极指标UO
    UO = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
    
    # 生成信号
    # UO > 70，生成买入信号1
    # UO < 30，生成卖出信号-1
    # 其他情况，生成中立信号0
    signal = np.where(UO > 70, 1, np.where(UO < 30, -1, 0))
    
    return pd.DataFrame({'UO': signal})


# 计算线性回归
def linear_regression_dfcf_zb(data, years_list):
    df_list = []
    for many_years in years_list:
        percent = round(len(data) / 7 * many_years)
        y = data.iloc[-percent:]['close'].values # 使用 'close' 列
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        expected_value = intercept + slope * len(y)
        residuals = y - (intercept + slope * x)
        std_residuals = np.std(residuals)

        columns = [f"expected_value_{many_years}year", f"std_residuals_{many_years}year"]
        result_data = [expected_value, std_residuals]
        result_df = pd.DataFrame(data=[result_data], columns=columns)

        df_list.append(result_df)
    result = pd.concat(df_list, axis=1)

    return result

def generate_stat_data_zb(stock_code):
    data = get_mt5_data(stock_code)

    ma10 = MA_zb(data, 10)
    ma20 = MA_zb(data, 20)
    ma30 = MA_zb(data, 30)
    ma50 = MA_zb(data, 50)
    ma100 = MA_zb(data, 100)
    ma200 = MA_zb(data, 200)

    ema10 = EMA_zb(data, 10)
    ema20 = EMA_zb(data, 20)
    ema30 = EMA_zb(data, 30)
    ema50 = EMA_zb(data, 50)
    ema100 = EMA_zb(data, 100)
    ema200 = EMA_zb(data, 200)

    ic = ichimoku_cloud_zb(data,9, 26, 52, 26)

    vwma = VWMA_zb(data, 20)

    hm = HullMA_zb(data, 9)

    rsi = RSI_zb(data, 14)

    stok = STOK_zb(data, 14, 3, 3)

    cci = CCI_zb(data, 20)

    adx = ADX_zb(data, 14)

    ao = AO_zb(data)

    mtm = MTM_zb(data)

    macd_level = MACD_Level_zb(data, 12, 26)

    stoch_rsi = Stoch_RSI_zb(data, 3, 3, 14, 14)

    wpr = WPR_zb(data, 14)

    bbp = BBP_zb(data, 13)

    uo = UO_zb(data, 7, 14, 28)

    stat_data = pd.concat([ ma10, ma20, ma30, ma50, ma100, ma200, ema10, ema20, ema30, ema50, ema100, ema200, ic, vwma, hm, rsi, stok, cci, adx, ao, mtm, macd_level, stoch_rsi, wpr, bbp, uo], axis=1)

    stat_data['ti_sum']= stat_data.sum(axis=1)

    mean_value = stat_data.drop('ti_sum', axis=1).mean(axis=1)

    # 判断条件并输出结果
    stat_data['技术方向'] = np.where((mean_value >= -1.0) & (mean_value < -0.5), -2,
         np.where((mean_value >= -0.5) & (mean_value < -0.1), -1,
         np.where((mean_value >= -0.1) & (mean_value <= 0.1), 0,
         np.where((mean_value > 0.1) & (mean_value <= 0.5), 1,
         np.where((mean_value > 0.5) & (mean_value <= 1.0), 2, 0)))))

    
    stat_data = pd.concat([data, stat_data], axis=1).iloc[-1:].reset_index(drop=True)

    lr = linear_regression_dfcf_zb(data, [7,3,1])

    stat_data = pd.concat([stat_data, lr], axis=1)

    stat_data['signal_lr7'] = np.where(stat_data['close'] < (stat_data['expected_value_7year'] - stat_data['std_residuals_7year']), 1, np.where(stat_data['close'] > (stat_data['expected_value_7year'] + stat_data['std_residuals_7year']), -1, 0))
    stat_data['signal_lr3'] = np.where(stat_data['close'] < (stat_data['expected_value_3year'] - stat_data['std_residuals_3year']), 1, np.where(stat_data['close'] > (stat_data['expected_value_3year'] + stat_data['std_residuals_3year']), -1, 0))
    stat_data['signal_lr1'] = np.where(stat_data['close'] < (stat_data['expected_value_1year'] - 2*stat_data['std_residuals_1year']), 1, np.where(stat_data['close'] > (stat_data['expected_value_1year'] + 2*stat_data['std_residuals_1year']), -1, 0))
    stat_data['lr_sum'] = stat_data['signal_lr7'] + stat_data['signal_lr3'] + stat_data['signal_lr1']
    stat_data['线性回归'] = np.where(stat_data['lr_sum'] <= -2, -2,
         np.where(stat_data['lr_sum'] == -1, -1,
         np.where(stat_data['lr_sum'] == 0, 0,
         np.where(stat_data['lr_sum'] == 1, 1,
         np.where(stat_data['lr_sum'] >= 2, 2, 0)))))

    # stat_data['code'] = stock_code
    # stat_data = stat_data.set_index('code')
    return stat_data


# 获取股票的线性回归和技术方向，并与价值估值结合，网格不考虑所述方向
def ex_fund_valuation(db_path, table_name_guojin, table_name_result):
    """
    从指定的表中读取数据，获取估值比率，并将结果保存到新表中。

    参数:
    db_path: str
        数据库文件的路径。
    table_name_guojin: str
        要从中读取数据的表名。
    table_name_result: str
        用于保存结果的新表名。

    返回:
    all_data: DataFrame
        包含估值比率的完整数据集。
    """
    try:
        # 尝试连接到数据库
        conn = sqlite3.connect(db_path)

        # 尝试从参数指定的表中读取数据
        guojin_data = pd.read_sql(f"SELECT * FROM {table_name_guojin}", conn)

    except Exception as e:
        print(f"An error occurred while reading from the database: {str(e)}")
        return pd.DataFrame()  # 返回一个空的DataFrame，表示无法读取数据

    # 初始化一个空的DataFrame来保存所有的结果
    all_data = pd.DataFrame()

    for code in guojin_data['mt5代码']:
        try:
            # 调用自定义函数generate_stat_data_zb获取指定代码的估值比率数据
            ratios = generate_stat_data_zb(code)  # 假设这是一个有效的函数调用

            # 从guojin_data中获取与当前代码匹配的行
            row_data_df = guojin_data[guojin_data['mt5代码'] == code]

            # 合并行数据与估值比率数据
            ratios_new = pd.concat([row_data_df.reset_index(drop=True), ratios.reset_index(drop=True)], axis=1)

            # 将结果添加到all_data中
            all_data = pd.concat([all_data, ratios_new])

        except Exception as e:
            print(f"An error occurred for code {code}: {str(e)}")
            continue

    try:
        # 尝试计算操作列
        all_data["网格"] = all_data["价值估值"] + all_data["线性回归"]
        all_data["基本技术"] = all_data["价值估值"] + all_data["线性回归"] + all_data["技术方向"]

        # 尝试将结果保存到参数指定的新表中
        all_data.to_sql(table_name_result, conn, if_exists='replace', index=False)

    except Exception as e:
        print(f"An error occurred while saving the data: {str(e)}")

    finally:
        # 关闭数据库连接
        conn.close()

    return all_data

# 获取指数外汇总评价数据，并保存到数据库
def ex_fund_forex_valuation(db_path, table_name_guojin, table_name_result):
    """
    从指定的表中读取数据，获取估值比率，并将结果保存到新表中。

    参数:
    db_path: str
        数据库文件的路径。
    table_name_guojin: str
        要从中读取数据的表名。
    table_name_result: str
        用于保存结果的新表名。

    返回:
    all_data: DataFrame
        包含估值比率的完整数据集。
    """
    try:
        # 尝试连接到数据库
        conn = sqlite3.connect(db_path)

        # 尝试从参数指定的表中读取数据
        guojin_data = pd.read_sql(f"SELECT * FROM {table_name_guojin}", conn)

    except Exception as e:
        print(f"An error occurred while reading from the database: {str(e)}")
        return pd.DataFrame()  # 返回一个空的DataFrame，表示无法读取数据

    # 初始化一个空的DataFrame来保存所有的结果
    all_data = pd.DataFrame()

    for code in guojin_data['mt5代码']:
        try:
            # 调用自定义函数generate_stat_data_zb获取指定代码的估值比率数据
            ratios = generate_stat_data_zb(code)  # 假设这是一个有效的函数调用

            # 从guojin_data中获取与当前代码匹配的行
            row_data_df = guojin_data[guojin_data['mt5代码'] == code]

            # 合并行数据与估值比率数据
            ratios_new = pd.concat([row_data_df.reset_index(drop=True), ratios.reset_index(drop=True)], axis=1)

            # 将结果添加到all_data中
            all_data = pd.concat([all_data, ratios_new])

        except Exception as e:
            print(f"An error occurred for code {code}: {str(e)}")
            continue

    try:
        # 尝试计算操作列
        all_data["网格"] = all_data["线性回归"] + all_data["技术方向"]

        # 尝试将结果保存到参数指定的新表中
        all_data.to_sql(table_name_result, conn, if_exists='replace', index=False)

    except Exception as e:
        print(f"An error occurred while saving the data: {str(e)}")

    finally:
        # 关闭数据库连接
        conn.close()

    return all_data


# 参数：数据库路径、EA_id（平仓策略代码）、magic（持仓策略代码）
def calculate_totals(db_path, ea_id, magic):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 执行SQL查询，筛选出EA_id为指定值的所有数据
    cursor.execute("""
    SELECT "交易佣金", "隔夜利息", "利润"
    FROM forex_trades
    WHERE "EA_id" = ?
    """, (ea_id,))

    # 获取查询结果
    results_forex_trades = cursor.fetchall()

    # 初始化总和变量
    total_commission = 0.0
    total_swap_forex_trades = 0.0
    total_profit_forex_trades = 0.0

    # 计算总和
    for row in results_forex_trades:
        total_commission += row[0]
        total_swap_forex_trades += row[1]
        total_profit_forex_trades += row[2]

    # 执行SQL查询，筛选出magic为指定值的所有数据
    cursor.execute("""
    SELECT "swap", "profit"
    FROM position
    WHERE "magic" = ?
    """, (magic,))

    # 获取查询结果
    results_position = cursor.fetchall()

    # 初始化总和变量
    total_swap_position = 0.0
    total_profit_position = 0.0

    # 计算总和
    for row in results_position:
        total_swap_position += row[0]
        total_profit_position += row[1]

    # 打印结果
    print(f"交易佣金总和: {total_commission}")
    print(f"forex_trades表中隔夜利息总和: {total_swap_forex_trades}")
    print(f"forex_trades表中利润总和: {total_profit_forex_trades}")
    print(f"position表中隔夜利息总和: {total_swap_position}")
    print(f"position表中利润总和: {total_profit_position}")
    # 打印所有列的总和
    print(f"总和: {total_commission + total_swap_forex_trades + total_profit_forex_trades + total_swap_position + total_profit_position}")

    # 关闭数据库连接
    conn.close()






if __name__ == '__main__':
    # db_path = r'D:\wenjian\python\smart\data\mt5.db'

    if not mt5.initialize(path=r"D:\jiaoyi\IC-MT5-Demo\terminal64.exe", login=51455171, password="LiEbcs6r", server="ICMarketsSC-Demo"):
        print("initialize()失败，错误代码=", mt5.last_error())
    else:
        print("MT5 initialized")
    
    # 获取数据
    stock_code = 'AAPL.NAS'
    df = generate_stat_data_zb(stock_code)

    # 数据库文件路径
    db_path = r"D:\wenjian\python\smart\data\backtest_data.db"

    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_path)

    # 将 DataFrame 写入数据库表
    table_name = '沪深300随机50总评价'
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # 关闭数据库连接
    conn.close()

    print(f"数据已成功导出到数据库表 {table_name} 中。")

    # 从问财网获取纳斯达克100指数成分股数据，参数查询语句，每年的1月1日运行一次，因为纳斯达克100指数成分股每年年底会有变化
    # wencai_conditional_query("纳斯达克100指数成分股")

    # 测试，被save_exchange_rates_to_db函数调用从东方财富网的API获取指定股票代码的汇率信息，并提取汇率数据
    # get_exchange_rate('133.USDCNH')

    # 调用函数保存汇率数据到数据库，常用
    # save_exchange_rates_to_db(db_path, "汇率换算")

    # 初始化MetaTrader 5连接
    # if not mt5.initialize(path = r"D:\jiaoyi\IC-MT5\terminal64.exe", login = 511231, password = "Li",server = "ICMarketsSC-Demo"):
    #     print("initialize()失败，错误代码=", mt5.last_error())
    # else:
    #     print("MT5 initialized")

    # 获取美股中国企业和纳指100的价值大师估值，每周五下等运行，每周运行1次
    # save_data(get_all_valuation_ratios_db, db_path, "美股中国企业", "美股中国企业价值")
    # save_data(get_all_valuation_ratios_db, db_path, "nasdaq_100", "纳指100价值")

    # 测试获取基本技术数据
    # print(generate_stat_data("USDJPY"))
    
    # ex_fund_valuation(db_path, "美股中国企业价值", "美股中国企业总评价")
    # ex_fund_valuation(db_path, "纳指100价值", "纳指100总评价")
    # ex_fund_forex_valuation(db_path, "指数外汇", "指数外汇总评价")

    
    
    # 完成所有数据获取后断开MT5连接
    # mt5.shutdown()



