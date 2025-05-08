
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import numpy as np
import os
import datetime
from scipy import stats
import math
from datetime import datetime, timedelta
from scipy import optimize
from xtquant import xtdata
import pywencai


# 数据处理文件，主要为技术指标的计算


# 定义函数，根据element_pj的值返回对应的得分
def get_score(element_pj):
    try:
        if element_pj == "股价被严重高估":
            score = -2
        elif element_pj == "股价被高估":
            score = -1
        elif element_pj == "股价被低估":
            score = 1
        elif element_pj == "股价被严重低估":
            score = 2
        else:
            score = 0
        return score
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0  # 或者您可以返回一个适当的默认值或错误标志


# 通过代码，获取价值大师的价值线。参数是股票代码，例如get_valuation_ratios("AAPL.NAS")
def get_valuation_ratios(code):
    try:
        url = f'https://www.gurufocus.cn/stock/{code}/term/gf_value'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 获取大师价值线
        element_jzx = soup.select_one('#term-page-title').text.strip()
        element_jzx = re.findall(r"\d+\.\d+", element_jzx)[0]

        # 获取大师价值估值评价
        element_pj = soup.select_one('#q-app > div > div > main > div:nth-child(3) > div > div.column.term-container.col > div.content-section.q-pa-md.q-mt-sm.q-card.q-card--bordered.q-card--flat.no-shadow > div > div:nth-child(9) > span').text.strip()

        # 调用函数get_score，对大师价值估值评价进行得分估值
        Fraction_pj = get_score(element_pj)

        # 获取名称
        element_mc = soup.select_one('html body div div main div:nth-child(1) div:nth-child(2) div:nth-child(1) div:nth-child(1) div:nth-child(2) div:nth-child(1) h1 span:nth-child(1)').text.strip()

        # 获取现价
        element_xj = soup.select_one('html body div div main div:nth-child(1) div:nth-child(2) div:nth-child(1) div:nth-child(1) div:nth-child(2) div:nth-child(2)')
        element_xj.span.decompose()  # 去掉span标签
        element_xj = re.findall(r'\d+\.\d+', element_xj.text.strip())[0]  # 只保留数字和小数点

        # 获取市净率、市盈率、股息率和总市值的基本数据
        element_data = soup.select_one('#q-app > div > div > main > div.q-pt-sm.bg-white.shadow-1 > div.row.page-width-1440.paywall-trigger.login-trigger > div.col-12.col-md.q-mb-xs.q-mx-xs.q-card.q-card--flat.no-shadow.q-px-md > div:nth-child(2) > div').text.strip()

        # 创建一个pandas DataFrame来保存比率
        ratios = pd.DataFrame({
            '价值代码': [code],
            '价值名称': [element_mc],
            '价值现价': [element_xj],
            '大师价值': [element_jzx],
            '评价': [element_pj],
            '价值估值': [Fraction_pj],
            '基本数据': [element_data],
        })

        # 将“代码”列设置为索引
        return ratios
    except Exception as e:
        print(f"发生错误: {e}")
        return None  # 或者您可以返回一个适当的默认值或错误标志


# 通过读取excel中的列“代码”，从而获取价值大师价格。参数分别是：数据库路径、表名。例如get_all_valuation_ratios_db(db_path, "指数价值")
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



# 将数据保存为sql。参数分别是：自定义函数、数据库路径、原始表名、新表名。例如save_data(get_all_valuation_ratios_db, db_path, "指数价值", "指数价值_sql")
def save_data(function, db_path, original_table_name, new_table_name):
    """
    执行 function 以获取数据，并将该数据与 original_table_name 表的内容合并，
    然后将合并后的数据保存到 new_table_name 表中。

    参数:
    function: callable
        获取数据的自定义函数。
    db_path: str
        数据库文件的路径。
    original_table_name: str
        原始数据表的名称。
    new_table_name: str
        用于保存合并后数据的新表名称。
    """
    # 创建数据库连接
    conn = sqlite3.connect(db_path)
    
    try:
        # 获取原始表中的数据
        original_data = pd.read_sql_query(f"SELECT * FROM `{original_table_name}`", conn)
        
        # 调用自定义函数获取新的数据
        new_data = function(db_path, original_table_name)
        
        # 如果新数据不为空，合并原始数据和新数据
        if not new_data.empty:
            merged_data = pd.merge(original_data, new_data, on='价值代码', how='left')
        else:
            merged_data = original_data
        
        # 将合并后的数据保存到新的表中
        merged_data.to_sql(new_table_name, conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭数据库连接
        conn.close()


# 获取data数据中的第几行数据。参数分别是：数据源、行数。例如get_row(data, 0)
def get_row(data, index):
    return data.iloc[[index]].reset_index(drop=True)




# 获取data数据中的第几行数据。参数分别是：数据库路径。例如get_stock_list_from_db(db_path)
def get_stock_list_from_db(db_path):
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


# 获取简单移动平均线，参数有2个，一个是数据源，一个是日期。例如MA(data, 20)
def MA(data, n):
    MA = pd.Series(data['close'].rolling(n).mean(), name='MA_' + str(n))
    return MA.dropna()


# 获取指数移动平均线，参数有2个，一个是数据源，一个是日期。例如EMA(data, 20)
def EMA(data, n):
    EMA = pd.Series(data['close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA.dropna()



# 获取一目均衡表基准线 (data, conversion_periods, base_periods, lagging_span2_periods, displacement)
# 参数有5个，第一个是数据源，其他4个分别是一目均衡表基准线 (9, 26, 52, 26)，即ichimoku_cloud(data,9, 26, 52, 26)
def ichimoku_cloud(data, conversion_periods, base_periods, lagging_span2_periods, displacement):
    def donchian(length):
        return (data['high'].rolling(length).max() + data['low'].rolling(length).min()) / 2
    
    conversion_line = donchian(conversion_periods)
    base_line = donchian(base_periods)
    lead_line1 = (conversion_line + base_line) / 2
    lead_line2 = donchian(lagging_span2_periods)
    
    lagging_span = data['close'].shift(-displacement + 1).shift(25).ffill()
    leading_span_a = lead_line1.shift(displacement - 1)
    leading_span_b = lead_line2.shift(displacement - 1)
    
    ichimoku_data = pd.concat([conversion_line, base_line, lagging_span, lead_line1, lead_line2, leading_span_a, leading_span_b], axis=1)
    ichimoku_data.columns = ['Conversion Line', 'Base Line', 'Lagging Span', 'lead_line1', 'lead_line2', 'Leading Span A', 'Leading Span B']
    
    return ichimoku_data.dropna()



# 成交量加权移动平均线 VWMA (data, 20)，参数有2个，1个是数据源，另一个是日期，通过为20
def VWMA(data, n):
    VWMA = pd.Series((data['close'] * data['volume']).rolling(n).sum() / data['volume'].rolling(n).sum(), name='VWMA_' + str(n))
    return VWMA.dropna()


# 计算Hull MA船体移动平均线 Hull MA (data,9)，参数有2，一个是数据源，另一个是日期，一般为9。例如HullMA(data, 9)
def HullMA(data, n=9):
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    source = data['close']
    wma1 = wma(source, n // 2) * 2
    wma2 = wma(source, n)
    hullma = wma(wma1 - wma2, int(math.floor(math.sqrt(n))))
    # 指定返回的Series对象的名称
    hullma.name = 'HullMA_' + str(n)
    return hullma.dropna()

# 计算RSI指标，参数有2，一个为数据源，另一个为日期，一般为14，即RSI(data, 14)
def RSI(data, n):
    lc = data['close'].shift(1)
    diff = data['close'] - lc
    up = diff.where(diff > 0, 0)
    down = -diff.where(diff < 0, 0)
    ema_up = up.ewm(alpha=1/n, adjust=False).mean()
    ema_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - 100 / (1 + rs)
    return pd.Series(rsi, index=data.index, name='RSI_' + str(n)).dropna()

# 计算Stochastic，k是主线，d_signal是信号线，参数有4，一个是数据源，另外三个为日期，一般为STOK(data, 14, 3, 3)
def STOK(data, n, m, t):
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
    
    # 创建一个新的DataFrame来存储结果
    result = pd.DataFrame({
        'Stochastic_%K': k,
        'Stochastic_%D': d,
        'Stochastic_%D_signal': d_signal
    }, index=data.index)  # 使用原始数据的索引
    
    return result.dropna()


# 计算CCI指标，参数有2，一个是数据源，另一个是日期，一般为20，即CCI(data, 20)
def CCI(data, n):
    TP = (data['high'] + data['low'] + data['close']) / 3
    MA = TP.rolling(n).mean()
    MD = TP.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean())
    CCI = (TP - MA) / (0.015 * MD)
    return pd.Series(CCI, index=data.index, name='CCI_' + str(n)).dropna()


# 平均趋向指数ADX(14)，参数有2，一个是数据源，另一个是日期，一般为14，即ADX(data,14)
def ADX(data, n):
    up = data['high'] - data['high'].shift(1)
    down = data['low'].shift(1) - data['low']
    plusDM = pd.Series(np.where((up > down) & (up > 0), up, 0))
    minusDM = pd.Series(np.where((down > up) & (down > 0), down, 0))
    truerange = np.maximum(data['high'] - data['low'], np.maximum(np.abs(data['high'] - data['close'].shift()), np.abs(data['low'] - data['close'].shift())))
    plus = 100 * plusDM.ewm(alpha=1/n, min_periods=n).mean() / truerange.ewm(alpha=1/n, min_periods=n).mean()
    minus = 100 * minusDM.ewm(alpha=1/n, min_periods=n).mean() / truerange.ewm(alpha=1/n, min_periods=n).mean()
    sum = plus + minus
    adx = 100 * (np.abs(plus - minus) / np.where(sum == 0, 1, sum)).ewm(alpha=1/n, min_periods=n).mean()
    return pd.Series(adx, index=data.index, name='ADX').dropna()


# 计算动量震荡指标(AO)，参数只有一个，即数据源
def AO(data):
    AO = (data['high'].rolling(5).mean() + data['low'].rolling(5).mean()) / 2 - (data['high'].rolling(34).mean() + data['low'].rolling(34).mean()) / 2
    # 指定返回的Series对象的索引为原始data的索引，并命名为'AO'
    return pd.Series(AO, index=data.index, name='AO').dropna()


# 计算动量指标(10)，参数只有一个，即数据源
def MTM(data):
    MTM = data['close'] - data['close'].shift(10)
    # 指定返回的Series对象的索引为原始data的索引
    return pd.Series(MTM, index=data.index, name='MTM').dropna()

# 计算MACD Lvel指标，参数有3个，第一个是数据源，其余两个为日期，一般取12和26，即MACD_Level(data, 12,26)
def MACD_Level(data, n_fast, n_slow):
    EMAfast = data['close'].ewm(span=n_fast, min_periods=n_slow).mean()
    EMAslow = data['close'].ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = EMAfast - EMAslow
    MACDsignal = MACD.ewm(span=9, min_periods=9).mean()
    MACDhist = MACD - MACDsignal
    
    # 创建一个新的DataFrame来存储结果
    result = pd.DataFrame({
        'MACD': MACD,
        'MACDsignal': MACDsignal,
        'MACDhist': MACDhist
    }, index=data.index)  # 使用原始数据的索引
    
    return result.dropna()

# 计算Stoch_RSI(data,3, 3, 14, 14)，有5个参数，第1个为数据源
def Stoch_RSI(data, smoothK, smoothD, lengthRSI, lengthStoch):
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
    k = stoch.rolling(window=smoothK).mean() * 100
    d = k.rolling(window=smoothD).mean()
    
    # 创建一个新的DataFrame来存储结果
    result = pd.DataFrame({
        'Stoch_RSI_K': k,
        'Stoch_RSI_D': d
    }, index=data.index)  # 使用原始数据的索引
    
    return result.dropna()

# 计算威廉百分比变动，参数有2，第1是数据源，第二是日期，一般为14，即WPR(data, 14)
def WPR(data, n):
    WPR = pd.Series((data['high'].rolling(n).max() - data['close']) / (data['high'].rolling(n).max() - data['low'].rolling(n).min()) * -100, name='WPR_' + str(n))
    return WPR.dropna()


# 计算Bull Bear Power牛熊力量(BBP)，参数有2，一个是数据源，另一个是日期，一般为20，但在tradingview取13，即BBP(data, 13)
def BBP(data, n):
    bullPower = data['high'] - data['close'].ewm(span=n).mean()
    bearPower = data['low'] - data['close'].ewm(span=n).mean()
    BBP = bullPower + bearPower
    return pd.Series(BBP, index=data.index, name='BBP').dropna()


# 计算Ultimate Oscillator终极震荡指标UO (data,7, 14, 28)，有4个参数，第1个是数据源，其他的是日期
def UO(data, n1, n2, n3):
    min_low_or_close = pd.concat([data['low'], data['close'].shift(1)], axis=1).min(axis=1)
    max_high_or_close = pd.concat([data['high'], data['close'].shift(1)], axis=1).max(axis=1)
    bp = data['close'] - min_low_or_close
    tr_ = max_high_or_close - min_low_or_close
    avg7 = bp.rolling(n1).sum() / tr_.rolling(n1).sum()
    avg14 = bp.rolling(n2).sum() / tr_.rolling(n2).sum()
    avg28 = bp.rolling(n3).sum() / tr_.rolling(n3).sum()
    UO = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
    return pd.Series(UO, index=data.index, name='UO').dropna()


# 计算线性回归，参数分别是：数据源，日期列表，一般为[5, 10, 20, 30, 60]，即linear_regression_dfcf(data, [5, 10, 20, 30, 60])
def linear_regression_dfcf(data, days_list):
    df_list = []
    for many_days in days_list:
        # 使用 rolling 函数创建滚动窗口
        expected_values = (
            data['close']
            .rolling(window=many_days)
            .apply(lambda y: stats.linregress(np.arange(len(y)), y)[1] + stats.linregress(np.arange(len(y)), y)[0] * (len(y) - 1), raw=True)
        )
        # 计算滚动窗口的残差标准差
        std_residuals = (
            data['close']
            .rolling(window=many_days)
            .apply(lambda y: np.std(y - (stats.linregress(np.arange(len(y)), y)[1] + stats.linregress(np.arange(len(y)), y)[0] * np.arange(len(y)))), raw=True)
        )

        # 创建新的 DataFrame
        temp_df = pd.DataFrame({
            f"expected_value_{many_days}day": expected_values,
            f"std_residuals_{many_days}day": std_residuals
        })
        df_list.append(temp_df)

    result = pd.concat(df_list, axis=1)

    return result


# 通过东方财富api获取K线数据，参数包括股票代码，天数，复权类型，K线类型
# `klt`：K 线周期，可选值包括 5（5 分钟 K 线）、15（15 分钟 K 线）、30（30 分钟 K 线）、60（60 分钟 K 线）、101（日 K 线）、102（周 K 线）、103（月 K 线）等。
# `fqt`：复权类型，可选值包括 0（不复权）、1（前复权）、2（后复权）。
def json_to_dfcf(stock_code, days=365, fqt=1, klt=101):
    if stock_code.endswith("SH"):
        stock_code = "1." + stock_code[:-3]
    else:
        stock_code = "0." + stock_code[:-3]
    try:
        today = datetime.now().date()
        start_time = (today - timedelta(days=days)).strftime("%Y%m%d")
        end_date = today.strftime('%Y%m%d')
        url = f'http://push2his.eastmoney.com/api/qt/stock/kline/get?&secid={stock_code}&fields1=f1,f3&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt={klt}&&fqt={fqt}&beg={start_time}&end={end_date}'
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        data = [x.split(',') for x in data['data']['klines']]
        column_names = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']
        df = pd.DataFrame(data, columns=column_names)

        # 将数值列转换为 float 类型，处理 NaN
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'percentage change', 'change amount', 'turnover rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.ffill(inplace=True)  # 向前填充 NaN

        return df
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except Exception as e:
        print(f"发生异常: {e}")
        return None
    


# 从东财获取8年数据，计算各种指标指标，参数为股票代码。例如generate_stat_data("AAPL.NAS")
def generate_stat_data(stock_code):
    data = json_to_dfcf(stock_code, days=365*8, fqt=1, klt=101)

    ma10 = MA(data, 10)
    ma20 = MA(data, 20)
    ma30 = MA(data, 30)
    ma50 = MA(data, 50)
    ma100 = MA(data, 100)
    ma200 = MA(data, 200)

    ema10 = EMA(data, 10)
    ema20 = EMA(data, 20)
    ema30 = EMA(data, 30)
    ema50 = EMA(data, 50)
    ema100 = EMA(data, 100)
    ema200 = EMA(data, 200)

    ic = ichimoku_cloud(data,9, 26, 52, 26)

    vwma = VWMA(data, 20)

    hm = HullMA(data, 9)

    rsi = RSI(data, 14)

    wpr = WPR(data, 14)

    cci = CCI(data, 20)

    adx = ADX(data, 14)

    stok = STOK(data, 14, 3, 3)

    ao = AO(data)

    mtm = MTM(data)

    madc_level = MACD_Level(data, 12, 26)

    stoch_rsi = Stoch_RSI(data, 3, 3, 14, 14)

    bbp = BBP(data, 13)

    uo = UO(data, 7, 14, 28)

    lr = linear_regression_dfcf(data, [10, 20, 60, 120])

    stat_data = pd.concat([data, ma10, ma20, ma30, ma50, ma100, ma200, ema10, ema20, ema30, ema50, ema100, ema200, ic, vwma, hm, rsi,  cci, adx, wpr, stok, ao, mtm, madc_level, stoch_rsi, bbp, uo, lr], axis=1) 

    # 获取众120个数据之后的数据，因为线性回归计算的最长的一个日期是120交易日
    stat_data_after_120 = stat_data.iloc[119:]
    return stat_data_after_120



# 计算xirr年化收益率。参数分别是：现金流、日期。例如calculate_xirr(cash_flows, dates)
def calculate_xirr(cash_flows, dates):
    dates = [pd.to_datetime(date) for date in dates]
    if len(cash_flows) == 0 or len(cash_flows) != len(dates):
        return -1
    years = np.array([(d - dates[0]).days / 365.0 for d in dates])
    
    def f(r):
        with np.errstate(all='ignore'):
            try:
                return np.sum(cash_flows / (1 + r) ** years)
            except FloatingPointError:
                return np.inf

    try:
        result = optimize.newton(f, x0=0.1, tol=1e-6, maxiter=1000)
        return result if -1 < result < 1e10 else -1
    except (RuntimeError, OverflowError, FloatingPointError):
        return -1

# 计算年化收益率、现金流之和和净现值。参数分别是：表名。例如calculate_annual_return("外汇")
def calculate_annual_return(table_name):
    db_path = r"D:\wenjian\python\smart\data\guojin_account.db"
    conn = sqlite3.connect(db_path)

    # 使用增加了"_hedge"后缀的表名
    table_name_hedge = f"{table_name}_hedge"
    query = f"SELECT 成交时间, 成交金额, 买卖 FROM {table_name_hedge}"
    data = pd.read_sql_query(query, conn)

    conn.close()

    # 将成交时间转换为日期格式
    data['成交时间'] = pd.to_datetime(data['成交时间'])

    # 计算现金流
    data['现金流'] = -data['成交金额'] * data['买卖']

    # 计算年化收益率
    annual_return = calculate_xirr(data['现金流'].values, data['成交时间'].values)

    # 计算“现金流”的和
    cash_flows_sum = data['现金流'].sum()

    # 计算净现值
    discount_rate = 0.1  # 折现率设为5%
    periods = (data['成交时间'] - data['成交时间'].min()).dt.days / 365.25  # 计算每笔现金流的时间（以年为单位）
    npv = sum(data['现金流'] / (1 + discount_rate) ** periods)

    # 创建一个新的DataFrame来返回结果
    result = pd.DataFrame({
        '年化收益率': [annual_return],
        '现金流之和': [cash_flows_sum],
        '净现值': [npv]
    })

    return result

# 股票代码增加市场信息。参数是股票代码，例如get_processed_code("000001.SZ")
def get_processed_code(stock_code):
    stock_code = str(stock_code)

    if stock_code.startswith('6'):  # 沪市股票
        market = 'SH'
    elif stock_code.startswith('0') or stock_code.startswith('3'):  # 深市股票
        market = 'SZ'
    elif stock_code.startswith('11') or stock_code.startswith('113'):  # 沪市可转债
        market = 'SH'
    elif stock_code.startswith('12'):  # 深市可转债
        market = 'SZ'
    elif stock_code.startswith('50') or stock_code.startswith('51') or stock_code.startswith('58'):  # 沪市场内基金
        market = 'SH'
    elif stock_code.startswith('15') or stock_code.startswith('16') or stock_code.startswith('18'):  # 深市场内基金
        market = 'SZ'
    else:
        return "轮动定时"

    return f"{stock_code}.{market}"

# 通达信代码转换。参数是股票代码，例如tongda_code_convert("000001.SZ")
def tongda_code_convert(stock_code):
    stock_code = str(stock_code)
    if stock_code.startswith('0'):
        return stock_code[1:] + '.SZ'
    elif stock_code.startswith('1'):
        return stock_code[1:] + '.SH'
    else:
        return stock_code  # 如果不满足条件，保持原样


# 清洗execute_general_trade表。参数是数据库连接，例如clean_execute_general_trade(conn)
def clean_execute_general_trade(conn):
    """
    清洗execute_general_trade表
    """
    cursor = conn.cursor()
    # 从receive_condition获取(策略名称, 证券代码)的集合
    cursor.execute("SELECT 策略名称, 证券代码 FROM receive_condition")
    receive_keys = set(cursor.fetchall())
    
    # 获取execute_general_trade中非'问财轮动-注询问'的所有行
    cursor.execute("SELECT rowid, 策略名称, 证券代码 FROM execute_general_trade WHERE 策略名称 != '问财轮动-注询问'")
    rows = cursor.fetchall()
    for rowid, strategy, code in rows:
        if (strategy, code) not in receive_keys:
            cursor.execute("DELETE FROM execute_general_trade WHERE rowid=?", (rowid,))
    conn.commit()


# 插入订单到place_general_order表。参数分别是：数据库游标、数据库连接、证券代码、委托价格、委托数量、买卖、策略名称、委托备注、日期时间。例如insert_order(cursor, conn, code, price, quantity, buy_sell, strategy, remark, datetime_str)
def insert_order(cursor, conn, code, price, quantity, buy_sell, strategy, remark, datetime_str):
    """
    插入订单到place_general_order表
    """
    cursor.execute("""
        INSERT INTO place_general_order (证券代码, 委托价格, 委托数量, 买卖, 策略名称, 委托备注, 日期时间)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (code, price, quantity, buy_sell, strategy, remark, datetime_str))
    conn.commit()


# 删除receive_condition表中的行。参数分别是：数据库游标、数据库连接、行id。例如delete_receive_condition_row(cursor, conn, rowid)
def delete_receive_condition_row(cursor, conn, rowid):
    """
    删除receive_condition表中的行
    """
    cursor.execute("DELETE FROM receive_condition WHERE rowid=?", (rowid,))
    conn.commit()


# 删除execute_general_trade表中的行。参数分别是：数据库游标、数据库连接、策略名称、证券代码。例如delete_execute_general_trade_row(cursor, conn, strategy, code)
def delete_execute_general_trade_row(cursor, conn, strategy, code):
    """
    删除execute_general_trade表中的行
    """
    cursor.execute("DELETE FROM execute_general_trade WHERE 策略名称=? AND 证券代码=?", (strategy, code))
    conn.commit()


# 处理价差网格策略。参数分别是：数据库游标、数据库连接、tick数据、证券代码、委托价格、委托数量、买卖、策略名称、委托备注、日期时间、行id。例如process_price_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid)
def process_price_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid):
    """
    处理价差网格策略
    """
    lastPrice = tick.get('lastPrice')
    if lastPrice is None:
        return
    委托备注_value = float(remark)
    委托数量_value = quantity
    # 查询execute_general_trade
    cursor.execute("""
        SELECT 成交均价, 成交数量, 买卖, 委托备注 FROM execute_general_trade
        WHERE 策略名称=? AND 证券代码=?
        ORDER BY rowid ASC
    """, (strategy, code))
    trade_rows = cursor.fetchall()
    if not trade_rows:
        if (price - 委托备注_value) >= lastPrice:
            new_price = price - 委托备注_value
            insert_order(cursor, conn, code, new_price, 委托数量_value, 1, strategy, remark, datetime_str)
    else:
        total_position = sum(row[1] * row[2] for row in trade_rows)
        last_trade = trade_rows[-1]
        last成交均价 = last_trade[0]
        委托备注_value = float(last_trade[3])
        if (last成交均价 - 委托备注_value) >= lastPrice:
            new_price = last成交均价 - 委托备注_value
            insert_order(cursor, conn, code, new_price, 委托数量_value, 1, strategy, remark, datetime_str)
        elif (last成交均价 + 委托备注_value) <= lastPrice:
            new_price = last成交均价 + 委托备注_value
            if total_position >= 委托数量_value:
                insert_order(cursor, conn, code, new_price, 委托数量_value, -1, strategy, remark, datetime_str)
            elif total_position > 0:
                insert_order(cursor, conn, code, new_price, total_position, -1, strategy, remark, datetime_str)
            else:
                pass  # 不做任何操作
    # 不删除receive_condition表中的行

# 处理振幅网格策略。参数分别是：数据库游标、数据库连接、tick数据、证券代码、委托价格、委托数量、买卖、策略名称、委托备注、日期时间、行id。例如process_amplitude_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid)
def process_amplitude_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid):
    """
    处理振幅网格策略
    """
    try:
        lastPrice = tick.get('lastPrice')
        if lastPrice is None:
            return
        
        委托备注_value = float(remark)
        委托数量_value = quantity
        
        cursor.execute("""
            SELECT 成交均价, 成交数量, 买卖, 委托备注 FROM execute_general_trade
            WHERE 策略名称=? AND 证券代码=?
            ORDER BY rowid ASC
        """, (strategy, code))
        trade_rows = cursor.fetchall()
        
        if not trade_rows:
            new_buy_price = price - 委托备注_value * price / 100
            if new_buy_price >= lastPrice:
                insert_order(cursor, conn, code, new_buy_price, 委托数量_value, 1, strategy, remark, datetime_str)
        else:
            total_position = sum(row[1] * row[2] for row in trade_rows)
            last_trade = trade_rows[-1]
            last成交均价 = last_trade[0]
            委托备注_value = float(last_trade[3])
            buy_price = last成交均价 - 委托备注_value * last成交均价 / 100
            sell_price = last成交均价 + 委托备注_value * last成交均价 / 100
            
            if buy_price >= lastPrice:
                insert_order(cursor, conn, code, buy_price, 委托数量_value, 1, strategy, remark, datetime_str)
            elif sell_price <= lastPrice:
                if total_position >= 委托数量_value:
                    insert_order(cursor, conn, code, sell_price, 委托数量_value, -1, strategy, remark, datetime_str)
                elif total_position > 0:
                    insert_order(cursor, conn, code, sell_price, total_position, -1, strategy, remark, datetime_str)
                else:
                    pass  # 不做任何操作
    except Exception as e:
        # 捕获所有异常并打印错误信息
        print(f"发生异常: {e}")
        # 回滚事务以确保数据一致性
        conn.rollback()
        return None  # 返回 None 或其他适当的值以表示失败

    # 不删除receive_condition表中的行


# 处理监控策略。参数分别是：行数据、数据库游标、数据库连接。例如process_immediate_rows(rows, cursor, conn)
def process_immediate_rows(rows, cursor, conn):
    """
    处理即时行数据
    """
    if not rows:
        return
    securities = list(set(row[1] for row in rows))  # row[1] 是 '证券代码'
    tick_data = xtdata.get_full_tick(securities)
    current_time = datetime.now()
    for row in rows:
        rowid, code, price, quantity, buy_sell, strategy, remark, datetime_str = row
        # 处理'日期时间'
        if datetime_str:
            row_time = datetime.strptime(datetime_str, "%Y%m%d %H:%M")
            if current_time > row_time:
                cursor.execute("DELETE FROM receive_condition WHERE rowid=?", (rowid,))
                conn.commit()
                # 删除对应的execute_general_trade行
                cursor.execute("DELETE FROM execute_general_trade WHERE 策略名称=? AND 证券代码=?", (strategy, code))
                conn.commit()
                continue
        # 获取tick数据
        tick = tick_data.get(code, {})
        if not tick:
            continue  # 如果没有tick数据，跳过
        lastPrice = tick.get('lastPrice')
        high = tick.get('high')
        low = tick.get('low')
        lastClose = tick.get('lastClose')
        volume = tick.get('volume')
        bidVol1 = tick.get('bidVol')[0]  # 买一量
        askVol1 = tick.get('askVol')[0]  # 卖一量
        if strategy == '即时委托':
            insert_order(cursor, conn, code, price, quantity, buy_sell, strategy, remark, datetime_str)
            cursor.execute("DELETE FROM receive_condition WHERE rowid=?", (rowid,))
            conn.commit()
            # 删除对应的execute_general_trade行
            cursor.execute("DELETE FROM execute_general_trade WHERE 策略名称=? AND 证券代码=?", (strategy, code))
            conn.commit()
        elif strategy == '上破价':
            if lastPrice >= price:
                insert_order(cursor, conn, code, price, quantity, buy_sell, strategy, remark, datetime_str)
                delete_receive_condition_row(cursor, conn, rowid)
                delete_execute_general_trade_row(cursor, conn, strategy, code)
        elif strategy == '下破价':
            if lastPrice <= price:
                insert_order(cursor, conn, code, price, quantity, buy_sell, strategy, remark, datetime_str)
                delete_receive_condition_row(cursor, conn, rowid)
                delete_execute_general_trade_row(cursor, conn, strategy, code)
        elif strategy == '封单不足-注数量':
            action, amount = remark.split(":")
            amount = float(amount)
            if (action == 'buy' and bidVol1 <= amount) or (action == 'sell' and askVol1 <= amount):
                insert_order(cursor, conn, code, price, quantity, buy_sell, strategy, remark, datetime_str)
                delete_receive_condition_row(cursor, conn, rowid)
                delete_execute_general_trade_row(cursor, conn, strategy, code)

        elif strategy == '回落卖-价涨幅-注回撤':
            try:
                if lastClose != 0:
                    price_change_ratio = (high - lastClose) / lastClose
                    retreat_ratio = (high - lastPrice) / high
                    if (price_change_ratio >= price / 100) and (retreat_ratio >= float(remark) / 100):
                        insert_order(cursor, conn, code, 0, quantity, buy_sell, strategy, '最优五档', datetime_str)
                        delete_receive_condition_row(cursor, conn, rowid)
                        delete_execute_general_trade_row(cursor, conn, strategy, code)
            except ZeroDivisionError:
                continue
        elif strategy == '反弹买-价跌幅-注反弹':
            try:
                if lastClose != 0:
                    price_drop_ratio = (lastClose - low) / lastClose
                    rebound_ratio = (lastPrice - low) / low
                    if (price_drop_ratio >= price / 100) and (rebound_ratio >= float(remark) / 100):
                        insert_order(cursor, conn, code, 0, quantity, buy_sell, strategy, '最优五档', datetime_str)
                        delete_receive_condition_row(cursor, conn, rowid)
                        delete_execute_general_trade_row(cursor, conn, strategy, code)
            except ZeroDivisionError:
                continue
        elif strategy.startswith('价差网格-注价差'):
            process_price_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid)
        elif strategy.startswith('振幅网格-注振幅'):
            process_amplitude_grid(cursor, conn, tick, code, price, quantity, buy_sell, strategy, remark, datetime_str, rowid)
        else:
            pass  # 其他策略



# 问财根据委托备注获取数据。参数分别是：委托备注、委托数量、策略名称。例如portfolio_rotation(order_note, order_quantity, strategy_name)
def portfolio_rotation(order_note, order_quantity, strategy_name):
    try:
        if '基金' in order_note:
            query_type = 'fund'
        elif '转债' in order_note:
            query_type = 'conbond'
        else:
            query_type = 'stock'

        data = pywencai.get(query=order_note, query_type=query_type, loop=True)
        
        # 提取第一列数据并重命名为"证券代码"
        securities_codes = data.iloc[:, 0].rename('证券代码').to_frame()
        
        # 增加多列数据
        securities_codes = securities_codes.assign(
            委托备注=order_note,
            委托数量=order_quantity,
            策略名称=strategy_name
        )

        return securities_codes
    
    except Exception as e:
        # 捕获所有异常并打印错误信息
        print(f"发生异常: {e}")
        return None  # 返回 None 或其他适当的值以表示失败
    


# 根据委托备注获取数据
def process_scheduled_tasks(scheduled_tasks, cursor, conn):
    """
    处理定时任务
    """
    try:
        for row in scheduled_tasks:
            rowid, code, price, quantity, buy_sell, strategy, remark, datetime_str = row
            # 从'委托备注'获取查询参数
            query = remark
            # 调用portfolio_rotation(query)
            A_data = portfolio_rotation(query, quantity, strategy)
            # 假设A_data有列['证券代码', '委托数量', '策略名称', '委托备注']
            if A_data.empty:
                continue
            # 删除'execute_general_trade'中'策略名称' == '问财轮动-注询问'且'买卖' == -1的行
            cursor.execute("""
                DELETE FROM execute_general_trade
                WHERE 策略名称='问财轮动-注询问' AND 买卖=-1
            """)
            conn.commit()
            # 从'execute_general_trade'获取相同'委托备注'的数据组
            cursor.execute("""
                SELECT 证券代码, 成交数量, 策略名称, 委托备注
                FROM execute_general_trade
                WHERE 委托备注=?
            """, (remark,))
            B_rows = cursor.fetchall()
            B_data = set(row[0] for row in B_rows)
            A_securities = set(A_data['证券代码'])
            securities_to_buy = A_securities - B_data
            securities_to_sell = B_data - A_securities
            for buy_code in securities_to_buy:
                row_data = A_data[A_data['证券代码'] == buy_code].iloc[0]
                insert_order(cursor, conn, buy_code, 0, quantity, 1, strategy, remark, datetime.now().strftime('%Y%m%d %H:%M'))
            for sell_code in securities_to_sell:
                cursor.execute("""
                    SELECT 成交数量
                    FROM execute_general_trade
                    WHERE 策略名称=? AND 证券代码=? AND 委托备注=?
                """, (strategy, sell_code, remark))
                execute_rows = cursor.fetchall()
                total_quantity = sum(row[0] for row in execute_rows)
                insert_order(cursor, conn, sell_code, 0, total_quantity, -1, strategy, remark, datetime.now().strftime('%Y%m%d %H:%M'))
                cursor.execute("""
                    DELETE FROM execute_general_trade
                    WHERE 策略名称=? AND 证券代码=? AND 委托备注=?
                """, (strategy, sell_code, remark))
                conn.commit()
    except Exception as e:
        # 捕获所有异常并打印错误信息
        print(f"发生异常: {e}")
        # 回滚事务以确保数据一致性
        conn.rollback()
        return None  # 返回 None 或其他适当的值以表示失败




