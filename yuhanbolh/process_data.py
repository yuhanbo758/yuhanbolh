
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


# 通过代码，获取价值大师的价值线
def get_valuation_ratios(code):
    try:
        url = f'https://www.gurufocus.cn/stock/{code}/term/gf_value'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 获取大师价值线
        element_jzx = soup.select_one('#term-page-title').text.strip()
        element_jzx = re.findall("\d+\.\d+", element_jzx)[0]

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



# 将数据保存为sql
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


# 获取data数据中的第几行数据
def get_row(data, index):
    return data.iloc[[index]].reset_index(drop=True)


def MA(data, n):
    MA = pd.Series(data['close'].rolling(n).mean(), name='MA_' + str(n))
    close = data['close']
    signal = np.where(MA < close, 1, np.where(MA > close, -1, 0))
    return pd.DataFrame(signal, columns=['MA_' + str(n)])  # 修改这行


# 获取指数移动平均线，参数有2个，一个是数据源，一个是日期
def EMA(data, n):
    EMA = pd.Series(data['close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    close = data['close']
    signal = np.where(EMA < close, 1, np.where(EMA > close, -1, 0))
    return pd.DataFrame(signal, columns=['EMA_' + str(n)])


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
    # 计算买入和卖出信号
    buy_signal = (base_line < data['close']) & (conversion_line > data['close'].shift(1)) & (conversion_line <= data['close']) & (lead_line1 > data['close']) & (lead_line1 > lead_line2)
    sell_signal = (base_line > data['close']) & (conversion_line < data['close'].shift(1)) & (conversion_line >= data['close']) & (lead_line1 < data['close']) & (lead_line1 < lead_line2)
    # 使用np.where生成信号列，买入为1，卖出为-1，中立为0
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame(signal, columns=['Ichimoku'])



# 成交量加权移动平均线 VWMA (data, 20)，参数有2个，1个是数据源，另一个是日期，通过为20
def VWMA(data, n):
    # 计算VWMA
    vwma = (data['close'] * data['volume']).rolling(n).sum() / data['volume'].rolling(n).sum()
    
    # 根据VWMA和收盘价计算买卖信号
    buy_signal = vwma < data['close']
    sell_signal = vwma > data['close']
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame(signal, columns=['VWMA_' + str(n)])


# 计算Hull MA船体移动平均线 Hull MA (data,9)，参数有2，一个是数据源，另一个是日期，一般为9
def HullMA(data, n=9):
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
def RSI(data, n):
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

    main_line = d
    signal_line = d_signal
    main_line_prev = d.shift(1)
    signal_line_prev = d_signal.shift(1)

    buy_signal = (main_line < 20) & (main_line_prev < signal_line_prev) & (main_line > signal_line)
    sell_signal = (main_line > 80) & (main_line_prev > signal_line_prev) & (main_line < signal_line)
    signal = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return pd.DataFrame({'STOK_' + str(n): signal})



# 计算CCI指标，参数有2，一个是数据源，另一个是日期，一般为20，即CCI(data, 20)
def CCI(data, n):
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
def ADX(data, n):
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
def AO(data):
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
def MTM(data):
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
def MACD_Level(data, n_fast, n_slow):
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
def Stoch_RSI(data, smoothK=3, smoothD=3, lengthRSI=14, lengthStoch=14):
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
def WPR(data, n):
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
def BBP(data, n):
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
def UO(data, n1, n2, n3):
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
def linear_regression_dfcf(data, years_list):
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

