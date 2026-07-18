
import pandas as pd
import requests
import time


# 获取东财a股全部股票数据
def stock_zh_a_spot_em() -> pd.DataFrame:
    """
    东方财富网-沪深京 A 股-实时行情
    https://quote.eastmoney.com/center/gridlist.html#hs_a_board
    :return: 实时行情
    :rtype: pandas.DataFrame
    """
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    
    # 创建一个空的DataFrame来存储所有数据
    all_data = pd.DataFrame()
    page = 1
    
    while True:
        try:
            params = {
                "pn": str(page),  # 页码
                "pz": "1000",     # 每页数据条数
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f3",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
                "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,"
                "f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
            }
            
            r = requests.get(url, timeout=15, params=params)
            r.raise_for_status()  # 检查请求是否成功
            data_json = r.json()
            
            if not data_json.get("data") or not data_json["data"].get("diff"):
                print(f"\n已到达数据末尾，共获取{len(all_data)}条数据")
                break
                
            temp_df = pd.DataFrame(data_json["data"]["diff"])
            if temp_df.empty:
                break
                
            temp_df.columns = [
                "_",
                "最新价",
                "涨跌幅",
                "涨跌额",
                "成交量",
                "成交额",
                "振幅",
                "换手率",
                "市盈率-动态",
                "量比",
                "5分钟涨跌",
                "代码",
                "_",
                "名称",
                "最高",
                "最低",
                "今开",
                "昨收",
                "总市值",
                "流通市值",
                "涨速",
                "市净率",
                "60日涨跌幅",
                "年初至今涨跌幅",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            ]
            
            temp_df = temp_df[
                [
                    "代码",
                    "名称",
                    "最新价",
                    "涨跌幅",
                    "涨跌额",
                    "成交量",
                    "成交额",
                    "振幅",
                    "最高",
                    "最低",
                    "今开",
                    "昨收",
                    "量比",
                    "换手率",
                    "市盈率-动态",
                    "市净率",
                    "总市值",
                    "流通市值",
                    "涨速",
                    "5分钟涨跌",
                    "60日涨跌幅",
                    "年初至今涨跌幅",
                ]
            ]
            
            # 数据类型转换
            numeric_columns = temp_df.columns.difference(["代码", "名称"])
            for col in numeric_columns:
                temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
                
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
            print(f"已获取第{page}页数据，当前共{len(all_data)}条记录")
            
            page += 1
            time.sleep(0.5)  # 添加短暂延时，避免请求过快
            
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误: {e}")
            break
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
            break
    
    if len(all_data) > 0:
        # 添加序号列
        all_data.insert(0, "序号", range(1, len(all_data) + 1))
        return all_data
    else:
        print("未获取到任何数据")
        return pd.DataFrame()





# 获取东财可转债比价表
def bond_cov_comparison() -> pd.DataFrame:
    """
    东方财富网-行情中心-债券市场-可转债比价表
    https://quote.eastmoney.com/center/fullscreenlist.html#convertible_comparison
    :return: 可转债比价表数据
    :rtype: pandas.DataFrame
    """
    url = "https://16.push2.eastmoney.com/api/qt/clist/get"
    
    # 创建一个空的DataFrame来存储所有数据
    all_data = pd.DataFrame()
    page = 1
    
    while True:
        try:
            params = {
                "pn": str(page),  # 页码
                "pz": "1000",     # 每页数据条数
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f243",
                "fs": "b:MK0354",
                "fields": "f1,f152,f2,f3,f12,f13,f14,f227,f228,f229,f230,f231,f232,f233,f234,"
                "f235,f236,f237,f238,f239,f240,f241,f242,f26,f243",
                "_": "1590386857527",
            }
            
            r = requests.get(url, timeout=15, params=params)
            r.raise_for_status()  # 检查请求是否成功
            json_data = r.json()
            
            if not json_data.get("data") or not json_data["data"].get("diff"):
                print(f"\n已到达数据末尾，共获取{len(all_data)}条数据")
                break
                
            temp_df = pd.DataFrame(json_data["data"]["diff"])
            if temp_df.empty:
                break
            temp_df.reset_index(inplace=True)
            temp_df["index"] = range(1, len(temp_df) + 1)
            temp_df.columns = [
                "序号",
                "_",
                "转债最新价",
                "转债涨跌幅",
                "转债代码",
                "_",
                "转债名称",
                "上市日期",
                "_",
                "纯债价值",
                "_",
                "正股最新价",
                "正股涨跌幅",
                "_",
                "正股代码",
                "_",
                "正股名称",
                "转股价",
                "转股价值",
                "转股溢价率",
                "纯债溢价率",
                "回售触发价",
                "强赎触发价",
                "到期赎回价",
                "开始转股日",
                "申购日期",
            ]
            
            temp_df = temp_df[
                [
                    "序号",
                    "转债代码",
                    "转债名称",
                    "转债最新价",
                    "转债涨跌幅",
                    "正股代码",
                    "正股名称",
                    "正股最新价",
                    "正股涨跌幅",
                    "转股价",
                    "转股价值",
                    "转股溢价率",
                    "纯债溢价率",
                    "回售触发价",
                    "强赎触发价",
                    "到期赎回价",
                    "纯债价值",
                    "开始转股日",
                    "上市日期",
                    "申购日期",
                ]
            ]
            
            # 数据类型转换
            numeric_columns = temp_df.columns.difference(["转债代码", "转债名称", "正股代码", "正股名称", "开始转股日", "上市日期", "申购日期"])
            for col in numeric_columns:
                temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
                
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
            print(f"已获取第{page}页数据，当前共{len(all_data)}条记录")
            
            page += 1
            time.sleep(0.5)  # 添加短暂延时，避免请求过快
            
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误: {e}")
            break
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
            break
    
    if len(all_data) > 0:
        # 重新设置序号列
        all_data["序号"] = range(1, len(all_data) + 1)
        return all_data
    else:
        print("未获取到任何数据")
        return pd.DataFrame()