

import sys
import pandas as pd

# 策略文件
from .create_strategy import (
    mole_hunting_delegation
)

# 获取金融数据文件
from .get_data import (
    get_tdx_market_address,
    get_financial_data,
    get_security_bars,
    get_security_count,
    get_index_bars,
    get_security_quotes,
    get_history_minute_time_data,
    get_transaction_data,
    get_finance_info,
    download_7_years_data,
    qmt_data_source_download,
    qmt_data_source,
    json_to_dfcf_qmt_jyr,
    json_to_dfcf_qmt,
    json_to_dfcf,
    process_and_merge_data,
    get_clean_data,
    get_snapshot,
    filter_bond_cb_redeem_data_and_save_to_db,
    get_satisfy_redemption,
    wencai_conditional_query
)

# 全局函数
from .global_functions import (
    calculate_unhedged_transactions,
    calculate_unhedged_transactions_sbb,
    calculate_stop_profit_loss,
    check_account,
    copy_table_to_mysql,
    copy_tables,
    generate_mole_strategy,
    get_decimal_places,
    get_existing_data,
    open_positions,
    process_data,
    save_to_database,
    sync_folders
)


# MT5自定义函数
from .mt5_ic_custom import (
    get_all_valuation_ratios_db,
    get_exchange_rate,
    get_mt5_data,
    get_valuation_ratios,
    save_data,
    save_exchange_rates_to_db,
    wencai_conditional_query_nz100,
    calculate_totals
)


# mt5交易委托文件
from .mt5_trade import (
    cancel_order_fn,
    cancel_pending_order,
    close_position_fn,
    execute_order_from_db,
    export_non_strategy_positions,
    export_positions_to_db,
    insert_into_db,
    limit_order_fn,
    market_order_fn,
    process_non_strategy_positions,
    remove_unavailable_products_mt5,
    save_unsettled_orders_to_db
)


# 数据处理文件，主要为技术指标的计算
from .process_data import (
    ADX,
    AO,
    BBP,
    calculate_annual_return,
    calculate_xirr,
    CCI,
    EMA,
    generate_stat_data,
    HullMA,
    ichimoku_cloud,
    linear_regression_dfcf,
    MA,
    MACD_Level,
    MTM,
    RSI,
    Stoch_RSI,
    STOK,
    UO,
    VWMA,
    WPR,
    get_processed_code,
    clean_execute_general_trade,
    insert_order,
    delete_receive_condition_row,
    delete_execute_general_trade_row,
    process_price_grid,
    process_amplitude_grid,
    process_immediate_rows,
    portfolio_rotation,
    process_scheduled_tasks

)


# qmt的委托、交易和推送文件
from .qmt_trade import (
    MyXtQuantTraderCallback, 
    calculate_remaining_holdings,
    cancel_all_orders,
    insert_buy_sell_data,
    place_order_based_on_asset,
    place_orders,
    run_weekdays_at,
    save_daily_data,
    save_daily_orders,
    save_daily_trades,
    save_positions,
    save_stock_asset,
    se_send_email_on_error,
    sort_and_update_table,

)


# 邮件发送文件
from .send_email import send_email


