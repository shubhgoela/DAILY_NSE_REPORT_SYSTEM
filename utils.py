import os
import pandas as pd
from datetime import datetime, date
import numpy as np
import requests


from logging_config import setup_logging
from bhavcopy_utils import file_exists, save_to_archive


NSE_base_url = os.getenv('NSE_base_url')
INDEX_LIST = os.getenv('INDEX_LIST').split(',')
environment = os.getenv('APP_ENV', 'DEBUG')

NSE_PRICE_DATA_OPEN = os.getenv("NSE_PRICE_DATA_OPEN")
NSE_PRICE_DATA_HIGH = os.getenv("NSE_PRICE_DATA_HIGH")
NSE_PRICE_DATA_LOW = os.getenv("NSE_PRICE_DATA_LOW")
NSE_PRICE_DATA_CLOSE = os.getenv("NSE_PRICE_DATA_CLOSE")
NSE_PRICE_DATA_DELIV_QTY = os.getenv("NSE_PRICE_DATA_DELIV_QTY")
NSE_PRICE_DATA_DELIV_PER = os.getenv("NSE_PRICE_DATA_DELIV_PER")
NSE_VOLUME_DATA = os.getenv("NSE_VOLUME_DATA")


logger = setup_logging(logger_name='nse_logger',
                       info_file='nse_logger_info.log', 
                       warning_file='nse_logger_warning.log', 
                       error_file='nse_logger_error.log', 
                       environment=environment)

o_logger = setup_logging(logger_name='o_loger',
                       info_file='o_logger_specific_info.log', 
                        warning_file='o_logger_specific_warning.log', 
                        error_file='o_logger_specific_error.log', 
                        environment=environment)

h_logger = setup_logging(logger_name='h_loger',
                       info_file='h_logger_specific_info.log', 
                        warning_file='h_logger_specific_warning.log', 
                        error_file='h_logger_specific_error.log', 
                        environment=environment)

l_logger = setup_logging(logger_name='l_loger',
                       info_file='l_logger_specific_info.log', 
                        warning_file='l_logger_specific_warning.log', 
                        error_file='l_logger_specific_error.log', 
                        environment=environment)    

c_logger = setup_logging(logger_name='c_loger',
                       info_file='c_logger_specific_info.log', 
                        warning_file='c_logger_specific_warning.log', 
                        error_file='c_logger_specific_error.log', 
                        environment=environment)

v_logger = setup_logging(logger_name='v_loger',
                       info_file='v_logger_specific_info.log', 
                        warning_file='v_logger_specific_warning.log', 
                        error_file='v_logger_specific_error.log', 
                        environment=environment)

d_logger = setup_logging(logger_name='d_loger',
                       info_file='d_logger_specific_info.log', 
                        warning_file='d_logger_specific_warning.log', 
                        error_file='d_logger_specific_error.log', 
                        environment=environment)

dp_logger = setup_logging(logger_name='dp_loger',
                       info_file='dp_logger_specific_info.log', 
                        warning_file='dp_logger_specific_warning.log',  
                        error_file='dp_logger_specific_error.log', 
                        environment=environment)


month_abbreviations = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

abbreviation_to_month = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11,'Dec': 12
    }

# ========= Column Configurations ========= #
base_columns = {
    "open": {
        "required": ['SYMBOL',' SERIES', ' DATE1', ' OPEN_PRICE'],
        "renamed": ['SYMBOL', 'SERIES', 'Date', 'OPEN_PRICE'],
        "pivot_value": 'OPEN_PRICE',
        "outfile": NSE_PRICE_DATA_OPEN,
        "logger": o_logger
    },
    "high": {
        "required": ['SYMBOL',' SERIES', ' DATE1', ' HIGH_PRICE'],
        "renamed": ['SYMBOL', 'SERIES', 'Date', 'HIGH_PRICE'],
        "pivot_value": 'HIGH_PRICE',
        "outfile": NSE_PRICE_DATA_HIGH,
        "logger": h_logger
    },
    "low": {
        "required": ['SYMBOL',' SERIES', ' DATE1', ' LOW_PRICE'],
        "renamed": ['SYMBOL', 'SERIES', 'Date', 'LOW_PRICE'],
        "pivot_value": 'LOW_PRICE',
        "outfile": NSE_PRICE_DATA_LOW,
        "logger": l_logger
    },
    "close": {
        "required": ['SYMBOL',' SERIES', ' DATE1', ' CLOSE_PRICE'],
        "renamed": ['SYMBOL', 'SERIES', 'Date', 'CLOSE_PRICE'],
        "pivot_value": 'CLOSE_PRICE',
        "outfile": NSE_PRICE_DATA_CLOSE,
        "logger": c_logger
    },
    "volume": {
        "required": ['SYMBOL',' SERIES', ' DATE1', ' TTL_TRD_QNTY'],
        "renamed": ['SYMBOL', 'SERIES', 'Date', 'TTL_TRD_QNTY'],
        "pivot_value": 'TTL_TRD_QNTY',
        "outfile": NSE_VOLUME_DATA,
        "logger": v_logger
    },
    "delivery_qty": {
        "required": ['SYMBOL',' SERIES', ' DATE1', ' DELIV_QTY'],
        "renamed": ['SYMBOL', 'SERIES', 'Date', 'DELIV_QTY'],
        "pivot_value": 'DELIV_QTY',
        "outfile": NSE_PRICE_DATA_DELIV_QTY,
        "logger": d_logger
    },
    "delivery_per": {
        "required": ['SYMBOL',' SERIES', ' DATE1', ' DELIV_PER'],
        "renamed": ['SYMBOL', 'SERIES', 'Date', 'DELIV_PER'],
        "pivot_value": 'DELIV_PER',
        "outfile": NSE_PRICE_DATA_DELIV_PER,
        "logger": dp_logger
    }
}


def create_or_add_master_data(filepath, date, config_key):
    config = base_columns[config_key]

    print(config)
    df = pd.read_csv(filepath)
    df = df[config["required"]]
    df.columns = config["renamed"]

    # Keep only EQ, BE series
    df = df[(df['SERIES'] == ' EQ') | (df['SERIES'] == ' BE')]
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

    pivot_data = df.pivot(columns='SYMBOL', index='Date', values=config["pivot_value"])
    pivot_data.reset_index(inplace=True)
    pivot_data['Date'] = pd.to_datetime(pivot_data['Date'])

    outfile = config["outfile"]
    print(outfile)
    logger = config["logger"]

    if file_exists(outfile):
        nse_data = pd.read_csv(outfile)
    else:
        nse_data = pd.DataFrame(columns=['Date'] + list(df['SYMBOL'].values))

    if pivot_data.empty:
        logger.critical(f"Pivot data is empty. Please check input. - {date}")
        return False

    # Handle missing new stocks
    missing_columns = set(pivot_data.columns) - set(nse_data.columns)
    if len(missing_columns) > 0:
        for col in missing_columns:
            logger.warning(f"Stock missing in nse_data: {col}  - {date}")
        missing_data = pd.DataFrame({col: pd.NA for col in missing_columns}, index=nse_data.index)
        nse_data = pd.concat([nse_data, missing_data], axis=1)

    # Handle missing old stocks
    missing_columns_pivot = set(nse_data.columns) - set(pivot_data.columns)
    if len(missing_columns_pivot) > 0:
        num_rows = pivot_data.shape[0]
        new_columns_df = pd.DataFrame({col: [np.nan]*num_rows for col in missing_columns_pivot})
        for col in missing_columns_pivot:
            logger.warning(f"Stock missing in pivot_data: {col}  - {date}")
        pivot_data = pd.concat([pivot_data, new_columns_df], axis=1)

    # Merge
    nse_data = pd.concat([nse_data, pivot_data], ignore_index=True)

    nse_data['Date'] = pd.to_datetime(nse_data['Date'])
    nse_data = nse_data.sort_values(by='Date')
    nse_data = nse_data.groupby('Date', as_index=False).first()
    nse_data = nse_data.copy()

    # Fill NaN only for volume-like data
    if config_key in ['volume']:
        nse_data.fillna(0, inplace=True)

    nse_data.to_csv(outfile, index=False)
    return True


def save_bhav_copy_data(bhav_copy_archive_path, date, data):
    """
    Save bhavcopy data to the respective file.
    
    Parameters:
        date (str): Date of the data in 'YYYY-MM-DD' format.
        config_key (str): Key to identify the configuration for saving data.
    """
    date = date.date().isoformat()
    year, month, day = date.split("-")
    month = month_abbreviations[int(month)]
    file_name = f"Bhav_Copy_{day}-{month}-{year}.csv"
    file_exist = file_exists(bhav_copy_archive_path+file_name)
    if not file_exist:
        if isinstance(data, pd.DataFrame):
                filepath = save_to_archive(bhav_copy_archive_path, data, file_name)

    return filepath

def get_google_script_data(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error if request failed
        data = response.json()       # Assuming endpoint returns JSON
        return pd.DataFrame(data['data'])    # Convert to DataFrame
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()