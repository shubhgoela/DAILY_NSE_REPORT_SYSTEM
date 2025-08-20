from dotenv import find_dotenv, load_dotenv
from datetime import datetime, timedelta
import os
import re
import pandas as pd

from NSE_Selenium_login import get_data_with_selenium_nse_api
from queries import save_corp_action, save_all_corp_action, get_adjusted_corp_actions
from utils import file_exists, base_columns


dotenv_path = find_dotenv()

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    print("No .env file found")

NSE_base_url = os.getenv('NSE_base_url')

def read_data(csv):
    df = pd.read_csv(csv, parse_dates=['Date'])
    df = df.loc[:, ~(df.columns.str.contains('^Unnamed') | df.columns.isnull())].copy()
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    return df

def filter_and_enrich_json(json_data):
    # Regular expression to detect patterns like "X:Y" (e.g., "1:1", "2:1")
    ratio_pattern = re.compile(r'(\d+\s*:\s*\d+)')
    face_value_split_pattern_1 = re.compile(r'From\s+Rs\s*(\d+)\s*/?-*\s*Per\s*Share\s*To\s+Rs\s*(\d+)\s*/?-*\s*Per\s*Share')
    face_value_split_pattern_2 = re.compile(r'From\s+Re\s*(\d+)\s*/?-*\s*Per\s*Share\s*To\s+Re\s*(\d+)\s*/?-*\s*Per\s*Share')
    face_value_split_pattern_3 = re.compile(r'From\s+Rs\s*(\d+)\s*/?-*\s*Per\s*Share\s*To\s+Re\s*(\d+)\s*/?-*\s*Per\s*Share')
    face_value_split_pattern_4 = re.compile(r'From\s+Rs\s*(\d+)\s*/?-*\s*Per\s*Share\sUnit\s*To\s+Rs\s*(\d+)\s*/?-*\s*Per\s*Share\sUnit\s')
    face_value_split_pattern_5 = re.compile(r'From\s+Re\s*(\d+)\s*/?-*\s*Per\s*Share\sUnit\s*To\s+Re\s*(\d+)\s*/?-*\s*Per\s*Share\sUnit\s')
    face_value_split_pattern_6 = re.compile(r'From\s+Rs\s*(\d+)\s*/?-*\s*Per\s*Share\sUnit\s*To\s+Re\s*(\d+)\s*/?-*\s*Per\s*Share\sUnit\s')
    face_value_split_pattern_7 = re.compile(r'Rs\.(\d+)\s*/-\s*To\s*Re\.(\d+)\s*/-\s*Per\s*Share')
    dates = []
    enriched_data = []

    for record in json_data:
        try:
            date_obj = datetime.strptime(record['exDate'], "%d-%b-%Y")
        except Exception as e:
            print(e)
            continue

        record['exDate'] = date_obj.strftime("%Y-%m-%d")
        subject = record['subject']

        if "Face Value Split" in subject:
            match = face_value_split_pattern_1.search(subject)
            if not match:
                match = face_value_split_pattern_2.search(subject)
                if not match:
                    match = face_value_split_pattern_3.search(subject)
                    if not match:
                        match = face_value_split_pattern_4.search(subject)
                        if not match:
                            match = face_value_split_pattern_5.search(subject)
                            if not match:
                                match = face_value_split_pattern_6.search(subject)
                                if not match:
                                    match = face_value_split_pattern_7.search(subject)
            if match:
                from_value = int(match.group(1))
                to_value = int(match.group(2))
                ratio = f"1:{from_value // to_value}" 

                # Add ratio and div_value
                record['ratio'] = ratio
                record['div_value'] = from_value / to_value  
                enriched_data.append(record)

        elif "Bonus" in subject:
            match = ratio_pattern.search(subject)
            if match:
                ratio = match.group(0)
                x, y = map(int, ratio.split(":"))
                # Add ratio and div_value for bonus
                record['ratio'] = ratio
                record['div_value'] = (x + y) / y 

                enriched_data.append(record)
                dates.append(record['exDate'])
                    
            # elif "Rights" in subject:
            #     enriched_data.append(record)
        
    return enriched_data, dates


def get_corp_actions(date = datetime.now()):
    try:
        date = date.date().strftime('%d-%m-%Y')
        print(date)
        endpoint = f'/api/corporates-corporateActions?index=equities&from_date={date}&to_date={date}'
        all_corp_actions = get_data_with_selenium_nse_api(NSE_base_url, endpoint)
        corp_actions, dates = filter_and_enrich_json(all_corp_actions)
        all_corp_actions = {
            'created_on': datetime.now(),
            'date': datetime.strptime(date, '%d-%m-%Y'),
            'actions': all_corp_actions
        }

        corp_actions = {
            'created_on': datetime.now(),
            'date': datetime.strptime(date, '%d-%m-%Y'),
            'actions': corp_actions
        }

        return all_corp_actions, corp_actions
    except Exception as e:
        print('####Exception')
        print(e)
        return None, None


def adjust_corp_actions(date = datetime.now()):

    adjusted_actions = get_adjusted_corp_actions(date)

    if adjusted_actions is not None:
        print('Corp actions already adjusted.')
        return True
    
    all_corp_actions, corp_actions = get_corp_actions(date)
    save_all_corp_action(all_corp_actions)
    print(len(corp_actions['actions']))
    if len(corp_actions['actions']) > 0:
        adjust_master_data_for_corp_actions(corp_actions)
        save_corp_action(corp_actions)
    else:
        print('No corporate actions found')


def adjust_master_data_for_corp_actions(corp_actions, configs=['open', 'high', 'low', 'close', 'volume', 'delivery_qty']):
    """
    Adjusts OHLC and Volume master datasets for corporate actions 
    like dividends/splits/bonuses.

    corp_actions: dict
        {
            "actions": [
                {"symbol": "INFY", "exDate": "2025-08-01", "div_value": 2},
                {"symbol": "TCS", "exDate": "2025-07-15", "div_value": 1.5}
            ]
        }
    configs: list of str
        Which datasets to adjust (default: ['open','high','low','close','volume'])
    """

    try:
        for config_key in configs:
            config = base_columns[config_key]
            outfile = config["outfile"]
            logger = config["logger"]

            if not file_exists(outfile):
                logger.warning(f"File not found: {outfile}. Skipping {config_key}")
                continue

            df = pd.read_csv(outfile)
            df['Date'] = pd.to_datetime(df['Date'])
            columns = list(df.columns)

            for action in corp_actions['actions']:
                stock = action['symbol']
                date = pd.to_datetime(action['exDate'])
                div_value = action['div_value']

                if stock not in columns:
                    logger.warning(f"{stock} not found in {config_key} data. Skipping...")
                    continue

                # Prices: divide by corporate action factor
                if config_key in ["open", "high", "low", "close"]:
                    df.loc[df['Date'] < date, stock] = (
                        df.loc[df['Date'] < date, stock] / div_value
                    )

                # Volumes: multiply
                elif config_key == "volume":
                    df.loc[df['Date'] < date, stock] = (
                        df.loc[df['Date'] < date, stock] * div_value
                    ).round(0)

                # Delivery qty also behaves like volume
                elif config_key == "delivery_qty":
                    df.loc[df['Date'] < date, stock] = (
                        df.loc[df['Date'] < date, stock] * div_value
                    ).round(0)

                # Delivery % -> leave unchanged (ratio)
                elif config_key == "delivery_per":
                    logger.info(f"Skipping adjustment for delivery_per ({stock})")

            # Save back
            df.to_csv(outfile, index=False)
            logger.info(f"Updated {config_key} data saved at {outfile}")

        return True

    except Exception as e:
        print("Error in corporate action adjustment:", str(e))
        return False
