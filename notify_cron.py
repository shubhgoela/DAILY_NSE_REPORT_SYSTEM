import time
import random
from dotenv import find_dotenv, load_dotenv
import os
import time
from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np
import time
from PIL import Image
import calendar


from bhavcopy_login import login
from bhavcopy_utils import *
from queries import get_mail_template, get_holidays_for_year, get_exception_trading_dates_to_year, get_expry_days
from utils import (save_bhav_copy_data, create_or_add_master_data, get_google_script_data)
from corp_actions import adjust_corp_actions
from financial_analyser import FinancialDataAnalyzer

dotenv_path = find_dotenv()

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    print("No .env file found")

NSE_base_url = os.getenv('NSE_base_url')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT'))
BHAVCOPY_FILE_NAMES = ['F&O-UDiFF Common Bhavcopy Final (zip)', 'F&O-Participant wise Open Interest (csv)', 'Full Bhavcopy and Security Deliverable data']
MAIL_RECIPIENTS = ['goela.shubh@gmail.com']
MAIL_RECIPIENTS_REPORT = ['goela.shubh@gmail.com', 'goela.engineers@gmail.com']

BHAV_COPY_ARCHIVE_PATH = os.getenv('BHAV_COPY_ARCHIVE_PATH', 'bhav_copy_archive')
FO_BHAV_COPY_ARCHIVE_PATH = os.getenv('FO_BHAV_COPY_ARCHIVE_PATH', 'fo_bhav_copy_archive')



def check_for_files( eq_section_id= "cr_equity_daily_Current", der_section_id="cr_deriv_equity_daily_Current"):

    filtered_docs = []
    print(eq_section_id,',',der_section_id)
    # cr_deriv_equity_daily_Previous, cr_deriv_equity_daily_Current
    derivative_file_details = get_display_and_file_names(NSE_base_url, endpoint = '/all-reports-derivatives', section_id = der_section_id)
    if derivative_file_details != []:
        for file in derivative_file_details:
            if file['display_name'] in BHAVCOPY_FILE_NAMES:
                filtered_docs.append(file)
    
    # cr_equity_daily_Previous, cr_equity_daily_Current
    equity_file_details = get_display_and_file_names(NSE_base_url, endpoint = '/all-reports', section_id= eq_section_id)
    if equity_file_details != []:
        for file in equity_file_details:
            if file['display_name'] in BHAVCOPY_FILE_NAMES:
                filtered_docs.append(file)
    
    file_names = [file.get('display_name') for file in filtered_docs]

    return file_names, filtered_docs


def process_filter_docs_for_noti(filtered_docs, template_name = 'bhavcopy_noti', sent_files = []):
    print('in process_filter_docs_for_noti')
    print(sent_files)
    print(filtered_docs)
    if len(filtered_docs) > 0:
        for file in filtered_docs:
            print('generating html...')
            template = get_mail_template(template_name)
            template = None
            body = generate_html_table(template, [file])

            if file.get('display_name') not in sent_files:
                # file_names.append(file.get('display_name'))
                if template is None:
                    mail_sent = send_email( 
                        recipient_emails=MAIL_RECIPIENTS,
                        # recipient_emails= ['shubh.goela@mnclgroup.com'],
                        subject=file.get('display_name'),
                        body=body,
                        html_body=body)  
                else:  
                    mail_sent = send_email( 
                        recipient_emails=template.get('recipients', MAIL_RECIPIENTS),
                        subject=file.get('display_name'),
                        body=body,
                        html_body=body)
    return True


def extract_date_from_csv(text, year):
    match_1 = re.search(r'([A-Za-z]+)\s(\d{1,2})', text)
    if match_1:
        month = match_1.group(1)
        day = match_1.group(2)
    
    match_2 = re.search(r'\d{4}', year)
    if match_2:
        year = int(match_2.group(0))  # Convert to integer
        
    date_str = f"{day} {month} {year}"
    date_obj = datetime.strptime(date_str, "%d %b %Y").strftime('%d/%m/%y')
    
    return date_obj


def process_open_interest_df(df):
    dt = extract_date_from_csv(df.columns[0], df.columns[1])
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = [v.strip() for v in df.iloc[0]]
    df = df.loc[1:]
    for column in df.columns:
        if df[column].apply(pd.to_numeric, errors='coerce').notna().all():
            df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df, dt


def create_index_futures_table(cr_df, prev_df, cr_dt, prev_dt):
    merged_df = cr_df.merge(prev_df, on="Client Type", suffixes=(f'_{cr_dt}', f'_{prev_dt}'))
    merged_df["Future Index Long Change"] = merged_df[f"Future Index Long_{cr_dt}"] - merged_df[f"Future Index Long_{prev_dt}"]
    merged_df["Future Index Short Change"] = merged_df[f"Future Index Short_{cr_dt}"] - merged_df[f"Future Index Short_{prev_dt}"]
    merged_df['Signal'] = np.where(
    (merged_df['Future Index Long Change'] - merged_df['Future Index Short Change']) > 0,
        'Bullish',
        'Bearish'
    )
    result_df = merged_df[[
        "Client Type", 
        f"Future Index Long_{cr_dt}", f"Future Index Short_{cr_dt}",  # Current values
        f"Future Index Long_{prev_dt}", f"Future Index Short_{prev_dt}", 
        "Future Index Long Change", "Future Index Short Change", 'Signal' # Daily change
    ]]
    result_df.iloc[-1, -1] = ''
    v1 = result_df.loc[result_df['Client Type'] == 'FII',f'Future Index Long_{cr_dt}'].values[0]
    v2 = result_df.loc[result_df['Client Type'] == 'FII',f'Future Index Short_{cr_dt}'].values[0]
    long_exposure = round(((v1/(v1+v2))*100),2)
    return result_df , long_exposure 


def create_index_call_table(cr_df, prev_df, cr_dt, prev_dt):
    merged_df = cr_df.merge(prev_df, on="Client Type", suffixes=(f'_{cr_dt}', f'_{prev_dt}'))

    merged_df["Option Index Call Long Change"] = merged_df[f"Option Index Call Long_{cr_dt}"] - merged_df[f"Option Index Call Long_{prev_dt}"]
    merged_df["Option Index Call Short Change"] = merged_df[f"Option Index Call Short_{cr_dt}"] - merged_df[f"Option Index Call Short_{prev_dt}"]

    merged_df['Signal'] = np.where(
    (merged_df['Option Index Call Long Change'] - merged_df['Option Index Call Short Change']) > 0,
        'Bullish',
        'Bearish'
    )
    result_df = merged_df[[
        "Client Type", 
        f"Option Index Call Long_{cr_dt}", f"Option Index Call Short_{cr_dt}",  # Current values
        f"Option Index Call Long_{prev_dt}", f"Option Index Call Short_{prev_dt}", 
        "Option Index Call Long Change", "Option Index Call Short Change", 'Signal' # Daily change
    ]]
    result_df.iloc[-1, -1] = ''
    return result_df


def create_index_put_table(cr_df, prev_df, cr_dt, prev_dt):
    merged_df = cr_df.merge(prev_df, on="Client Type", suffixes=(f'_{cr_dt}', f'_{prev_dt}'))

    merged_df["Option Index Put Long Change"] = merged_df[f"Option Index Put Long_{cr_dt}"] - merged_df[f"Option Index Put Long_{prev_dt}"]
    merged_df["Option Index Put Short Change"] = merged_df[f"Option Index Put Short_{cr_dt}"] - merged_df[f"Option Index Put Short_{prev_dt}"]

    merged_df['Signal'] = np.where(
    (merged_df['Option Index Put Long Change'] - merged_df['Option Index Put Short Change']) < 0,
        'Bullish',
        'Bearish'
    )
    result_df = merged_df[[
        "Client Type", 
        f"Option Index Put Long_{cr_dt}", f"Option Index Put Short_{cr_dt}",  # Current values
        f"Option Index Put Long_{prev_dt}", f"Option Index Put Short_{prev_dt}", 
        "Option Index Put Long Change", "Option Index Put Short Change", 'Signal' # Daily change
    ]]
    result_df.iloc[-1, -1] = ''
    return result_df


def create_stock_futures_table(cr_df, prev_df, cr_dt, prev_dt):
    merged_df = cr_df.merge(prev_df, on="Client Type", suffixes=(f'_{cr_dt}', f'_{prev_dt}'))
    merged_df["Future Stock Long Change"] = merged_df[f"Future Stock Long_{cr_dt}"] - merged_df[f"Future Stock Long_{prev_dt}"]
    merged_df["Future Stock Short Change"] = merged_df[f"Future Stock Short_{cr_dt}"] - merged_df[f"Future Stock Short_{prev_dt}"]
    merged_df['Signal'] = np.where(
    (merged_df['Future Stock Long Change'] - merged_df['Future Stock Short Change']) > 0,
        'Bullish',
        'Bearish'
    )
    result_df = merged_df[[
        "Client Type", 
        f"Future Stock Long_{cr_dt}", f"Future Stock Short_{cr_dt}",  # Current values
        f"Future Stock Long_{prev_dt}", f"Future Stock Short_{prev_dt}", 
        "Future Stock Long Change", "Future Stock Short Change", 'Signal' # Daily change
    ]]
    result_df.iloc[-1, -1] = ''
    v1 = result_df.loc[result_df['Client Type'] == 'FII',f'Future Stock Long_{cr_dt}'].values[0]
    v2 = result_df.loc[result_df['Client Type'] == 'FII',f'Future Stock Short_{cr_dt}'].values[0]
    long_exposure = round(((v1/(v1+v2))*100),2)
    return result_df , long_exposure 


def create_stock_call_table(cr_df, prev_df, cr_dt, prev_dt):
    merged_df = cr_df.merge(prev_df, on="Client Type", suffixes=(f'_{cr_dt}', f'_{prev_dt}'))

    merged_df["Option Stock Call Long Change"] = merged_df[f"Option Stock Call Long_{cr_dt}"] - merged_df[f"Option Stock Call Long_{prev_dt}"]
    merged_df["Option Stock Call Short Change"] = merged_df[f"Option Stock Call Short_{cr_dt}"] - merged_df[f"Option Stock Call Short_{prev_dt}"]

    merged_df['Signal'] = np.where(
    (merged_df['Option Stock Call Long Change'] - merged_df['Option Stock Call Short Change']) > 0,
        'Bullish',
        'Bearish'
    )
    result_df = merged_df[[
        "Client Type", 
        f"Option Stock Call Long_{cr_dt}", f"Option Stock Call Short_{cr_dt}",  # Current values
        f"Option Stock Call Long_{prev_dt}", f"Option Stock Call Short_{prev_dt}", 
        "Option Stock Call Long Change", "Option Stock Call Short Change", 'Signal' # Daily change
    ]]
    result_df.iloc[-1, -1] = ''
    return result_df


def create_stock_put_table(cr_df, prev_df, cr_dt, prev_dt):
    merged_df = cr_df.merge(prev_df, on="Client Type", suffixes=(f'_{cr_dt}', f'_{prev_dt}'))

    merged_df["Option Stock Put Long Change"] = merged_df[f"Option Stock Put Long_{cr_dt}"] - merged_df[f"Option Stock Put Long_{prev_dt}"]
    merged_df["Option Stock Put Short Change"] = merged_df[f"Option Stock Put Short_{cr_dt}"] - merged_df[f"Option Stock Put Short_{prev_dt}"]

    merged_df['Signal'] = np.where(
    (merged_df['Option Stock Put Long Change'] - merged_df['Option Stock Put Short Change']) < 0,
        'Bullish',
        'Bearish'
    )
    result_df = merged_df[[
        "Client Type", 
        f"Option Stock Put Long_{cr_dt}", f"Option Stock Put Short_{cr_dt}",  # Current values
        f"Option Stock Put Long_{prev_dt}", f"Option Stock Put Short_{prev_dt}", 
        "Option Stock Put Long Change", "Option Stock Put Short Change", 'Signal' # Daily change
    ]]
    result_df.iloc[-1, -1] = ''
    return result_df


def create_OI_table(fo_bhav_copy, XpryDts, index):
    df = pd.DataFrame()
    df[index] = ['Max Call OI','Max Put OI', 'Change in Call OI max', 'Change in Put OI max']
    df['Option Type'] = ['CE','PE','CE','PE']

    for date in XpryDts:
        
        nifty_call = fo_bhav_copy[(fo_bhav_copy['XpryDt'] == date.strftime('%Y-%m-%d')) & (fo_bhav_copy['TckrSymb'] == index)  & (fo_bhav_copy['OptnTp'] == 'CE')]
        nifty_call = nifty_call.loc[nifty_call['OpnIntrst'].idxmax()]['StrkPric']


        nifty_put = fo_bhav_copy[(fo_bhav_copy['XpryDt'] == date.strftime('%Y-%m-%d')) & (fo_bhav_copy['TckrSymb'] == index)  & (fo_bhav_copy['OptnTp'] == 'PE')]
        nifty_put = nifty_put.loc[nifty_put['OpnIntrst'].idxmax()]['StrkPric']

        nifty_call_max_oi_chng = fo_bhav_copy[(fo_bhav_copy['XpryDt'] == date.strftime('%Y-%m-%d')) & (fo_bhav_copy['TckrSymb'] == index)  & (fo_bhav_copy['OptnTp'] == 'CE')]
        nifty_call_max_oi_chng = nifty_call_max_oi_chng.loc[nifty_call_max_oi_chng['ChngInOpnIntrst'].idxmax()]['StrkPric']

        nifty_put_max_oi_chng = fo_bhav_copy[(fo_bhav_copy['XpryDt'] == date.strftime('%Y-%m-%d')) & (fo_bhav_copy['TckrSymb'] == index)  & (fo_bhav_copy['OptnTp'] == 'PE')]
        nifty_put_max_oi_chng = nifty_put_max_oi_chng.loc[nifty_put_max_oi_chng['ChngInOpnIntrst'].idxmax()]['StrkPric']

        df[date.strftime('%d-%b')] = [nifty_call, nifty_put, nifty_call_max_oi_chng, nifty_put_max_oi_chng]

    return df


def get_pcr(fo_bhav_copy, XpryDt, index, ref_date = None):
    if ref_date is None:
        ref_date = datetime.today()
    elif isinstance(ref_date, str):
        ref_date = datetime.strptime(ref_date, "%Y-%m-%d")

    if ref_date.date() == XpryDt:
        fo_data_filterd = fo_bhav_copy[(fo_bhav_copy['XpryDt'] != ref_date.strftime('%Y-%m-%d')) & (fo_bhav_copy['TckrSymb'] == index)]
    else:
        fo_data_filterd = fo_bhav_copy[(fo_bhav_copy['TckrSymb'] == index)]

    fo_data_filterd_ce = fo_data_filterd[(fo_data_filterd['OptnTp'] == 'CE')]
    fo_data_filterd_pe = fo_data_filterd[(fo_data_filterd['OptnTp'] == 'PE')]

    # return fo_data_filterd, fo_data_filterd_ce, fo_data_filterd_pe
    return fo_data_filterd_pe['OpnIntrst'].sum()/ fo_data_filterd_ce['OpnIntrst'].sum()

def create_stock_analysis_report(curr_bhav, fo_bhav_copy, month_expry, symbols):
    """
    Create a stock analysis report using the FinancialDataAnalyzer class.
    
    Args:
        fo_bhav_copy: DataFrame with futures and options data
        symbol: Trading symbol (e.g., 'NIFTY')
        month_expry: Current month expiry date
        curr_close: Current closing price
        prev_close: Previous closing price
    
    Returns:
        HTML report string
    """


    html = ''' '''
    for symbol in symbols:
        prev_close = curr_bhav[curr_bhav['SYMBOL'] == symbol][' PREV_CLOSE'].values[0]
        curr_close = curr_bhav[curr_bhav['SYMBOL'] == symbol][' CLOSE_PRICE'].values[0]
        analyzer = FinancialDataAnalyzer(fo_bhav_copy, symbol, month_expry, curr_close, prev_close)
        html += analyzer.get_html_report() + '<br><hr>'
    
    html += '<p style="margin-top:20px; font-size:12px; color:#6b7280;">Generated automatically</p>'
    return html


def loop_question_between_times(start_time="00:00", end_time="23:00", interval=60):
    """
    Continuously prompts a question between specified times.

    Parameters:
        question (str): The question to be asked in the loop.
        start_time (str): The start time in HH:MM format (24-hour format).
        end_time (str): The end time in HH:MM format (24-hour format).
        interval (int): The number of seconds to wait between each prompt.

    """
    today = datetime.now().strftime('%Y-%m-%d')
    start = datetime.strptime(start_time, "%H:%M").time()
    end = datetime.strptime(end_time, "%H:%M").time()

    session = login(NSE_base_url)
    MAX_RETRIES = 5 
    retry_count = 0
    current_df, prev_df, curr_fo_bhav, curr_bhav = None, None, None, None

    sent_files = []
    filtered_docs = []
    while True:
        now = datetime.now().time()
        # print('now: ', now)
        # print(files)
        if start <= now <= end and set(sent_files) != set(BHAVCOPY_FILE_NAMES):
            f, fd = check_for_files( eq_section_id="cr_equity_daily_Current" , der_section_id="cr_deriv_equity_daily_Current")
            process_filter_docs_for_noti(filtered_docs = fd, sent_files=sent_files)
            if f != []:
                sent_files.extend(f)
                sent_files = list(set(sent_files))
                filtered_docs.extend(fd)
                filtered_docs = list({tuple(sorted(d.items())) for d in filtered_docs})
                filtered_docs = [dict(t) for t in filtered_docs]

            print('sleeping...')
            time.sleep(interval)
        else:
            if now > end:
                print("The time window has closed. Exiting loop.")
            break
    


    while retry_count <= MAX_RETRIES:
        retry_count += 1
        for file in filtered_docs:
            if file['display_name'] == 'F&O-Participant wise Open Interest (csv)':
                response = session.get(file['file_link'])
                try:
                    response.raise_for_status()
                    if response.status_code == 200:
                        current_df = handle_file_response(response, today)
                        current_df, current_date = process_open_interest_df(current_df)
                except Exception as e:
                    current_df = None
                    print(f"Error processing current OI file: {e}")
                    continue
            
            if file['display_name'] == 'F&O-UDiFF Common Bhavcopy Final (zip)':
                response = session.get(file['file_link'])
                try:
                    response.raise_for_status()
                    if response.status_code == 200:
                        curr_fo_bhav = handle_file_response(response, today)
                except Exception as e:
                    curr_fo_bhav = None
                    print(f"Error processing current F&O Bhavcopy file: {e}")
                    continue
            
            if file['display_name'] == 'Full Bhavcopy and Security Deliverable data':
                response = session.get(file['file_link'])
                try:
                    response.raise_for_status()
                    if response.status_code == 200:
                        curr_bhav = handle_file_response(response, today)
                except Exception as e:
                    curr_bhav = None
                    print(f"Error processing current F&O Bhavcopy file: {e}")
                    continue

            if current_df is not None and curr_fo_bhav is not None and curr_bhav is not None:
                break
            

        f, fd = check_for_files( eq_section_id="cr_equity_daily_Previous" , der_section_id="cr_deriv_equity_daily_Previous")
        for file in fd:
            if file['display_name'] == 'F&O-Participant wise Open Interest (csv)':
                response = session.get(file['file_link'])
                try:
                    response.raise_for_status()
                    if response.status_code == 200:
                        prev_df = handle_file_response(response, today)
                        prev_df, prev_date = process_open_interest_df(prev_df)
                        break
                except Exception as e:
                    prev_df = None
                    print(f"Error processing previous OI file: {e}")
                    continue
        
        if current_df is not None and prev_df is not None:
            break


    print('########curren_df')
    print(current_df)
    print('########previous_df')
    print(prev_df)


    r1, index_long_exposure = create_index_futures_table(current_df, prev_df, current_date, prev_date)
    r2 = create_index_call_table(current_df, prev_df, current_date, prev_date)
    r3 = create_index_put_table(current_df, prev_df, current_date, prev_date)
    r4, stock_long_exposure = create_stock_futures_table(current_df, prev_df, current_date, prev_date)
    r5 = create_stock_call_table(current_df, prev_df, current_date, prev_date)
    r6 = create_stock_put_table(current_df, prev_df, current_date, prev_date)

    month_expry_day = get_expry_days(exchange='NSE', index='NIFTY', expiry_type='month')
    week_expry_day = get_expry_days(exchange='NSE', index='NIFTY', expiry_type='week')
    week_expry, month_expry = get_expiry_dates(weekly_day= week_expry_day, monthly_day=month_expry_day)
    OI_table = create_OI_table(curr_fo_bhav, [week_expry, month_expry], 'NIFTY')
    nifty_pcr = get_pcr(curr_fo_bhav, XpryDt = week_expry, index = 'NIFTY' )


    extra_table_data = create_html_for_exposure(index_long_exposure, stock_long_exposure)
    # commentry = get_commentry(r1, index_long_exposure)
    commentry = get_commentry(r1, r2, r3, index_long_exposure, nifty_pcr, OI_table, week_expry, month_expry)
    html = create_html_table_with_predefined_html([{'heading': 'Index Futures', 'df': r1},
                                                {'heading': 'Stock Futures', 'df': r4},
                                                {'heading': 'Index Call Options', 'df': r2},
                                                {'heading': 'Stock Call Options', 'df': r5},
                                                {'heading': 'Index Put Options', 'df': r3},
                                                {'heading': 'Stock Put Options', 'df': r6}], extra_table_data, commentry)
    


    try:
        save_html_as_png(html_string=html)
    except Exception as e:
        print('Failed to save html as image')
        pass

    send_email( 
                recipient_emails=['shubh.goela@mnclgroup.com'],
                bcc_emails=MAIL_RECIPIENTS_REPORT,
                subject='Participant Wise Derivatives FII-DII Data',
                body=html,
                html_body=html,
                attachments=['output.png'])

    delete_file('output.png')

    curr_bhav_filepath = save_bhav_copy_data(bhav_copy_archive_path=BHAV_COPY_ARCHIVE_PATH, date=datetime.now(), data=curr_bhav)
    curr_bhav_fo_filepath = save_bhav_copy_data(bhav_copy_archive_path=FO_BHAV_COPY_ARCHIVE_PATH, date=datetime.now(), data=curr_fo_bhav)

    create_or_add_master_data(curr_bhav_filepath, today, "open")
    create_or_add_master_data(curr_bhav_filepath, today, "high")
    create_or_add_master_data(curr_bhav_filepath, today, "low")
    create_or_add_master_data(curr_bhav_filepath, today, "close")
    create_or_add_master_data(curr_bhav_filepath, today, "volume")
    create_or_add_master_data(curr_bhav_filepath, today, "delivery_qty")
    create_or_add_master_data(curr_bhav_filepath, today, "delivery_per")

    adjust_corp_actions(date=datetime.now())

    month_expry_day = get_expry_days(exchange='NSE', index='stock_options', expiry_type='month')
    week_expry_day = get_expry_days(exchange='NSE', index='stock_options', expiry_type='week')
    week_expry, month_expry = get_expiry_dates(weekly_day= week_expry_day, monthly_day=month_expry_day)

    url = "https://script.google.com/macros/s/AKfycbw1xGzZU3wBUIZgXRYbWQGtXWPtr85EhJous66iToxI3xsLG-3YtyEUYfWf6Qd1G9phSw/exec?endpoint=getData&key=goela1008&sheet=TICKER%20LIST"
    df = get_google_script_data(url)
    symbols = list(df['STOCK_NAME'])
    html = create_stock_analysis_report(curr_bhav, curr_fo_bhav, month_expry, symbols)

    send_email( 
        recipient_emails=MAIL_RECIPIENTS_REPORT,
        subject='Stock Analysis Report',
        body=html,
        html_body=html)
    
    return True


def daily_runner():
    last_processed_date = None

    while True:
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        start_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
        end_time = now.replace(hour=23, minute=59, second=59, microsecond=0)

        if start_time <= now <= end_time and today_str != last_processed_date and is_valid_date(now):
            print(f"Running process for {today_str}")
            try:
                result = loop_question_between_times()
                if result:
                    last_processed_date = today_str
            except Exception as e:
                print(f"Error processing data for {today_str}: {e}")
        else:
            print(f"Waiting... ({now.strftime('%H:%M:%S')})")
        
        next_run_time = get_next_valid_trading_datetime(now)
        sleep_duration = (next_run_time - datetime.now()).total_seconds()
        print(f"Sleeping for {int(sleep_duration)} seconds until {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(sleep_duration)


if __name__ == "__main__":
    daily_runner()
