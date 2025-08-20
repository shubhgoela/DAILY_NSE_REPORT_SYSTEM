import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import calendar
import random
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import io
import xml.etree.ElementTree as ET
from email.message import EmailMessage
import mimetypes
import smtplib
import base64
import json
import subprocess
from requests.exceptions import RequestException


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException



from dotenv import find_dotenv, load_dotenv

from utils import month_abbreviations, abbreviation_to_month
from queries import get_holidays_for_year, get_exception_trading_dates_to_year
from logging_config import setup_logging


logger = setup_logging(logger_name='noti_logger',
                       info_file='noti_logger_info.log', 
                        warning_file='noti_logger_warning.log', 
                        error_file='noti_logger_error.log', 
                        environment="DEBUG")

dotenv_path = find_dotenv()

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    print("No .env file found")

SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT'))

def file_exists(filepath):
    return Path(filepath).is_file()


def get_shared_strings(zf):
    """Extract shared strings from sharedStrings.xml"""
    shared_strings = []
    with zf.open('xl/sharedStrings.xml') as shared_file:
        tree = ET.parse(shared_file)
        root = tree.getroot()
        for si in root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
            shared_strings.append(''.join(node.text for node in si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t") if node.text))
    return shared_strings


def extract_data_from_sheet(zf, sheet_name, shared_strings):
    """Extract data from a specific worksheet, replacing shared string indices with actual values"""
    with zf.open(sheet_name) as sheet_xml:
        tree = ET.parse(sheet_xml)
        root = tree.getroot()

        ns = {'spreadsheet': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        data = []

        for row in root.findall('.//spreadsheet:row', ns):
            row_data = []
            for cell in row.findall('spreadsheet:c', ns):
                # Check if the cell contains a shared string (t="s")
                if cell.attrib.get('t') == 's':
                    # Get the shared string index and fetch the corresponding string
                    shared_string_index = int(cell.find('spreadsheet:v', ns).text)
                    row_data.append(shared_strings[shared_string_index])
                else:
                    # Handle as normal value
                    value = cell.find('spreadsheet:v', ns)
                    row_data.append(value.text if value is not None else None)
            data.append(row_data)
    return data


def read_excel_file(file, logger):
    # Try different Excel engines
    df = []
    engines = ['openpyxl', 'xlrd']
    for engine in engines:
        try:
            df = pd.read_excel(file, engine=engine)
        except ValueError:
            continue
        except Exception as e:
            logger.warning(f"Error reading Excel with {engine} engine: {str(e)}")
            continue
    
    if df == []:
        logger.critical("Failed to read excel from all engined")
        return False
    else:
        return df


def handle_bhav_copy_response(response, date, bhav_copy_logger):
    try:
        content_type = response.headers.get('Content-Type', '')
        bhav_copy_logger.info(f"Received content type: {content_type}")
        bhav_copy_logger.info(f"Response status code: {response.status_code}")
        
        # Check for ZIP file
        if response.content.startswith(b'PK'):
            bhav_copy_logger.info("Detected ZIP file in response")

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                data_file = None
                for file_name in zf.namelist():
                    if (file_name.endswith(('.csv', '.xls', '.xlsx')) or 'sheet' in file_name.lower()) and (not file_name.endswith(('.xml'))):
                        data_file = file_name
                        break
                
                if data_file:
                    with zf.open(data_file) as file:
                        if data_file.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:  # Excel file
                            df = read_excel_file(file, bhav_copy_logger)
                    return df
                
                else:
                    shared_strings = get_shared_strings(zf)

                    worksheet_files = [f for f in  zf.namelist() if f.startswith('xl/worksheets/') and f.endswith('.xml')]

                    if not worksheet_files:
                        raise ValueError("No worksheet XML files found in the archive.")
                    
                    data = extract_data_from_sheet(zf, worksheet_files[0], shared_strings)

                    df = pd.DataFrame(data[1:], columns=data[0])
                    return df

            bhav_copy_logger.warning(f"No suitable data file found in ZIP - {date}")
            return None
        
        # Check for Excel file
        elif b'workbook.xml' in response.content or b'spreadsheetml' in response.content:
            bhav_copy_logger.info("Detected Excel file in response")
            df = pd.read_excel(io.BytesIO(response.content))
            return df
        
        # Check for CSV
        elif 'text/csv' in content_type:
            bhav_copy_logger.info("Detected CSV content in response")
            df = pd.read_csv(io.StringIO(response.text))
            return df
        
        else:
            bhav_copy_logger.critical(f"Content Type not detected: {date}")
            raise Exception(f"Content Type not detected: {date}")
        
    except Exception as e:
        raise Exception(str(e))


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory


def save_to_archive(bhav_copy_archive_path, text, filename):
    bhav_copy_archive_path = create_directory(bhav_copy_archive_path)
    file_path = bhav_copy_archive_path + filename
    text.to_csv(file_path)
    return file_path


def check_date_in_csv(file_path, date_to_check, date_column='Date'):
    """
    Reads a CSV file with a 'Date' column and checks if a specific date is present in that column.

    Parameters:
    file_path (str): The path to the CSV file.
    date_to_check (str): The date to check in the format 'YYYY-MM-DD'.

    Returns:
    bool: True if the date is present, False otherwise.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path, parse_dates=[date_column])
    except FileNotFoundError:
        print("The file was not found.")
        return False
    except ValueError:
        print("Ensure there is a 'Date' column in the CSV.")
        return False
    
    # Check if the date is in the 'Date' column
    return pd.to_datetime(date_to_check) in df[date_column].values


def check_date_in_bhavcopy(data, date_to_check):
    """
    Reads a BhavCopy file with a ' DATE1' column and checks if a specific date is present in that column.

    Parameters:
    data (pd.DataFrame): The dataframe to check.
    date_to_check (str): The date to check in the format 'YYYY-MM-DD'.

    Returns:
    bool: True if the date is present, False otherwise.
    """
    # Read the CSV file
    try:
        if isinstance(data, str):
            data = pd.read_csv(data)

        dates = pd.to_datetime(data[' DATE1'])

    except FileNotFoundError:
        print("The file was not found.")
        return False
    except ValueError:
        print("Ensure there is a 'Date' column in the CSV.")
        return False
    
    # Check if the date is in the 'Date' column
    return pd.to_datetime(date_to_check) in dates.values


def is_valid_date(date_to_check):
    """
    Checks if a given date is valid (i.e., it's a weekday and not a holiday).

    Parameters:
    date_to_check (str): The date to check in the format 'YYYY-MM-DD'.
    holiday_strings (list): A list of holiday dates as strings in the format 'A, %d %B %Y'.

    Returns:
    bool: True if the date is a valid trading date, False otherwise.
    """
    # Convert the date_to_check to a date object
    date = date_to_check.date()
    holiday_dates = get_holidays_for_year(date.year)
    if holiday_dates is None:
        holiday_dates = []
    else:
        holiday_dates = holiday_dates['dates']

    exception_trading_dates = get_exception_trading_dates_to_year(date.year)

    if exception_trading_dates is None:
        exception_trading_dates = []
    else:
        exception_trading_dates = [d.date() for d in list(set(exception_trading_dates['dates']))]
    
    if any(d == date for d in exception_trading_dates):
        return True
    # Check if the date is a weekday and not a holiday
    is_weekday = calendar.day_abbr[date.weekday()] not in ['Sat', 'Sun']
    is_a_holiday =  any(d.date() == date for d in holiday_dates)

    return is_weekday and not is_a_holiday


def generate_html_table(template, data):
    """
    Generates an HTML table from a list of dictionaries, with support for clickable file download links.
    
    Parameters:
        data (list[dict]): A list of dictionaries where each dictionary represents a row. 
                           If a key is 'file_link', its value will be rendered as a clickable link.
    
    Returns:
        str: The generated HTML table as a string.
    """
    if not data:
        return "<p>No data available</p>"
    
    # Extract headers from the keys of the first dictionary
    headers = data[0].keys()
    
    # Start building the HTML table
    html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    
    # Add table headers
    html += "<thead><tr>"
    for header in headers:
        html += f"<th style='padding: 8px; text-align: left;'>{header}</th>"
    html += "</tr></thead>"
    
    # Add table rows
    html += "<tbody>"
    for row in data:
        html += "<tr>"
        for header in headers:
            value = row.get(header, "")
            
            # Handle 'file_link' key by creating a download link
            if header == "file_link" and value:
                value = f"<a href='{value}' target='_blank' style='color: blue; text-decoration: none;'>Click to download</a>"
            
            html += f"<td style='padding: 8px;'>{value}</td>"
        html += "</tr>"
    html += "</tbody>"
    
    html += "</table>"

    if template is not None:
        body = template.get('body'," {{table}} ")
        body = body.replace("{{table}}", html)
    else:
        body = html
        
    return body


def send_email(
    recipient_emails, subject, body, 
    html_body=None, cc_emails=None, bcc_emails=None, 
    attachments=None, use_ssl=True
):
    """
    Sends an email with optional CC, BCC, HTML body, and attachments.
    
    Parameters:
        smtp_server (str): The SMTP server address.
        port (int): The port to connect to the SMTP server.
        sender_email (str): The sender's email address.
        sender_password (str): The sender's email account password.
        recipient_emails (list[str]): List of recipient email addresses.
        subject (str): The subject of the email.
        body (str): The plain text body content of the email.
        html_body (str, optional): HTML version of the email content. Defaults to None.
        cc_emails (list[str], optional): List of CC email addresses. Defaults to None.
        bcc_emails (list[str], optional): List of BCC email addresses. Defaults to None.
        attachments (list[str], optional): List of file paths to attach to the email. Defaults to None.
        use_ssl (bool): Whether to use SSL for the SMTP connection. Defaults to True.
    
    Raises:
        Exception: If the email fails to send for any reason.
    """
    try:
        # Create the email message
        msg = EmailMessage()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ', '.join(recipient_emails)
        msg['Subject'] = subject
        
        if cc_emails:
            msg['Cc'] = ', '.join(cc_emails)
        
        if bcc_emails:
            msg['Bcc'] = ', '.join(bcc_emails)
        
        # Email body: Add plain text and optional HTML content
        if html_body:
            msg.set_content(body)
            msg.add_alternative(f"<p>{html_body}<p>", subtype='html')
        else:
            msg.set_content(body)
        
        # Attach files if provided
        if attachments:
            for file_path in attachments:
                if os.path.exists(file_path):
                    ctype, encoding = mimetypes.guess_type(file_path)
                    ctype = ctype or 'application/octet-stream'
                    maintype, subtype = ctype.split('/', 1)
                    
                    with open(file_path, 'rb') as file:
                        file_data = file.read()
                        file_name = os.path.basename(file_path)
                        msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)
                else:
                    print(f"Warning: File not found: {file_path}")
        
        # Prepare recipient list (includes BCC for sending but not visible in email headers)
        all_recipients = recipient_emails + (cc_emails or []) + (bcc_emails or [])
        
        # Connect to the SMTP server and send the email
        if use_ssl:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg, from_addr=SENDER_EMAIL, to_addrs=all_recipients)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.send_message(msg, from_addr=SENDER_EMAIL, to_addrs=all_recipients)
        
        print("Email sent successfully.")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
    

def handle_file_response(response, date):
    try:
        content_type = response.headers.get('Content-Type', '')
        logger.info(f"Received content type: {content_type}")
        logger.info(f"Response status code: {response.status_code}")
        
        # Check for ZIP file
        if response.content.startswith(b'PK'):
            logger.info("Detected ZIP file in response")

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                data_file = None
                for file_name in zf.namelist():
                    if (file_name.endswith(('.csv', '.xls', '.xlsx')) or 'sheet' in file_name.lower()) and (not file_name.endswith(('.xml'))):
                        data_file = file_name
                        break
                
                if data_file:
                    with zf.open(data_file) as file:
                        if data_file.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:  # Excel file
                            df = read_excel_file(file, logger)
                    return df
                
                else:
                    shared_strings = get_shared_strings(zf)

                    worksheet_files = [f for f in  zf.namelist() if f.startswith('xl/worksheets/') and f.endswith('.xml')]

                    if not worksheet_files:
                        raise ValueError("No worksheet XML files found in the archive.")
                    
                    data = extract_data_from_sheet(zf, worksheet_files[0], shared_strings)

                    df = pd.DataFrame(data[1:], columns=data[0])
                    return df

            logger.warning(f"No suitable data file found in ZIP - {date}")
            return None
        
        # Check for Excel file
        elif b'workbook.xml' in response.content or b'spreadsheetml' in response.content:
            logger.info("Detected Excel file in response")
            df = pd.read_excel(io.BytesIO(response.content))
            return df
        
        # Check for CSV
        elif 'text/csv' in content_type:
            logger.info("Detected CSV content in response")
            df = pd.read_csv(io.StringIO(response.text))
            return df
        
        else:
            logger.critical(f"Content Type not detected: {date}")
            raise Exception(f"Content Type not detected: {date}")
        
    except Exception as e:
        raise Exception(str(e))
    

def get_next_valid_trading_datetime(current_datetime):
    next_date = current_datetime + timedelta(days=1)
    while not is_valid_date(next_date):
        next_date += timedelta(days=1)
    return next_date.replace(hour=16, minute=0, second=0, microsecond=0)

def get_styled_html(df):
    def apply_styles(row):
        def style_cell(value, is_signal=False, is_currency=False):
            """Applies styles to table cells, formatting integer currency values and setting background colors for the 'Signal' column."""
            bg_color = ""
            if is_signal:
                if value == "Bullish":
                    bg_color = "background-color: rgb(106, 168, 79);"
                elif value == "Bearish":
                    bg_color = "background-color: rgb(224, 102, 102);"

            # Format integer currency values with commas
            if is_currency and isinstance(value, int):
                value = f"{value:,}"  # Format as integer with commas (e.g., 1000000 → 1,000,000)

            return f"<td style='border: 1px solid black; padding: 5px; text-align: center; {bg_color}'>{value}</td>"

        # Detect integer columns dynamically
        int_columns = row.index[row.map(lambda x: isinstance(x, int))].tolist()

        # Generate row with styles applied
        row_html = "".join(
            style_cell(value, is_signal=(col == "Signal"), is_currency=(col in int_columns))
            for col, value in zip(row.index, row)
        )

        return f"<tr>{row_html}</tr>"
    # Apply styles to each row
    rows = "".join(df.apply(apply_styles, axis=1))
    
    # Create the complete table with inline styles
    html_table = f"""
    <table style='width: auto; border-collapse: collapse; border: 1px solid black;'>
        <thead>
            <tr>
                {''.join(f"<th style='border: 1px solid black; padding: 5px; text-align: center;'>{col}</th>" for col in df.columns)}
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """
    return html_table


def create_html_for_exposure(index_long_exposure, stock_long_exposure):
    html = f'<tr><td style="vertical-align: middle; text-align: center; padding: 10px;"> FII Net Long Exposure - Index : {index_long_exposure}%</td><td style="vertical-align: middle; text-align: center; padding: 10px;"> FII Net Long Exposure - Stock : {stock_long_exposure}%</td></tr>'
    return html


def get_commentry(index_futures, index_participent_ce, index_participent_pe,  index_long_exposure, pcr, OI_table, week_expry, month_expry):
    filename = 'data.json'

    def save_file(key, value):
        # Check if file exists and load existing data
        if os.path.exists(filename):
            with open(filename, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Update or add the key-value pair
        data[key] = value

        # Save back to file
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        return

    def get_file():
        with open(filename, "r") as file:
            return json.load(file)
    
    commentry = [f"*Derivatives data – Brief commentary {datetime.now().strftime('%d %b%y')}*"]

    loaded_data = get_file()
    
    prev_index_long_exposure = loaded_data['index_long_exposure']
    prev_index_pcr = loaded_data['pcr']
    prev_date = loaded_data['prev_date']


    week_expry_formated = week_expry.strftime("%d-%b")
    max_oi_ce_we = OI_table.loc[OI_table['NIFTY'] == 'Max Call OI', week_expry_formated].iloc[0]
    max_oi_pe_we = OI_table.loc[OI_table['NIFTY'] == 'Max Put OI', week_expry_formated].iloc[0]
    max_oi_addition_ce_we = OI_table.loc[OI_table['NIFTY'] == 'Change in Call OI max', week_expry_formated].iloc[0]
    max_oi_addition_pe_we = OI_table.loc[OI_table['NIFTY'] == 'Change in Put OI max', week_expry_formated].iloc[0]

    month_expry_formated = month_expry.strftime("%d-%b")
    max_oi_ce_me = OI_table.loc[OI_table['NIFTY'] == 'Max Call OI', month_expry_formated].iloc[0]
    max_oi_pe_me = OI_table.loc[OI_table['NIFTY'] == 'Max Put OI', month_expry_formated].iloc[0]
    max_oi_addition_ce_me = OI_table.loc[OI_table['NIFTY'] == 'Change in Call OI max', month_expry_formated].iloc[0]
    max_oi_addition_pe_me = OI_table.loc[OI_table['NIFTY'] == 'Change in Put OI max', month_expry_formated].iloc[0]

    commentry.append(f'•For weekly ({week_expry.strftime("%d %b")}), max OI addition was seen at {int(max_oi_addition_ce_we)} call and {int(max_oi_addition_pe_we)} put. Max OI is at {int(max_oi_ce_we)} call and {int(max_oi_pe_we)} put. <br> \
                     For Monthly expiry ({month_expry.strftime("%d %b")}), max OI addition was seen at {int(max_oi_addition_ce_me)} call and {int(max_oi_addition_pe_me)} put. Max OI is at {int(max_oi_ce_me)} call and {int(max_oi_pe_me)} put.')
    commentry.append(f'• Cumulative Nifty PCR stands at {round(pcr,2)} ({datetime.now().strftime("%d %b%y")}) Vs {prev_index_pcr} ({prev_date})')
    
    FII_sentiment = 'Positive' if index_futures.loc[index_futures['Client Type'] == 'FII','Signal'].values[0] == 'Bullish' else 'Negative'
    commentry.append(f"*Overall FII derivatives data is {FII_sentiment} for {datetime.now().strftime('%A')} ({datetime.now().strftime('%d %b%y')})")

    final_commentry = 'In Index futures, there was '

    fii_long_change = index_futures.loc[index_futures['Client Type'] == 'FII','Future Index Long Change'].values[0]
    fii_short_change = index_futures.loc[index_futures['Client Type'] == 'FII','Future Index Short Change'].values[0]
    if (fii_long_change > 0 and fii_short_change > 0):
        if (fii_long_change > fii_short_change):
            final_commentry += 'net long addition today,'
        elif (fii_long_change < fii_short_change):
            final_commentry += 'net short addition today,'
        else:
            final_commentry += 'equal long and short addition today,'

    elif (fii_long_change < 0 and fii_short_change < 0):
        if (fii_long_change > fii_short_change):
            final_commentry += 'net short unwinding today,'
        elif (fii_long_change < fii_short_change):
            final_commentry += 'net long unwinding today,'
        else:
            final_commentry += 'equal long and short unwinding today,'

    else:
        if (fii_long_change > fii_short_change):
            final_commentry += 'net long addition and short unwinding today,'
        elif (fii_long_change < fii_short_change):
            final_commentry += 'net short addition and long unwinding today,'
        else:
            final_commentry += 'no change in net long and short today,'


    if prev_index_long_exposure > index_long_exposure:
        final_commentry += f" with Net long exposure decreasing to {index_long_exposure}% ({datetime.now().strftime('%d %b%y')}) Vs {prev_index_long_exposure}% ({prev_date}). "
    elif prev_index_long_exposure < index_long_exposure:
        change = round(index_long_exposure - prev_index_long_exposure, 2)
        final_commentry += f" with Net long exposure increasing to {index_long_exposure}% ({datetime.now().strftime('%d %b%y')}) Vs {prev_index_long_exposure}% ({prev_date}). "
    else:
        final_commentry += ' with no change in net long exposure. '
        

    final_commentry += '<br>In index options, there was '


    fii_ce_long_change = index_participent_ce.loc[index_participent_ce['Client Type'] == 'FII', 'Option Index Call Long Change'].values[0]
    fii_ce_short_change = index_participent_ce.loc[index_participent_ce['Client Type'] == 'FII', 'Option Index Call Short Change'].values[0]

    fii_pe_long_change = index_participent_pe.loc[index_participent_ce['Client Type'] == 'FII', 'Option Index Put Long Change'].values[0]
    fii_pe_short_change = index_participent_pe.loc[index_participent_ce['Client Type'] == 'FII', 'Option Index Put Short Change'].values[0]

    if (fii_ce_long_change > 0 and fii_ce_short_change > 0):
        final_commentry += 'net addition in call options'
        if (fii_ce_long_change > fii_ce_short_change):
            final_commentry += ' - long side and '
        elif (fii_ce_long_change < fii_ce_short_change):
            final_commentry += '- short side and '
        else:
            final_commentry += 'equal long and short addition today and '

    elif (fii_ce_long_change < 0 and fii_ce_short_change < 0):
        final_commentry += 'net uwinding in call options'
        if (fii_ce_long_change > fii_ce_short_change):
            final_commentry += '- short side and '
        elif (fii_ce_long_change < fii_ce_short_change):
            final_commentry += '- long side and '
        else:
            final_commentry += 'equal long and short unwinding today and '

    else:
        if (fii_ce_long_change > fii_ce_short_change):
            final_commentry += 'net long addition and short unwinding in call options today and '
        elif (fii_ce_long_change < fii_ce_short_change):
            final_commentry += 'net short addition and long unwinding in call options today and '
        else:
            final_commentry += 'no change in net long and short in call options today and '

    if (fii_pe_long_change > 0 and fii_pe_short_change > 0):
        final_commentry += 'net addition in put options'
        if (fii_pe_long_change > fii_pe_short_change):
            final_commentry += ' - long side,'
        elif (fii_pe_long_change < fii_pe_short_change):
            final_commentry += '- short side'
        else:
            final_commentry += 'equal long and short addition today.'

    elif (fii_pe_long_change < 0 and fii_pe_short_change < 0):
        final_commentry += 'net uwinding in put options'
        if (fii_pe_long_change > fii_pe_short_change):
            final_commentry += '- short side'
        elif (fii_pe_long_change < fii_pe_short_change):
            final_commentry += '- long side'
        else:
            final_commentry += 'equal long and short unwinding today.'

    else:
        if (fii_pe_long_change > fii_pe_short_change):
            final_commentry += 'net long addition and short unwinding in put options today.'
        elif (fii_pe_long_change < fii_pe_short_change):
            final_commentry += 'net short addition and long unwinding in put options today.'
        else:
            final_commentry += 'no change in net long and short in put options today.'

    commentry.append(final_commentry)
    save_file('index_long_exposure', index_long_exposure)
    save_file('pcr', round(pcr,2))
    save_file('prev_date',datetime.now().strftime("%d %b%y") )

    html_commentry = ''
    for comment in commentry:
        html_commentry += f'<b>{comment}</b><br><br>'
    
    # html_commentry = f'<tr><td style="vertical-align: middle; text-align: left; padding: 10px;"> {html_commentry} </td><td style="vertical-align: middle; text-align: center; padding: 10px;"></td></tr>'
    html_commentry = f'<tr><td colspan="2" style="vertical-align: middle; text-align: left; padding: 10px; font-size: 20px; font-weight: bold;"> {html_commentry} </td></tr>'

    return html_commentry


def create_html_table_with_predefined_html(df_dict_list, extra_table_data, commentry):
    # Ensure the list contains exactly 6 dictionaries
    if len(df_dict_list) != 6:
        raise ValueError("The list must contain exactly 6 dictionaries with 'heading' and 'df' keys.")
    
    # Initialize the table HTML string

    table_html = '<table border="1" style="border-collapse: collapse; width: auto; font-family: Arial, sans-serif;">'
    
    table_html += commentry
    # table_html += extra_table_data
    
    # Loop to create the table rows and columns
    for i in range(3):  # 3 rows
        table_html += '<tr>'
        for j in range(2):  # 2 columns
            index = i * 2 + j  # Calculate the dictionary index
            if index < len(df_dict_list):
                heading = df_dict_list[index]['heading']
                df_html = get_styled_html(df_dict_list[index]['df'])  # HTML string of the DataFrame
                
                # Create the table cell with the heading and DataFrame HTML
                # table_html += f'<td><strong style="text-align: center; display: block; width: 100%;">{heading}</strong><br>{df_html}</td>'
                table_html += f'''
                    <td style="vertical-align: middle; text-align: center; padding: 10px;">
                        <div style="background: lightblue;"><strong>{heading}</strong></div><br>{df_html}
                    </td>
                '''
        table_html += '</tr>'
    
    table_html += '</table>'
    
    return table_html


def save_html_as_png(html_string, output_file="output.png"):
    """Captures a full-page screenshot of an HTML string using Chrome DevTools Protocol (CDP)."""
    
    # Wrap HTML properly
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Test</title>
        <style>
            body {{ margin: 0; padding: 0; zoom: 1.0; }}
            ::-webkit-scrollbar {{
                width: 10px;
            }}
            ::-webkit-scrollbar-thumb {{
                background: gray;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        {html_string}
    </body>
    </html>
    """
    
    encoded_html = base64.b64encode(full_html.encode()).decode()
    data_url = f"data:text/html;base64,{encoded_html}"

    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Enables full-page screenshot mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    driver.get(data_url)

    # Give the page time to render
    time.sleep(2)

    # Use Chrome DevTools Protocol for full-page screenshot
    screenshot = driver.execute_cdp_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True, "fromSurface": True})
    
    driver.quit()

    # Save the screenshot
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(screenshot["data"]))

    print(f"✅ PNG saved as {output_file}")

    return


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"Error: {e}")
    
    return True

def run_terminal_command(command):
    """
    Runs a terminal command and returns the result.
    """
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        logger.info(f"Command succeeded: {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}. Error: {e.stderr}")
        return None


def update_chromedriver():
    """
    This function forces an update of the ChromeDriver to the latest version.
    """
    try:
        logger.info("Updating ChromeDriver...")
        # Run the webdriver-manager to install the latest ChromeDriver
        run_terminal_command("webdriver-manager update")
        logger.info("ChromeDriver updated successfully.")
    except Exception as e:
        logger.error(f"Failed to update ChromeDriver: {e}")


def clean_user_temp():
    """
    Clean user-specific temporary directories without needing sudo.
    """
    try:
        # Clean user cache directories
        user_cache_dirs = [
            "~/Library/Caches",
            "~/.cache",
            "/tmp"
        ]
        for directory in user_cache_dirs:
            command = f"rm -rf {directory}/*"
            run_terminal_command(command)
            logger.info(f"Cleaned {directory}")
    except Exception as e:
        logger.error(f"Error cleaning user temp directories: {e}")


def get_display_and_file_names(base_url, endpoint, section_id="cr_deriv_equity_daily_Current", max_retries=3, scroll_pause_time=2):
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--headless')  # Uncomment for headless operation
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')

    driver = None
    results = []
    for attempt in range(max_retries):
        try:
            full_url = base_url + endpoint
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            driver.get(full_url)
            time.sleep(7)  # Allow initial content load
            
            # Progressive scrolling
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time + random.uniform(1, 3))  # Random delay for bot detection evasion
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    driver.execute_script("window.scrollTo(document.body.scrollHeight, 0);")
                    break
                last_height = new_height

            # Wait for the section to load
            section = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, section_id)))

            # Extract display names and file names
            report_elements = section.find_elements(By.CLASS_NAME, "reportsDownload")
            
            if report_elements == []:
                if driver:
                    driver.quit()
                print(f"Element not found on attempt {attempt + 1}. Retrying...")
                continue
            else:
                for element in report_elements:
                    d = {}
                    display_name = element.find_element(By.TAG_NAME, "label").text.strip().replace("\n", "")
                    file_name = element.find_element(By.CLASS_NAME, "reportCardSegment").text.strip()
                    file_link = element.get_attribute("data-link")  # Extract the download link
                    d = {'display_name': display_name,
                        'file_name': file_name,
                        'file_link': file_link}
                    results.append(d)

                return results

        except TimeoutException:
            logger.error(f"Timeout on attempt {attempt + 1}. Retrying...")
            run_terminal_command("pkill -f chromedriver")
        except NoSuchElementException:
            logger.error(f"Element not found on attempt {attempt + 1}. Retrying...")
            try:
                run_terminal_command("rm -rf ~/.caches/google-chrome/Default/Cache/*")
            except Exception as e:
                continue
        except (WebDriverException, RequestException) as e:
            logger.error(f"WebDriverException: {e} - Retrying... (Attempt {attempt+1})")
            run_terminal_command("pkill -f chromedriver")
            update_chromedriver()
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            clean_user_temp()

        finally:
            if driver:
                driver.quit()

        time.sleep(10 + random.uniform(1, 5))  # Randomized delay before retry

    print("All attempts failed")
    return results

def get_weekdays_after_date(year, weekday, after_date):
    """
    Get the first two dates for a given weekday in a specific year that are after a particular date.
    
    year: int, the year to get weekdays for.
    weekday_str: str, the weekday name ('Monday', 'Tuesday', ..., 'Sunday')
    after_date: str, the date string in the format 'YYYY-MM-DD' after which to filter.
    """
    # Convert weekday string to integer using calendar
    if weekday not in calendar.day_name:
        raise ValueError("Invalid weekday string. Use 'Monday', 'Tuesday', ..., 'Sunday'.")
    
    weekday = list(calendar.day_name).index(weekday)
    
    # Start from the first day of the year
    start_date = datetime(year, 1, 1)
    
    # Find the first occurrence of the desired weekday
    days_to_weekday = (weekday - start_date.weekday()) % 7  # Calculate days to desired weekday
    first_weekday = start_date + timedelta(days=days_to_weekday)
    
    # Collect all occurrences of the desired weekday
    weekdays = []
    current_date = first_weekday
    while current_date.year == year:
        if current_date.date() > after_date:  # Filter dates after the given date
            weekdays.append(current_date.date())
        current_date += timedelta(weeks=1)  # Move to the next weekday of the same type
    
    # Return the first two filtered weekdays
    return weekdays[:2]


def get_expiry_dates(ref_date=None, weekly_day='Thursday', monthly_day='Thursday'):
    if ref_date is None:
        ref_date = datetime.today()
    elif isinstance(ref_date, str):
        ref_date = datetime.strptime(ref_date, "%Y-%m-%d")

    ref_date = ref_date.date()
    year, month = ref_date.year, ref_date.month

    # Load holidays and exception trading dates
    holiday_dates = get_holidays_for_year(year)
    exception_trading_dates = get_exception_trading_dates_to_year(year)

    holiday_dates = [] if holiday_dates is None else [d.date() for d in set(holiday_dates['dates'])]
    exception_trading_dates = [] if exception_trading_dates is None else [d.date() for d in set(exception_trading_dates['dates'])]
    
    def is_trading_day(d):
        return (
            d in exception_trading_dates or
            (d.weekday() < 5 and d not in holiday_dates)
        )

    def get_prev_trading_day(start_date):
        while not is_trading_day(start_date):
            start_date -= timedelta(days=1)
        return start_date

    weekly_target = list(calendar.day_name).index(weekly_day)
    monthly_target = list(calendar.day_name).index(monthly_day)
    
    # --- Part 1: Weekly expiry (updated with your provided logic) ---
    next_2_expiries = get_weekdays_after_date(year=ref_date.year, weekday=weekly_day, after_date=ref_date)

    next_weekly_expiry = get_prev_trading_day(next_2_expiries[0])

    if next_weekly_expiry == ref_date:
        next_weekly_expiry = next_2_expiries[-1]

    # --- Part 2: Monthly expiry (your original) ---
    while True:
        last_day = calendar.monthrange(year, month)[1]
        last_expiry = datetime(year, month, last_day).date()

        while last_expiry.weekday() != monthly_target:
            last_expiry -= timedelta(days=1)

        last_expiry = get_prev_trading_day(last_expiry)

        if last_expiry > ref_date:
            break
        else:
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

    return next_weekly_expiry, last_expiry
