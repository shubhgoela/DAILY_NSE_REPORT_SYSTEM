import os
from dotenv import find_dotenv, load_dotenv
from datetime import datetime
from pymongo import MongoClient
import certifi


dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path, override=True)

DB_URI = os.getenv('DB_URI')
DB_NAME = os.getenv('DB_NAME')

print(f"Connecting to MongoDB at {DB_URI} for database {DB_NAME}")
client = MongoClient(DB_URI, tlsCAFile=certifi.where())


db = client[DB_NAME]


def save_portfolio(collection_name, portfolio):
    data = db[collection_name].insert_one(portfolio)
    return data.acknowledged


def save_all_corp_action(action):
    data = db['collection_corp_action'].insert_one(action)
    return data.acknowledged


def save_corp_action(action):
    data = db['corp_action_adjusted'].insert_one(action)
    return data.acknowledged


def get_index_constituents(index):
    data = db['collection_index_constituents'].find_one({'index': index})
    return data['scrip_list']


def update_index_constituents(index, update):
    update['updated_on'] = datetime.now()
    db['collection_index_constituents'].find_one_and_update(
            {"index": index},
            {"$set": update}
        )
    return True


def get_adjusted_corp_actions(date):
    data = db['corp_action_adjusted'].find_one({'date': date})
    return data


def add_holiday_to_year(date):
    if isinstance(date, list):
        data = db['collection_holidays'].find_one_and_update({"year": date[0].year},  
                                                             {"$push": {"dates": {"$each": date}},
                                                             "$set": {'updated_on': datetime.now()}},
                                                             upsert=True,
                                                             return_document=True
                                                             )
    else:
        data = db['collection_holidays'].find_one_and_update({"year": date.year},  
                                                            {"$push": {"dates": date}, 
                                                            "$set": {'updated_on': datetime.now()}},
                                                            upsert=True,
                                                            return_document=True
                                                            )
    return data


def get_holidays_for_year(year):
    data = db['collection_holidays'].find_one({'year': year})
    return data


def get_mail_template(template_name):
    data = db['collection_mail_templates'].find_one({'template_name': template_name})
    return data


def add_exception_trading_dates_to_year(date):
    if isinstance(date, list):
        data = db['collection_exception_trading_days'].find_one_and_update({"year": date[0].year},  
                                                             {"$push": {"dates": {"$each": date}},
                                                             "$set": {'updated_on': datetime.now()}},
                                                             upsert=True,
                                                             return_document=True
                                                             )
    else:
        data = db['collection_exception_trading_days'].find_one_and_update({"year": date.year},  
                                                            {"$push": {"dates": date}, 
                                                            "$set": {'updated_on': datetime.now()}},
                                                            upsert=True,
                                                            return_document=True
                                                            )
    return data


def get_exception_trading_dates_to_year(year):
    data = db['collection_exception_trading_days'].find_one({'year': year})
    return data

def get_expry_days(exchange, index, expiry_type):
    data = db['collection_expry_days'].find_one({'exchange': exchange, 'instrument': index, 'expiry_type': expiry_type})
    return data['day']