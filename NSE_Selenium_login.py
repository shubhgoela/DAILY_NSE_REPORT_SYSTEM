from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import undetected_chromedriver as uc
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_with_selenium_nse_api(base_url, api_endpoint, use_undetected=False, max_retries=3):
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-http2')
    chrome_options.add_argument('--headless')  # Commented out to allow visual inspection if needed
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')

    driver = None
    for attempt in range(max_retries):
        try:
            if use_undetected:
                driver = uc.Chrome(options=chrome_options)
            else:
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            
            driver.get(base_url)

            time.sleep(10)

             # Scroll to trigger any lazy-loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 0);")

            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Navigate to the API URL
            api_url = base_url+api_endpoint
            driver.get(api_url)
            
            # Wait for the content to load
            page_source = driver.page_source
            # print(page_source)
            wait = WebDriverWait(driver, 60)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 0);")
            content = wait.until(EC.presence_of_element_located((By.TAG_NAME, "pre")))
            # content = driver.execute_script("return document.querySelector('pre').textContent;")
            # Try to parse the content as JSON
            try:
                data = json.loads(content.text)
                if driver:
                    driver.quit()
                return data  # Return formatted JSON string
            except json.JSONDecodeError:
                logger.warning("Content is not valid JSON. Returning raw text.")
                return content.text
        
        except TimeoutException:
            logger.warning(f"Timeout on attempt {attempt + 1}. Retrying...")
        except NoSuchElementException:
            logger.warning(f"Element not found on attempt {attempt + 1}. Retrying...")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        finally:
            if driver:
                driver.quit()
        
        # Wait before retrying
        time.sleep(10)
    
    logger.error("All attempts failed")
    return None
