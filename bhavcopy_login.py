from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import requests


def get_cookies_from_driver(base_url):
    # Set up Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')  # Required for some environments
    chrome_options.add_argument('--disable-dev-shm-usage')  # Required for some environments
    chrome_options.add_argument('--disable-images')  # Optional: disable images for faster loading

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(base_url)
        cookies = driver.get_cookies()
    except (WebDriverException, TimeoutException) as e:
        print(f"An error occurred: {e}")
        cookies = []
    finally:
        if driver:
            driver.quit()

    return cookies



def login(base_url, extention = ''):
    cookies = get_cookies_from_driver(base_url)

    session = requests.Session()

    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Upgrade-Insecure-Requests': '1',
        'Referer': base_url,
        'Sec-CH-UA': '"Google Chrome"; v="126", "Not=A?Brand"; v="8", "Chromium"; v="126"',
        'Sec-CH-UA-Mobile': '?0',
        'Sec-CH-UA-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1'
    }

    session.headers = headers
    url = base_url + extention

    response = session.get(url)

    if response.status_code == 200:
        return session
    else:
        print(f"Login failed: {response.text}")
        return Exception(f"Login failed: {response}")


def get_data_with_selenium_nse_api(base_url, endpoint, timeout=10):
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run headless for speed
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-images')  # Speeds up loading

    # Mimic user behavior
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36')

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        full_url = f"{base_url}{endpoint}"
        driver.get(full_url)

        # Wait for the content to load. You can modify the condition here if you know what specific element you're waiting for.
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Extract page content
        page_source = driver.page_source

    except TimeoutException:
        print("Timeout occurred while loading the page.")
        return None
    finally:

        if driver:
            driver.quit()
            return page_source
        else:
            return None