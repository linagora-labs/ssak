import os
import re
import time

import requests  # python -m pip install requests
from bs4 import BeautifulSoup  # python -m pip install beautifulsoup4
from selenium import webdriver  # python -m pip install selenium
from selenium.webdriver.common.by import By


def scraping_driver(url=None, headless=True):
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    if url:
        driver.get(url)
    return driver


def dump_current_page(driver, filename=None):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    if filename:
        with open(filename, "w") as f:
            f.write(soup.prettify())
    else:
        print(soup.prettify())


def find_all_links(
    driver,
    contain=None,
    startswith=None,
    endswith=None,
):
    if not contain and startswith:
        contain = startswith
    if not contain and endswith:
        contain = endswith
    if contain:
        matching_links = driver.find_elements(By.XPATH, f'//a[contains(@href, "{contain}")]')

    for link in matching_links:
        href = link.get_attribute("href")
        # WTF?
        href = re.sub("[^/]+/[^/]+?t=../", "", href)
        if (not startswith or href.startswith(startswith)) and (not endswith or href.endswith(endswith)):
            yield href


def download_link_to_file(url, filename, mode="wb", verbose=False, ignore_errors=False, ignore_if_exists=True, sleep_time=0):
    if ignore_if_exists and os.path.isfile(filename):
        # if verbose:
        #     print(f"File {filename} already exists, skipping download")
        return

    # Use the requests library to download the PDF
    if verbose:
        print(f"Downloading {url} into {filename}")
    try:
        response = requests.get(url)
    except Exception as err:
        raise RuntimeError(f"Failed to download {url}") from err

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode) as f:
            f.write(response.content)
        if verbose:
            print("OK")
        if sleep_time:
            time.sleep(sleep_time)
    else:
        if ignore_errors:
            print(f"Failed to download {url} with status code {response.status_code}")
        else:
            raise RuntimeError(f"Failed to download {url}")
