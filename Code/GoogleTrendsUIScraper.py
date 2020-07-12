from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

import Keywords
keywords = Keywords.keywords

GEO = 'US'
webdriver_path = 'C:/Users/Piyush/Downloads/chromedriver_win32_(1)/chromedriver'

def enable_headless_download(browser, download_path):
    # Add missing support for chrome "send_command" to selenium webdriver
    browser.command_executor._commands["send_command"] = \
        ("POST", '/session/$sessionId/chromium/send_command')

    params = {'cmd': 'Page.setDownloadBehavior',
              'params': {'behavior': 'allow', 'downloadPath': download_path}}
    browser.execute("send_command", params)

# Add arguments telling Selenium to not actually open a window


for keyword, keyword_list in keywords.items():

    keyword = keyword.replace('_', ' ')

    date_ranges = [
        ('2018-01-01', '2018-06-30'),
        ('2018-07-01', '2018-12-31'),
        ('2019-01-01', '2019-05-14')
    ]

    for date_range in date_ranges:
        download_path = 'C:\\Users\\Piyush\\Google Drive\\MScA\\time_series\\project\\data_{0}\\{1}\\{2}'.format(GEO, keyword, date_range[0])

        #download_path = 'C:\\Users\\Piyush\\Downloads\\' + keyword
        chrome_options = Options()

        download_prefs = {
            'download.default_directory': download_path,
            'download.prompt_for_download': False,
            'profile.default_content_settings.popups': 0
        }

        chrome_options.add_experimental_option('prefs', download_prefs)
        #chrome_options.add_argument('--headless')
        chrome_options.add_argument('--window-size=1920x1080')

        url = 'https://trends.google.com/trends/explore?date={0}%20{1}&geo={2}&q={3}'.format(date_range[0], date_range[1], GEO, keyword)

        # Start up browser
        print("Starting browser")
        browser = webdriver.Chrome(executable_path=webdriver_path,chrome_options=chrome_options)
        browser.get(url)
        enable_headless_download(browser, download_path)

        # Load webpage
        print("Loading page")
        browser.get(url)
        time.sleep(5)
        button = browser.find_element_by_css_selector('.widget-actions-item.export')
        button.click()
        time.sleep(5)
        browser.quit()
        print("Closing browser")
