from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import os

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome('C:\\Users\\cedri\\Downloads\\chromedriver.exe', options=chrome_options)

driver.get("https://www.gesetze-im-internet.de/aktuell.html")

az_links = driver.execute_script("return Array.from(document.querySelectorAll('#paddingLR12 > p > a')).map(o => o.getAttribute('href'))")

full_text = ''

for az_link in az_links:
    driver.get("https://www.gesetze-im-internet.de/" + az_link[1:])
    sub_links = driver.execute_script("return Array.from(document.querySelectorAll('#paddingLR12 > p > a')).map(o => o.getAttribute('href')).filter(o => o.endsWith('.html'))")

    for sub_link in sub_links:
        current_url = "https://www.gesetze-im-internet.de" + sub_link[1:]
        driver.get(current_url)
        html_url = driver.execute_script("return document.querySelector('#content_12793 h2.headline a:first-child').getAttribute('href')")
        driver.get('/'.join(current_url.split('/')[:-1]) + '/' + html_url)
        full_text += driver.execute_script("return Array.from(document.querySelectorAll('#paddingLR12 .jnnorm:not(:first-child)')).map(o => o.textContent.replace('Nichtamtliches Inhaltsverzeichnis', '')).join(' ')") + ' '


filepath = os.path.join('datasets', 'law', 'gesetzte_full.txt')
with open(filepath, 'w', encoding='utf-8') as file:
    file.write(full_text)