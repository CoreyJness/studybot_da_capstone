import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time


#Use Selenium to drive chrome
def scrape_website(website, topic,):
    print("Launching chrome browswer")
    
    chrome_driver_path = "./chromedriver.exe"
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)
    
    driver.get("https://yahoo.com")
    
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//*[@id='uh-sbq']"))
    )
        
    input_element = driver.find_element(By.XPATH, "//*[@id='uh-sbq']")
    input_element.send_keys(topic + website + Keys.ENTER)
   
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, topic))
    )
    
    link = driver.find_element(By.PARTIAL_LINK_TEXT, topic)
    link.click()
    time.sleep(10)
    
    
    driver.quit

    
#Remove the data using BeautifulSoup        
def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    body_content = soup.body
    if body_content:
        return str(body_content)
        
    return ""


#Rewrite content so it is legible
def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
        
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content  = "\n".join(
        line.strip() for line in cleaned_content.split("\n") if line.strip()  
    )
    return cleaned_content


#Create guidance for length and scraping capacity
def split_dom_content(dom_content, max_length=7500):
    return [
        dom_content[i:i + max_length]
        for i in range(0, len(dom_content), max_length)
    ]