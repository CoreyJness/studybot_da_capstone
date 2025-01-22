import streamlit as st
from bs4 import BeautifulSoup
import requests
from scrape import scrape_website, extract_body_content, clean_body_content, split_dom_content


st.title("Corey's Web Scraper")
url = st.text_input("Enter the URL of the website you want to scrape:")

if st.button("Scrape Site"):
    st.write("Scraping site...")
    result = scrape_website(url)
    body_content = extract_body_content(result)
    cleaned_content = clean_body_content(body_content)
    st.session_state.dom_content = cleaned_content
    
    with st.expander("View DOM Content"):
        st.text_area("DOM Content", cleaned_content, height=300)

with open("scraper_data.txt", "w", encoding="utf-8") as f:
    f.write(st.session_state.dom_content)        
