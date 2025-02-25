import streamlit as st
from bs4 import BeautifulSoup
import requests
from scrape import scrape_website, extract_body_content, clean_body_content, split_dom_content


##Create streamlit interface
st.title("StudyBot")


#Choose which website you would like to use or enter your own
website = st.selectbox("Select which site you would like to use to create your study materials:", ["https://www.encyclopedia.com", "https://www.britannica.com/"])
alt_url = st.text_input("Please paste the alternative site you wish to use")
topic = st.text_input("What would you like to study?")


# Run the functions from scrape.py on the material
if st.button("Create Study Materials"):
    st.write("Creating...")
    if website:
        result = scrape_website(website, topic)
    if alt_url:
        result = scrape_website(alt_url, topic)
    body_content = extract_body_content(result)
    cleaned_content = clean_body_content(body_content)
    st.session_state.dom_content = cleaned_content
    
    with st.expander("View DOM Content"):
        st.text_area("DOM Content", cleaned_content, height=300)

with open("study_materials.txt", "w", encoding="utf-8") as f:
    f.write(st.session_state.dom_content)        
