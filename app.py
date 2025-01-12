import streamlit as st 
from phi.agent import Agent 
from phi.model.google import Gemini 
from phi.tools.duckduckgo import DuckDuckGo 
from google.generativeai import upload_file, get_file 
import google.generativeai as genai 
import time 
from pathlib import Path 
import tempfile
from dotenv import load_dotenv
import os 

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_AI_API")
if GOOGLE_API_KEY: 
    genai.configure(api_key=GOOGLE_API_KEY)



st.set_page_config(
    page_title="MultiModel AI Agent", 
    page_icon="", 
    layout="wide"
)


st.title("PhiData Multimodal Video AI Agent")
st.header("Powered by Gemini")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer", 
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()], 
        markdown=True
    )

# get agent 
video_agent = initialize_agent()

# get video file via st file upload 
video_file = st.file_uploader(
    "Upload a video file", type=["mp4", "mov", "avi"], help="Upload a video file for AI analysis"
)

if video_file: 
    pass 