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