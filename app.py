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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name 


    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What Insights are you seeking from  the video", 
        placeholder="Ask Anything about the video content. The AI Agent will analyze the content for you", 
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query: 
            st.warning("Please enter a question or insight to analyze the video")
        else: 
            try: 
                with st.spinner("Processing video and gathering insights"): 
                    # upload to google gen ai 
                    video = upload_file(video_path)
                    while video.state.name == "PROCESSING":
                        time.sleep(1)
                        video = get_file(video.name)

                    
                     # Prompt generation for analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    response = video_agent.run(analysis_prompt, videos=[video])
                
                st.subheader("Anaylsis Results")
                st.markdown(response.content)


            except Exception as e: 
                st.error(f"An error occurred during analysis: {e}")

            finally: 
                Path(video_path).unlink(missing_ok=True)

else: 
    st.info("Upload a video file to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)