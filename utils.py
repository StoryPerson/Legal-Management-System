# utils.py
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

setup_nltk()
STOPWORDS_SET = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    return " ".join([LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in STOPWORDS_SET])

@st.cache_data
def load_pickle(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* --- MAIN BACKGROUND --- */
        .stApp { background-color: #F8F9FA; color: #000000; }
        
        /* --- SIDEBAR STYLING --- */
        section[data-testid="stSidebar"] { 
            background-color: #1A1A1A; /* Dark Sidebar */
        }
        
        /* FORCE SIDEBAR TEXT TO BE WHITE */
        section[data-testid="stSidebar"] .stMarkdown, 
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] span {
            color: #FFFFFF !important;
        }

        /* --- HOME PAGE CARDS --- */
        .feature-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Force text inside cards to be dark */
        .feature-card h3 { color: #1E3A8A !important; margin-top: 0; }
        .feature-card p { color: #4B5563 !important; }

        /* --- BUTTONS --- */
        div.stButton > button { 
            background-color: #FFFFFF; 
            border: 2px solid #C9A227; 
            color: #1A2B4C !important; 
            font-weight: 700; 
        }
        
        /* Red Clear Chat Button */
        div.stButton > button[kind="primary"] { 
            background-color: #DC2626 !important; 
            color: white !important; 
            border: none !important; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )
