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
        .stApp { background-color: #F8F9FA; }
        section[data-testid="stSidebar"] { background-color: #2C2C2C; }
        section[data-testid="stSidebar"] * { color: white !important; }
        div.stButton > button { background-color: #FFFFFF; border: 2px solid #C9A227; color: #1A2B4C; font-weight: 700; }
        div.stButton > button[kind="primary"] { background-color: #DC2626 !important; color: white !important; border: none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )
