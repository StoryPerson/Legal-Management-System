import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import streamlit as st
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# --- CORRECTED IMPORTS FOR QDRANT & LANGCHAIN ---
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Page config
st.set_page_config(page_title="Legal Case Management & Precedent Search", layout="wide")

# Custom UI Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #F8F9FA; }
    section[data-testid="stSidebar"] { background-color: #2C2C2C; color: white; }
    section[data-testid="stSidebar"] * { color: white !important; }
    div.stButton > button {
        background-color: #FFFFFF !important;
        border-radius: 10px;
        border: 2px solid #C9A227 !important;
        padding: 0.6em 1.2em;
        font-size: 16px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #C9A227 !important;
        color: #FFFFFF !important;
        border: 2px solid #1A2B4C !important;
    }
    h1, h2, h3, h4 { color: #1A2B4C; font-family: 'Georgia', serif; }
    .stMarkdown, p, label { color: #000000 !important; font-family: 'Georgia', serif; }
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Helper Functions
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    sw = set(stopwords.words('english'))
    lemm = WordNetLemmatizer()
    return " ".join([lemm.lemmatize(tok) for tok in tokens if tok not in sw])

@st.cache_resource
def load_pickle(path: str):
    if not os.path.exists(path):
        st.warning(f"Missing file: {path}")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

# Sidebar
st.sidebar.title("⚖️ Legal AI Toolkit")
mode = st.sidebar.radio(
    "Choose a tool:",
    ("Home", "Case Classification", "Case Prioritization", "Legal Precedent Search (RAG)")
)

# Home Screen
if mode == "Home":
    st.title("AI Powered Legal Case Management & Precedent Search")
    st.markdown("""
        ### What this project does
        - **Case Classification**: Automatically classify court cases by category (Civil, Criminal, or Constitutional)
        - **Case Prioritization**: Predict the urgency level of cases (High, Medium, Low)
        - **Legal Precedent Search (RAG)**: Retrieve related case precedents using Cloud Vector Database (Qdrant)
    """)
    st.info("Select a tool from the sidebar to get started.")
    st.stop()

# Case Classification
if mode == "Case Classification":
    st.title("⚖️ Case Classification")
    pipeline_path = "Case Classification/voting_pipeline.pkl"
    label_path = "Case Classification/label_encoder.pkl"

    with st.spinner("Loading classification model..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Paste case text here:", height=300)

    if st.button("Predict Category"):
        if not text_input.strip():
            st.warning("Please enter some case text.")
        elif pipeline is None:
            st.error("Pipeline not loaded.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Case Category: **{pred_label}**")

# Case Prioritization
if mode == "Case Prioritization":
    st.title("⚖️ Case Prioritization")
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("Loading prioritization model..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Paste case text here:", height=300)

    if st.button("Predict Priority"):
        if not text_input.strip():
            st.warning("Please enter some case text.")
        elif pipeline is None:
            st.error("Pipeline not loaded.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Case Priority: **{pred_label}**")

# Legal Precedent Search (RAG) - UPDATED & FIXED
if mode == "Legal Precedent Search (RAG)":
    st.title("📚 Legal Precedent Retrieval Engine (RAG)")
    st.markdown("Ask a question like: *What were previous precedents regarding X?*")

    # Qdrant Configuration
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    # ⚠️ MAKE SURE TO PASTE YOUR KEY HERE
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j5Kv9gmGOtLHLL4RGMJpeqzdVJSrbmsFLlNdbtvmtYs"
    COLLECTION_NAME = "legal_precedents"

    @st.cache_resource
    def load_rag_chain():
        try:
            # 1. Setup Embeddings (Must match what you used for migration)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # 2. Connect to Qdrant Cloud
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # 3. Create Vector Store Wrapper (CORRECTED)
            # We use 'QdrantVectorStore' instead of 'Qdrant'
            # We use 'embedding' (singular) instead of 'embeddings'
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings, 
            )

            # 4. Setup LLM (Groq)
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
            if not api_key:
                return {"error": "Groq API key not found in .env file."}
            
            llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=api_key, temperature=0.2)

            # 5. Create Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """
                You are a legal assistant. Use the following pieces of retrieved context to answer the question.
                If the context does not contain the answer, say that you don't know. 
                Keep the answer professional and concise.

                Context:
                {context}

                Question:
                {input}
                """
            )

            # 6. Build Chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": f"Failed to initialize RAG: {e}"}

    with st.spinner("Connecting to Legal Knowledge Base (Qdrant)..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(rag_resources["error"])
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    
    query = st.text_area("Enter your legal question:", height=150)
    
    if st.button("Search Precedents"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing precedents..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response.get("answer")
                    
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    # Optional: Show Sources
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Source {i+1}:**")
                            # Safely access page_content with fallback
                            content = getattr(doc, 'page_content', "No content available")
                            st.caption(content[:500] + "...")
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Error during retrieval: {e}")