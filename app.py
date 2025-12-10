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

# --- IMPORTS ---
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page config
st.set_page_config(
    page_title="Legal AI Toolkit", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SECURE API KEY RETRIEVAL ---
# This looks for the key in Streamlit Secrets (Cloud) or local secrets.toml
def get_groq_api_key():
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    elif os.getenv("GROQ_API_KEY"):
        return os.getenv("GROQ_API_KEY")
    else:
        return None

# --- MODERN LEGAL UI THEME (High Contrast) ---
st.markdown(
    """
    <style>
    /* Global Reset */
    .stApp {
        background-color: #F8FAFC; 
        color: #1E293B; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A; 
    }
    section[data-testid="stSidebar"] h1, p, label, .stRadio div {
        color: #F1F5F9 !important; 
    }

    /* Headers */
    h1, h2, h3 {
        color: #1E3A8A; 
        font-family: 'Playfair Display', serif;
        font-weight: 700;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2563EB; 
        color: white !important;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Input Fields */
    .stTextArea textarea {
        background-color: #FFFFFF;
        color: #0F172A;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
    }
    .stTextArea textarea:focus {
        border-color: #2563EB;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    }

    /* Feature Cards */
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #334155;
    }
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
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/law.png", width=70)
    st.title("Legal AI Toolkit")
    st.markdown("---")
    mode = st.radio(
        "Navigation",
        ("Home", "Case Classification", "Case Prioritization", "Legal Precedent Search"),
        label_visibility="collapsed"
    )
    st.markdown("---")
    
    # Check API Status in Sidebar
    if get_groq_api_key():
        st.success("API Key Loaded")
    else:
        st.error("API Key Missing")

# Home Screen
if mode == "Home":
    st.title("AI-Powered Legal Intelligence")
    st.markdown("Streamlining Case Management & Research")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📂 Classification</h3>
            <p>Automatically categorize legal documents into Civil, Criminal, or Constitutional domains.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Prioritization</h3>
            <p>Predict case urgency (High, Medium, Low) to optimize workflow allocation.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Precedent Search</h3>
            <p>Retrieve relevant case law using semantic search from the Qdrant database.</p>
        </div>
        """, unsafe_allow_html=True)

# Case Classification
if mode == "Case Classification":
    st.title("📂 Case Classification")
    st.markdown("Input the case summary below to determine its legal jurisdiction.")
    
    pipeline_path = "Case Classification/voting_pipeline.pkl"
    label_path = "Case Classification/label_encoder.pkl"

    with st.spinner("Loading AI Models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Case Brief", height=200, placeholder="Example: The plaintiff filed a suit...")

    if st.button("Classify Case"):
        if not text_input.strip():
            st.warning("⚠️ Please provide a case description.")
        elif pipeline is None:
            st.error("❌ Model files missing.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            st.markdown("---")
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-left: 5px solid #2563EB; border-radius: 5px;">
                <h4 style="margin:0; color: #64748B;">PREDICTED CATEGORY</h4>
                <h2 style="margin:5px 0 0 0; color: #1E293B;">{pred_label}</h2>
            </div>
            """, unsafe_allow_html=True)

# Case Prioritization
if mode == "Case Prioritization":
    st.title("⚡ Case Prioritization")
    st.markdown("Assess case urgency to optimize resource allocation.")
    
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("Loading AI Models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Case Brief", height=200, placeholder="Enter case details...")

    if st.button("Assess Priority"):
        if not text_input.strip():
            st.warning("⚠️ Please provide a case description.")
        elif pipeline is None:
            st.error("❌ Model files missing.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            color_map = {"High": "#DC2626", "Medium": "#D97706", "Low": "#059669"}
            color = color_map.get(pred_label, "#2563EB")
            
            st.markdown("---")
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-left: 5px solid {color}; border-radius: 5px;">
                <h4 style="margin:0; color: #64748B;">RECOMMENDED PRIORITY</h4>
                <h2 style="margin:5px 0 0 0; color: {color};">{pred_label} Priority</h2>
            </div>
            """, unsafe_allow_html=True)

# Legal Precedent Search (RAG)
if mode == "Legal Precedent Search":
    st.title("🔍 Precedent Research")
    st.markdown("Retrieve relevant case law and generate legal memos.")

    # Configuration
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j5Kv9gmGOtLHLL4RGMJpeqzdVJSrbmsFLlNdbtvmtYs"
    COLLECTION_NAME = "legal_precedents"

    @st.cache_resource
    def load_rag_chain():
        api_key = get_groq_api_key()
        if not api_key:
            return {"error": "GROQ_API_KEY not found in Secrets."}

        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings, 
            )

            llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key, temperature=0.1)

            prompt = ChatPromptTemplate.from_template(
                """
                You are an expert legal research assistant. 
                Answer the question based EXCLUSIVELY on the provided context.
                
                Format your response as a Legal Memorandum:
                1. **Summary of Findings**: Direct answer.
                2. **Relevant Precedents**: Bullet points citing specific facts/cases from context.
                3. **Conclusion**: Final legal opinion.

                Context:
                {context}

                Question:
                {input}
                """
            )

            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": str(e)}

    with st.spinner("Connecting to Legal Knowledge Base..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(f"❌ Connection Error: {rag_resources['error']}")
        st.info("Please add `GROQ_API_KEY` to your Streamlit Secrets.")
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    
    query = st.text_area("Legal Query", height=120, placeholder="E.g., What is the precedent for granting bail in non-bailable offenses regarding medical grounds?")
    
    if st.button("Search Database"):
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("Analyzing precedents and drafting memo..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response.get("answer")
                    
                    st.markdown("### 📝 Legal Memorandum")
                    st.markdown(f"""
                    <div style="background-color: white; padding: 30px; border-radius: 5px; border: 1px solid #E2E8F0; font-family: 'Times New Roman', serif; color: #000;">
                        {answer.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>### 📚 Source Citations", unsafe_allow_html=True)
                    for i, doc in enumerate(response["context"]):
                        content = getattr(doc, 'page_content', "N/A")
                        preview = content[:300].replace("\n", " ") + "..."
                        with st.expander(f"Citation {i+1}"):
                            st.markdown(f"**Extract:** *{preview}*")
                            st.caption("Source: Qdrant Cloud Database")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
