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

# --- STANDARD LANGCHAIN IMPORTS (Fixing the 'classic' error) ---
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- RERANKING IMPORTS ---
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Page config
st.set_page_config(page_title="Legal AI Toolkit", layout="wide", page_icon="⚖️")

# --- 1. SETUP RESOURCES (Cached) ---
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

setup_nltk()
STOPWORDS_SET = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- 2. ROBUST API KEY RETRIEVAL ---
def get_secret(key_name):
    """Checks Streamlit Secrets first, then Local Environment."""
    try:
        return st.secrets[key_name]
    except (FileNotFoundError, KeyError):
        return os.getenv(key_name)

# --- 3. UI STYLING ---
st.markdown(
    """
    <style>
    .stApp { background-color: #F8F9FA; color: black; }
    section[data-testid="stSidebar"] { background-color: #1A1A1A; }
    section[data-testid="stSidebar"] * { color: white !important; }
    
    /* Buttons */
    div.stButton > button {
        background-color: #FFFFFF;
        border: 2px solid #C9A227;
        color: #1A2B4C !important;
        font-weight: 700;
    }
    
    /* Red Clear Button */
    div.stButton > button[kind="primary"] {
        background-color: #DC2626 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        background-color: white;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    return " ".join([LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in STOPWORDS_SET])

@st.cache_data
def load_pickle(path: str):
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except Exception: return None

# --- SIDEBAR ---
st.sidebar.title("⚖️ Legal AI Toolkit")
mode = st.sidebar.radio(
    "Navigation",
    ("Home", "Case Classification", "Case Prioritization", "Legal Assistant (Chat)")
)

st.sidebar.markdown("<br>" * 5, unsafe_allow_html=True)
if st.sidebar.button("🗑️ Clear Chat History", type="primary", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

# --- MAIN LOGIC ---

if mode == "Home":
    st.title("AI Powered Legal Intelligence")
    st.info("Select a module from the sidebar.")

elif mode == "Case Classification":
    st.title("⚖️ Case Classification")
    pipeline = load_pickle("Case Cateogarization/voting_pipeline.pkl")
    encoder = load_pickle("Case Cateogarization/label_encoder.pkl")
    
    txt = st.text_area("Case Brief:", height=200)
    if st.button("Classify"):
        if pipeline and txt:
            pred = pipeline.predict([preprocess_text(txt)])
            lbl = encoder.inverse_transform(pred)[0]
            st.success(f"Category: **{lbl}**")
        else:
            st.error("Model missing or empty text.")

elif mode == "Case Prioritization":
    st.title("⚡ Case Prioritization")
    pipeline = load_pickle("Case Prioritization/stacking_pipeline.pkl")
    encoder = load_pickle("Case Prioritization/label_encoder.pkl")
    
    txt = st.text_area("Case Brief:", height=200)
    if st.button("Prioritize"):
        if pipeline and txt:
            pred = pipeline.predict([preprocess_text(txt)])
            lbl = encoder.inverse_transform(pred)[0]
            st.success(f"Priority: **{lbl}**")

elif mode == "Legal Assistant (Chat)":
    st.title("💬 Legal Research Assistant")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- CONFIGURATION ---
    # 1. Get Keys (Stop if missing)
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    # Try fetching from secrets first, fallback to hardcoded (if safe) or env
    QDRANT_API_KEY = get_secret("QDRANT_API_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j5Kv9gmGOtLHLL4RGMJpeqzdVJSrbmsFLlNdbtvmtYs"
    GROQ_API_KEY = get_secret("GROQ_API_KEY")
    COLLECTION_NAME = "legal_precedents"

    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY is missing. Please add it to Streamlit Secrets.")
        st.stop()

    @st.cache_resource
    def load_rag_chain():
        try:
            # 1. Connection Check
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            # Test connection immediately
            try:
                client.get_collections()
            except Exception as e:
                return {"error": f"🚨 Qdrant Connection Failed: {e}"}

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings, 
            )

            # 2. Reranker
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
            model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = CrossEncoderReranker(model=model, top_n=5)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )

            # 3. LLM & Chain
            llm = ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.1)

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, contextualize_q_prompt)

            qa_system_prompt = """
            You are a Senior Legal Research Assistant.
            STRICT INSTRUCTIONS:
            1. Answer ONLY using the provided Context.
            2. Format as a Legal Memo (Executive Summary -> Analysis -> Conclusion).
            3. If the answer is not in the context, say: "The provided documents do not contain sufficient information."

            Context:
            {context}
            """
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            return {"chain": rag_chain}

        except Exception as e:
            return {"error": f"Init Failed: {e}"}

    # Load Brain
    with st.spinner("Initializing AI Knowledge Base..."):
        data = load_rag_chain()
    
    if "error" in data:
        st.error(data["error"])
        st.stop()
    
    chain = data["chain"]

    # Chat Interface
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    if prompt := st.chat_input("Ask a legal question..."):
        st.chat_message("user").write(prompt)
        with st.spinner("Researching..."):
            try:
                response = chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = response['answer']
                
                with st.chat_message("assistant"):
                    st.write(answer)
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Source {i+1}**")
                            st.caption(doc.page_content[:300] + "...")
                            
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"Search Error: {e}")

