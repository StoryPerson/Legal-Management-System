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

# --- IMPORTS ---
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

# Load environment variables (Local)
load_dotenv(dotenv_path=".env", override=True)

# Page config
st.set_page_config(page_title="Legal AI Toolkit", layout="wide", page_icon="⚖️")

# --- OPTIMIZATION: CACHED RESOURCE LOADING ---
@st.cache_resource
def setup_nltk():
    """Download NLTK resources once and cache them."""
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

setup_nltk()

# Define globally to avoid re-initialization overhead
STOPWORDS_SET = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- OPTIMIZATION: ROBUST KEY RETRIEVAL ---
def get_groq_api_key():
    """Checks Secrets (Cloud) first, then Environment (Local)."""
    # 1. Streamlit Secrets (Cloud)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, AttributeError):
        pass
    
    # 2. Local Environment
    if os.getenv("GROQ_API_KEY"):
        return os.getenv("GROQ_API_KEY")
    
    return None

# Custom UI Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #F8F9FA; }
    section[data-testid="stSidebar"] { background-color: #2C2C2C; color: white; }
    section[data-testid="stSidebar"] * { color: white !important; }
    
    /* Standard Buttons */
    div.stButton > button {
        background-color: #FFFFFF;
        border-radius: 10px;
        border: 2px solid #C9A227;
        color: #1A2B4C;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #C9A227;
        color: #FFFFFF !important;
        border: 2px solid #1A2B4C;
    }
    
    /* 🔴 RED CLEAR CHAT BUTTON */
    div.stButton > button[kind="primary"] {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    h1, h2, h3, h4 { color: #1A2B4C; font-family: 'Georgia', serif; }
    .stMarkdown, p, label { color: #000000 !important; font-family: 'Georgia', serif; }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def preprocess_text(text: str) -> str:
    """Optimized preprocessing using global constants."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    # Use global LEMMATIZER and STOPWORDS_SET for speed
    return " ".join([LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in STOPWORDS_SET])

@st.cache_data
def load_pickle(path: str):
    """Loads and caches pickle files."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# Sidebar
st.sidebar.title("⚖️ Legal AI Toolkit")
mode = st.sidebar.radio(
    "Choose a tool:",
    ("Home", "Case Classification", "Case Prioritization", "Legal Assistant (Chat)")
)

# --- SIDEBAR BOTTOM SECTION ---
st.sidebar.markdown("<br>" * 10, unsafe_allow_html=True) 

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History", type="primary", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

# --- MAIN PAGE LOGIC ---

# 1. Home Screen
if mode == "Home":
    st.title("AI Powered Legal Case Management")
    st.markdown("""
        ### Modules:
        - **Case Classification**: Categorize court cases (Civil, Criminal, Constitutional).
        - **Case Prioritization**: Predict urgency (High, Medium, Low).
        - **Legal Assistant**: RAG-based Chat with Memory & Re-ranking.
    """)
    st.info("Select a tool from the sidebar to get started.")

# 2. Case Classification
elif mode == "Case Classification":
    st.title("⚖️ Case Classification")
    pipeline_path = "Case Cateogarization/voting_pipeline.pkl"
    label_path = "Case Cateogarization/label_encoder.pkl"

    with st.spinner("Loading models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Paste case text here:", height=300)

    if st.button("Predict Category"):
        if not text_input.strip():
            st.warning("Please enter text.")
        elif pipeline is None:
            st.error("Models not found.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Category: **{pred_label}**")

# 3. Case Prioritization
elif mode == "Case Prioritization":
    st.title("⚖️ Case Prioritization")
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("Loading models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Paste case text here:", height=300)

    if st.button("Predict Priority"):
        if not text_input.strip():
            st.warning("Please enter text.")
        elif pipeline is None:
            st.error("Models not found.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Priority: **{pred_label}**")

# 4. Legal Assistant (Chat)
elif mode == "Legal Assistant (Chat)":
    st.title("💬 Legal Research Assistant")
    st.markdown("Ask questions about precedents. The AI uses **Re-ranking** and **Memory**.")

    # Initialize History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Config
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j5Kv9gmGOtLHLL4RGMJpeqzdVJSrbmsFLlNdbtvmtYs"
    COLLECTION_NAME = "legal_precedents"

    @st.cache_resource
    def load_rag_chain():
        """Initializes the RAG chain with caching to prevent reload lag."""
        try:
            api_key = get_groq_api_key()
            if not api_key:
                return {"error": "GROQ_API_KEY not found in Secrets or .env"}

            # 1. Embeddings & Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings, 
            )

            # 2. Reranker (Top 20 -> Top 5)
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
            model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = CrossEncoderReranker(model=model, top_n=5)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )

            # 3. LLM (Using Llama 3 for speed on Groq)
            llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key, temperature=0.1)

            # 4. Memory: Reformulate Question
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(
                llm, compression_retriever, contextualize_q_prompt
            )

            # 5. Answer Generation
# --- MEMORY: Step 2 - Answer Question (Fortified) ---
            qa_system_prompt = """
            You are a Senior Legal Research Assistant. Your mandate is to analyze the provided legal documents and answer the user's question with **forensic precision**.
            
            ### CRITICAL INSTRUCTIONS:
            1. **Zero External Knowledge:** You must answer strictly based *only* on the provided "Context" below. Do not use outside legal knowledge, general principles, or laws not explicitly present in the text.
            2. **No Hallucination:** If the answer is not found in the context, you must state: *"The provided legal documents do not contain sufficient information to answer this specific query."* Do not attempt to guess or fabricate an answer.
            3. **Evidence-Based:** Every claim you make must be supported by a specific reference from the text (e.g., "According to Case X...").
            4. **Tone:** Maintain a formal, objective, and non-advisory tone (avoid saying "You should").

            ### REQUIRED OUTPUT FORMAT:
            
            #### 1. Executive Summary
            (A direct, 2-3 sentence answer to the core legal question.)

            #### 2. Relevant Precedents & Analysis
            (Detailed bullet points analyzing the retrieved text.)
            * **[Case/Section Name]:** [Key holding or fact relevant to the question]
            * **[Case/Section Name]:** [Key holding or fact relevant to the question]

            #### 3. Conclusion
            (A final summary statement on the legal position based solely on the provided context.)

            ### CONTEXT:
            {context}
            """
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": f"Init Failed: {e}"}

    # Load Chain
    with st.spinner("Connecting to Knowledge Base..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(rag_resources["error"])
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    
    # Display Chat
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Input
    if user_query := st.chat_input("Ask a follow-up question..."):
        st.chat_message("user").markdown(user_query)
        
        with st.spinner("Researching..."):
            try:
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response['answer']
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("📚 View Sources"):
                         for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Source {i+1}:**")
                            content = getattr(doc, 'page_content', "No content available")
                            st.caption(content[:300] + "...")
                            st.divider()

                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=answer))
                            
            except Exception as e:
                st.error(f"Error: {e}")
