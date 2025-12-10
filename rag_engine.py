# rag_engine.py
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import config

@st.cache_resource
def build_rag_chain():
    """Initializes the RAG chain components."""
    try:
        groq_key = config.get_groq_api_key()
        qdrant_key = config.get_qdrant_api_key()
        
        if not groq_key or not qdrant_key:
            return {"error": "API Keys missing"}

        # 1. Embeddings & Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = QdrantClient(url=config.QDRANT_URL, api_key=qdrant_key)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding=embeddings, 
        )

        # 2. Reranking (The heavy lifting)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # 3. LLM
        llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_key, temperature=0.1)

        # 4. History Awareness
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
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

        # 5. Final QA Prompt (The Fortified Version)
        qa_system_prompt = """
        You are a Senior Legal Research Assistant. Your mandate is to analyze the provided legal documents and answer the user's question with **forensic precision**.
        
        ### CRITICAL INSTRUCTIONS:
        1. **Zero External Knowledge:** You must answer strictly based *only* on the provided "Context" below.
        2. **No Hallucination:** If the answer is not found in the context, you must state: *"The provided legal documents do not contain sufficient information."*
        3. **Evidence-Based:** Every claim you make must be supported by a specific reference from the text.

        ### REQUIRED OUTPUT FORMAT:
        #### 1. Executive Summary
        (A direct, 2-3 sentence answer.)
        #### 2. Relevant Precedents & Analysis
        (Detailed bullet points analyzing the retrieved text.)
        #### 3. Conclusion
        (A final summary statement.)

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
