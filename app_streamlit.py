

import streamlit as st
from rag_pipeline import RAGPipeline
from utils import pdf_to_text
import os


st.set_page_config(page_title="🧠 Mental Health QA", layout="centered")
st.title("🧠 ICD-10 Mental Health Question Answering")

# Upload a new ICD-10 PDF
uploaded_file = st.file_uploader("📄 Upload a new ICD-10 or mental health classification PDF", type="pdf")


pdf_path = "data/uploaded.pdf"  #set pdf path here
txt_path = "data/icd10_text.txt"

# If user uploads a new PDF
if uploaded_file:
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_to_text(pdf_path, txt_path)
    st.success("✅ PDF uploaded and converted to text!")

# Initialize RAG pipeline with latest text
rag = RAGPipeline(doc_path=txt_path)

# First-time setup: Load text, split, and build index
if "vectorstore_built" not in st.session_state:
    with st.spinner("🔍 Preparing the document..."):
        # Check if index already exists
        if os.path.exists("faiss_index/index.faiss"):
            try:
                st.info("📂 Loading existing vector index...")
                rag.load_vectorstore()
                st.session_state["vectorstore_built"] = True
                st.success("✅ Vector index loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Could not load existing index: {e}. Rebuilding...")
                text = rag.load_text()
                docs = rag.split_chunks(text)
                rag.build_vectorstore(docs)
                st.session_state["vectorstore_built"] = True
                st.success("✅ Vector index built successfully!")
        else:
            text = rag.load_text()
            docs = rag.split_chunks(text)
            rag.build_vectorstore(docs)
            st.session_state["vectorstore_built"] = True
            st.success("✅ Vector index built successfully!")

# Query input
query = st.text_input("❓ Ask your mental health-related question",
                      placeholder="e.g., What is the code for Recurrent depressive disorder in remission?")

# Mode selection
use_simple_mode = st.checkbox("⚡ Fast mode (instant, retrieval-only)", value=True, 
                               help="Fast mode shows relevant text chunks instantly. AI mode generates answers but takes 30-60 seconds on first query.")

# Initialize cache for responses
if "cache" not in st.session_state:
    st.session_state["cache"] = {}

# Handle query
if query:
    cache_key = f"{query}_{'simple' if use_simple_mode else 'ai'}"
    
    if cache_key in st.session_state["cache"]:
        result = st.session_state["cache"][cache_key]
        st.info("🔁 Using cached response")
    else:
        with st.spinner("💬 Searching..." if use_simple_mode else "💬 Thinking (this may take 30-60 seconds on first query)..."):
            if use_simple_mode:
                # Fast retrieval-only mode
                result = rag.simple_search(query, k=3)
            else:
                # Full AI mode with LLM
                qa_chain = rag.get_qa_chain()
                result = qa_chain.invoke(query)
                # Extract result text
                if isinstance(result, dict):
                    result = result.get('result', str(result))
            st.session_state["cache"][cache_key] = result

    st.markdown("### 💡 Answer")
    st.success(result)

# Footer
st.markdown("---")
st.caption("Built by Gaurav Patil · Powered by LangChain + FAISS")
