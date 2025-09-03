

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
        text = rag.load_text()
        docs = rag.split_chunks(text)
        rag.build_vectorstore(docs)
        st.session_state["vectorstore_built"] = True
        st.success("✅ Vector index built successfully!")

# Query input
query = st.text_input("❓ Ask your mental health-related question",
                      placeholder="e.g., What is the code for Recurrent depressive disorder in remission?")

# Initialize cache for responses
if "cache" not in st.session_state:
    st.session_state["cache"] = {}

# Handle query
if query:
    if query in st.session_state["cache"]:
        result = st.session_state["cache"][query]
        st.info("🔁 Using cached response")
    else:
        with st.spinner("💬 Thinking..."):
            qa_chain = rag.get_qa_chain()
            result = qa_chain.run(query)
            st.session_state["cache"][query] = result

    st.markdown("### 💡 Answer")
    st.success(result)

# Footer
st.markdown("---")
st.caption("Built by Devanshu Patil for Wundrsight Internship Assignment · Powered by LangChain + Falcon + FAISS")
