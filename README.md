# NeuroRAG
# 🧠 NeuroRAG – Intelligent Mental Health Code & Diagnosis Assistant

**Created by [Devanshu Patil]((https://www.linkedin.com/in/devanshu-patil-8618b3222/))**  
📍 Pune, India | 🎓 B.Tech Computer Engineering | 



## 🚀 Project Overview

**NeuroRAG** is a lightweight, Retrieval-Augmented Generation (RAG)-based AI assistant designed to answer natural language questions about mental health diagnoses and their ICD-10 classification codes.

The system leverages open-source LLMs, vector databases, and semantic search to intelligently interpret queries and retrieve structured information from the official **ICD-10 Mental and Behavioural Disorders** dataset (PDF document).



## 🎯 Key Features

✅ Question Answering over ICD-10 using natural language  
✅ Intelligent retrieval with FAISS + sentence-transformer embeddings  
✅ Local, open-source LLMs (e.g., Falcon-RW-1B)  
✅ Fully offline & privacy-friendly (no OpenAI or cloud APIs)  
✅ Fast response via cached answers  
✅ Streamlit web app + CLI interface  
✅ Upload your own ICD or medical PDFs dynamically  
✅ Relevance reranking with MMR for better semantic results  



## 🛠️ Tech Stack

| Component              | Technology Used                             |
|------------------------|----------------------------------------------|
| LLM                    | Falcon-RW-1B via HuggingFace Transformers   |
| Vector DB              | FAISS                                       |
| Embeddings             | `all-MiniLM-L6-v2` via Sentence-Transformers|
| RAG Framework          | LangChain (v0.2+) + LangChain-Community     |
| Frontend (Web)         | Streamlit                                   |
| Backend & Pipeline     | Python 3.10+                                |
| PDF to Text Conversion | PyMuPDF (fitz)                              |



## 🧠 Use Case Examples


❓ What are the diagnostic criteria for Obsessive-Compulsive Disorder?
💡 Answer: Obsessions and/or compulsions present for at least two weeks and cause distress or impairment in functioning.

❓ What is the ICD-10 code for Recurrent depressive disorder in remission?
💡 Answer: F33.4 – Recurrent depressive disorder, currently in remission.
