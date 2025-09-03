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

🖥️ How to Run
1. Clone the Repository
git clone https://github.com/devanshupatil04/NeuroRAG.git
cd NeuroRAG

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows

3. Install Dependencies
pip install -r requirements.txt

4. Extract PDF to Text
python -c "from utils import pdf_to_text; pdf_to_text('data/9241544228_eng.pdf', 'data/icd10_text.txt')"

5. Launch Streamlit Web Interface
streamlit run app_streamlit.py


Or use terminal version:

python app.py

⚙️ Requirements
langchain
langchain-community
transformers
torch
accelerate
sentence-transformers
faiss-cpu
pymupdf
tqdm
streamlit

🔍 How It Works

1)Converts ICD-10 PDF into raw text.
2)Chunks the document into 500-token blocks with overlaps.
3)Generates dense semantic embeddings using MiniLM.
4)Stores vector embeddings in FAISS.
5)Uses a local Falcon LLM to answer queries by retrieving top-K relevant chunks.

✨ Advanced Features

-🔄 Dynamic PDF Upload – Upload your own ICD, WHO, or DSM docs.

-🚀 Streamlit GUI – Clean, interactive web interface for medical professionals.

-🧠 MMR Retrieval – Diversified results for better answer grounding.

-💾 Caching – Faster repeat queries, lower compute load.

📌 Limitations

-Runs best on machines with at least 8GB RAM.

-Falcon-RW-1B used for lightweight inference; more complex LLMs may need GPU.

-Currently answers only English queries.

👨‍💻 About Me

I’m Devanshu Patil, an aspiring AI engineer from India with a deep interest in:

GenAI

Medical NLP

Multi-modal LLMs

Vector DBs & RAG systems


📄 License

This project is open-source and free to use under the MIT License.

🙌 Acknowledgements

-HuggingFace 🤗 for open LLMs and embeddings

-LangChain for powerful RAG pipelines

-WHO ICD-10 Documentation
