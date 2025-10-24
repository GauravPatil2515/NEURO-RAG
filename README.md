# NeuroRAG
# 🧠 NeuroRAG – Intelligent Mental Health Code & Diagnosis Assistant

**Created by Gaurav Patil**  
📍 India | 🎓 B.Tech Computer Engineering 



## 🚀 Project Overview

**NeuroRAG** is a lightweight, Retrieval-Augmented Generation (RAG)-based AI assistant designed to answer natural language questions about mental health diagnoses and their ICD-10 classification codes.

The system leverages open-source LLMs, vector databases, and semantic search to intelligently interpret queries and retrieve structured information from the official **ICD-10 Mental and Behavioural Disorders** dataset (PDF document).



## 🎯 Key Features

✅ Question Answering over ICD-10 using natural language  
✅ Intelligent retrieval with FAISS + sentence-transformer embeddings  
✅ Local, open-source LLMs (e.g., Falcon-RW-1B)  
✅ Fully offline & privacy-friendly (no OpenAI or cloud APIs)  
✅ Fast response via cached answers  
✅ Beautiful Flask web dashboard + CLI interface  
✅ Upload your own ICD or medical PDFs dynamically  
✅ Relevance reranking with MMR for better semantic results  



## 🛠️ Tech Stack

| Component              | Technology Used                             |
|------------------------|----------------------------------------------|
| LLM                    | Falcon-RW-1B via HuggingFace Transformers   |
| Vector DB              | FAISS                                       |
| Embeddings             | `all-MiniLM-L6-v2` via Sentence-Transformers|
| RAG Framework          | LangChain (v0.2+) + LangChain-Community     |
| Frontend (Web)         | Flask + Beautiful Dashboard                 |
| Backend & Pipeline     | Python 3.10+                                |
| PDF to Text Conversion | PyMuPDF (fitz)                              |



## 🧠 Use Case Examples


❓ What are the diagnostic criteria for Obsessive-Compulsive Disorder?
💡 Answer: Obsessions and/or compulsions present for at least two weeks and cause distress or impairment in functioning.

❓ What is the ICD-10 code for Recurrent depressive disorder in remission?
💡 Answer: F33.4 – Recurrent depressive disorder, currently in remission.

🖥️ How to Run

## Quick Start (3 Steps)

1. **Navigate to project folder**
2. **Double-click** `START_FLASK.bat`
3. **Open browser** to http://127.0.0.1:5000

✅ See `QUICKSTART.md` for details!

## Detailed Setup

1. Clone the Repository
```bash
git clone https://github.com/GauravPatil2515/NEURO-RAG.git
cd NEURO-RAG
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run Tests (Optional)
```bash
python test_system.py
```

4. Launch Flask Web Interface
```bash
# Windows
START_FLASK.bat

# Or manually
python app_flask.py
```

Open browser to: **http://127.0.0.1:5000**

Or use CLI terminal version:
```bash
python app.py
```

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

-�️ Flask Dashboard – Clean, interactive web interface for medical professionals.

-🧠 MMR Retrieval – Diversified results for better answer grounding.

-💾 Caching – Faster repeat queries, lower compute load.

📌 Limitations

-Runs best on machines with at least 8GB RAM.

-Falcon-RW-1B used for lightweight inference; more complex LLMs may need GPU.

-Currently answers only English queries.

👨‍💻 About

NeuroRAG is an AI-powered project built with a focus on:

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
