---
title: NeuroRAG - ICD-10 Mental Health Assistant
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🧠 NeuroRAG - ICD-10 Mental Health Diagnostic Assistant

<div align="center">

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Retrieval-Augmented Generation (RAG) system for ICD-10 mental health diagnoses**

[Try it Now](https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag) • [View on GitHub](https://github.com/GauravPatil2515/NEURO-RAG)

</div>

---

## 🎯 What is NeuroRAG?

NeuroRAG is an intelligent diagnostic assistant that helps healthcare professionals and students quickly search and retrieve information from ICD-10 Chapter V (Mental and Behavioral Disorders). It uses advanced AI techniques to understand natural language queries and provide accurate, contextual answers.

### ✨ Key Features

- 🔍 **Semantic Search**: Understands the meaning behind your questions
- ⚡ **Fast Retrieval**: Returns results in ~30ms using FAISS vector search
- 🤖 **AI Mode**: Optional LLM-powered answers (Phi-3-Mini)
- 📊 **1,438 Diagnostic Codes**: Complete ICD-10 Chapter V coverage
- 🎨 **Modern UI**: Professional forest-green themed dashboard
- 📈 **Real-time Stats**: Database insights and search analytics

---

## 🚀 How It Works

```
User Query → Embedding Model → FAISS Search → Top 5 Results → Display
               (384-dim)        (Vector DB)      (29ms avg)
```

**Technology Stack:**
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask (Python 3.9+)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM**: Phi-3-Mini (optional, 3.8B parameters)

---

## 📖 Use Cases

1. **Medical Students**: Quick reference for ICD-10 codes
2. **Healthcare Professionals**: Diagnostic code lookup
3. **Researchers**: Mental health classification research
4. **Coders**: Medical billing and coding

---

## 🎓 Example Queries

Try these questions:

- "What is schizophrenia?"
- "What is the code for recurrent depressive disorder?"
- "Symptoms of bipolar disorder"
- "Difference between F32 and F33"
- "OCD criteria"
- "PTSD diagnostic code"

---

## 🏗️ Architecture

### RAG Pipeline

```python
1. Document Loading
   ├── ICD-10 text file (658 KB)
   └── 14,061 lines of diagnostic criteria

2. Text Chunking
   ├── Chunk size: 500 characters
   ├── Overlap: 50 characters
   └── Total chunks: 1,438

3. Embedding Generation
   ├── Model: all-MiniLM-L6-v2
   ├── Dimension: 384
   └── Downloaded: 90.9 MB

4. Vector Storage
   ├── FAISS IndexFlatL2
   ├── Size: 2.11 MB
   └── Exact similarity search

5. Query Processing
   ├── Query → Embedding
   ├── FAISS search (top 5)
   └── Results display
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Average Retrieval Time | 29.14 ms |
| Queries per Second | 34.32 |
| Semantic Discrimination | 6.13x |
| Test Pass Rate | 100% (8/8) |
| Database Size | 658.84 KB |
| Vector Index Size | 2.11 MB |

---

## 🎨 UI Features

- **Modern Design**: Clean, professional forest-green theme
- **Responsive**: Works on desktop, tablet, and mobile
- **Accessibility**: WCAG 2.1 compliant
- **Dark Accents**: Easy on the eyes
- **Grid Background**: Subtle visual design
- **Smooth Animations**: Professional transitions

---

## 🔧 Local Development

```bash
# Clone repository
git clone https://github.com/GauravPatil2515/NEURO-RAG.git
cd NEURO-RAG

# Install dependencies
pip install -r requirements.txt

# Run locally
python run_server.py

# Visit
http://localhost:5000
```

---

## 📦 Dependencies

- Flask==3.0.0
- sentence-transformers==2.2.2
- faiss-cpu==1.7.4
- langchain==0.1.0
- transformers==4.36.0

See `requirements.txt` for complete list.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Gaurav Patil**
- GitHub: [@GauravPatil2515](https://github.com/GauravPatil2515)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- ICD-10 data from WHO
- Sentence-Transformers by UKPLab
- FAISS by Facebook AI Research
- Hugging Face for hosting

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for healthcare professionals and students

</div>
