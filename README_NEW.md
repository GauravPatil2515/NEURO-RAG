# 🧠 NeuroRAG - Mental Health AI Assistant

[![Status](https://img.shields.io/badge/status-working-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**AI-powered Mental Health Question Answering System using RAG (Retrieval-Augmented Generation)**

Built by **Gaurav Patil** | B.Tech Computer Engineering | India 🇮🇳

---

## ✨ Features

- 🔍 **Natural Language Search** - Ask questions in plain English about ICD-10 mental health codes
- ⚡ **Fast Response** - Instant answers using semantic search (< 2 seconds)
- 🎨 **Beautiful Dashboard** - Modern, responsive web interface
- 🔒 **100% Private** - Runs locally, no data sent to external APIs
- 🤖 **AI-Powered** - Uses advanced RAG pipeline with FAISS + HuggingFace
- 📊 **Real-time Stats** - Monitor system health and performance

---

## 🚀 Quick Start

### ⚡ Super Quick (3 Steps)

1. **Navigate to project folder**
2. **Double-click** `START_SERVER.bat`
3. **Open browser** to http://127.0.0.1:5000

✅ **Done! Start asking questions!**

---

## 📋 Prerequisites

- Python 3.10 or higher
- 8GB RAM recommended
- Windows/Linux/Mac

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GauravPatil2515/NEURO-RAG.git
cd NEURO-RAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Server

**Windows:**
```bash
START_SERVER.bat
```

**Linux/Mac:**
```bash
python run_server.py
```

### 4. Open Dashboard

Navigate to: **http://127.0.0.1:5000**

---

## 💡 Usage Examples

### Example Queries

- *"What is the code for Recurrent depressive disorder in remission?"*
  - **Answer:** F33.4 with full diagnostic criteria

- *"What are the diagnostic criteria for OCD?"*
  - **Answer:** Detailed ICD-10 information about Obsessive-Compulsive Disorder

- *"Tell me about bipolar disorder"*
  - **Answer:** Classification and diagnostic information

- *"Explain schizophrenia classification"*
  - **Answer:** ICD-10 Chapter V relevant sections

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (Dashboard)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask Server   │
│  (run_server.py)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│ ┌─────────────┐ │
│ │   FAISS     │ │  ← Vector Database
│ │  Embeddings │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ Falcon-1B   │ │  ← Language Model
│ │   (LLM)     │ │
│ └─────────────┘ │
└─────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Flask (Python) |
| **RAG Framework** | LangChain |
| **Vector DB** | FAISS |
| **Embeddings** | all-MiniLM-L6-v2 |
| **LLM** | Falcon-RW-1B |
| **Data** | ICD-10 Chapter V |

---

## 📁 Project Structure

```
NEURO-RAG/
├── run_server.py          # Main Flask server ⭐
├── rag_pipeline.py        # RAG logic & AI
├── templates/
│   └── index.html         # Dashboard UI
├── static/
│   ├── style.css          # Styling
│   └── script.js          # Interactivity
├── data/
│   └── icd10_text.txt     # Mental health data
├── faiss_index/
│   └── index.faiss        # Vector database
├── START_SERVER.bat       # Quick start script
├── test_complete.py       # Test suite
└── requirements.txt       # Dependencies
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_complete.py
```

**Expected Output:**
```
✅ Imports             PASS
✅ Files               PASS
✅ RAG Pipeline        PASS
✅ Flask Routes        PASS

🎉 ALL TESTS PASSED!
```

---

## 🎯 API Endpoints

### `GET /`
Dashboard homepage

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "message": "NeuroRAG server is running",
  "version": "1.0.0"
}
```

### `POST /api/search`
Search for mental health information
```json
{
  "query": "What is depression?"
}
```

### `GET /api/stats`
Get system statistics
```json
{
  "vectorstore_loaded": true,
  "data_file_exists": true,
  "index_exists": true,
  "status": "online"
}
```

---

## ⚙️ Configuration

### Environment Variables

- `USE_TF=0` - Disable TensorFlow
- `TRANSFORMERS_NO_TF=1` - Use PyTorch only
- `PYTHONUNBUFFERED=1` - Real-time output

### Server Settings

Edit `run_server.py`:
- Change port: `port=5000`
- Change host: `host='127.0.0.1'`
- Toggle debug: `debug=True`

---

## 🐛 Troubleshooting

### Issue: Connection Refused

**Solution:** Make sure the server is running
- Check for green terminal window
- Look for: "Running on http://127.0.0.1:5000"
- Restart with `START_SERVER.bat`

### Issue: Port Already in Use

**Solution:** Close other apps on port 5000 or change the port

### Issue: Import Errors

**Solution:** Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Vector Store Not Found

**Solution:** Build the vector store
```bash
python test_system.py
```

---

## 📊 Performance

- **Search Speed:** < 2 seconds
- **Memory Usage:** ~2GB RAM
- **Accuracy:** High (RAG-based retrieval)
- **Offline:** 100% local execution

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is open-source under the MIT License.

---

## 🙏 Acknowledgments

- **HuggingFace** 🤗 for open-source models
- **LangChain** for RAG framework
- **WHO ICD-10** for medical classification data
- **FAISS** for efficient vector search

---

## 👨‍💻 Author

**Gaurav Patil**  
B.Tech Computer Engineering  
📍 India  

**Contact:**
- GitHub: [@GauravPatil2515](https://github.com/GauravPatil2515)
- Project: [NEURO-RAG](https://github.com/GauravPatil2515/NEURO-RAG)

---

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Export search results to PDF
- [ ] Advanced filtering options
- [ ] User authentication
- [ ] API rate limiting
- [ ] Docker containerization
- [ ] Cloud deployment guide

---

## ⭐ Star This Project

If you find NeuroRAG helpful, please give it a star! ⭐

---

**Built with ❤️ using AI & Open Source Technology**

*Last Updated: October 25, 2025*
