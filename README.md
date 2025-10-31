# 🧠 NeuroRAG - Mental Health AI Assistant---

title: NeuroRAG - Mental Health AI Assistant

[![Status](https://img.shields.io/badge/status-working-brightgreen)]()emoji: 🧠

[![Python](https://img.shields.io/badge/python-3.9+-blue)]()colorFrom: green

[![License](https://img.shields.io/badge/license-MIT-green)]()colorTo: blue

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/GauravPatil2515/neuro-rag)sdk: docker

app_file: app.py

**AI-powered Mental Health Question Answering System using RAG (Retrieval-Augmented Generation)**pinned: false

license: mit

Built by **Gaurav Patil** | B.Tech Computer Engineering | India 🇮🇳---



---# 🧠 NeuroRAG - Mental Health AI Assistant



## ✨ Features[![Status](https://img.shields.io/badge/status-working-brightgreen)]()

[![Python](https://img.shields.io/badge/python-3.9+-blue)]()

- 🔍 **Natural Language Search** - Ask questions in plain English about ICD-10 mental health codes[![License](https://img.shields.io/badge/license-MIT-green)]()

- ⚡ **Fast Response** - Instant answers using semantic search (< 2 seconds)[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/GauravPatil2515/neuro-rag)

- 🎨 **Beautiful Dashboard** - Modern, responsive web interface

- 🔒 **100% Private** - Runs locally, no data sent to external APIs**AI-powered Mental Health Question Answering System using RAG (Retrieval-Augmented Generation)**

- 🤖 **AI-Powered** - Uses advanced RAG pipeline with FAISS + HuggingFace

- 📊 **Real-time Stats** - Monitor system health and performanceBuilt by **Gaurav Patil** | B.Tech Computer Engineering | India 🇮🇳



------



## 🚀 Quick Start## ✨ Features



### ⚡ Super Quick (3 Steps)- 🔍 **Natural Language Search** - Ask questions in plain English about ICD-10 mental health codes

- ⚡ **Fast Response** - Instant answers using semantic search (< 2 seconds)

1. **Navigate to project folder**- 🎨 **Beautiful Dashboard** - Modern, responsive web interface

2. **Double-click** `scripts\START_SERVER.bat`- 🔒 **100% Private** - Runs locally, no data sent to external APIs

3. **Open browser** to http://127.0.0.1:5000- 🤖 **AI-Powered** - Uses advanced RAG pipeline with FAISS + HuggingFace

- 📊 **Real-time Stats** - Monitor system health and performance

✅ **Done! Start asking questions!**

---

---

## 🚀 Quick Start

## 📋 Prerequisites

### ⚡ Super Quick (3 Steps)

- Python 3.10 or higher

- 8GB RAM recommended1. **Navigate to project folder**

- Windows/Linux/Mac2. **Double-click** `START_SERVER.bat`

3. **Open browser** to http://127.0.0.1:5000

---

✅ **Done! Start asking questions!**

## 🛠️ Installation

---

### 1. Clone the Repository

## 📋 Prerequisites

```bash

git clone https://github.com/GauravPatil2515/NEURO-RAG.git- Python 3.10 or higher

cd NEURO-RAG- 8GB RAM recommended

```- Windows/Linux/Mac



### 2. Install Dependencies---



```bash## 🛠️ Installation

pip install -r requirements.txt

```### 1. Clone the Repository



### 3. Start the Server```bash

git clone https://github.com/GauravPatil2515/NEURO-RAG.git

**Windows:**cd NEURO-RAG

```bash```

scripts\START_SERVER.bat

```### 2. Install Dependencies



**Linux/Mac:**```bash

```bashpip install -r requirements.txt

python run_server.py```

```

### 3. Start the Server

### 4. Open Dashboard

**Windows:**

Navigate to: **http://127.0.0.1:5000**```bash

START_SERVER.bat

---```



## 💡 Usage Examples**Linux/Mac:**

```bash

### Example Queriespython run_server.py

```

- *"What is the code for Recurrent depressive disorder in remission?"*

  - **Answer:** F33.4 with full diagnostic criteria### 4. Open Dashboard



- *"What are the diagnostic criteria for OCD?"*Navigate to: **http://127.0.0.1:5000**

  - **Answer:** Detailed ICD-10 information about Obsessive-Compulsive Disorder

---

- *"Tell me about bipolar disorder"*

  - **Answer:** Classification and diagnostic information## 💡 Usage Examples



- *"Explain schizophrenia classification"*### Example Queries

  - **Answer:** ICD-10 Chapter V relevant sections

- *"What is the code for Recurrent depressive disorder in remission?"*

---  - **Answer:** F33.4 with full diagnostic criteria



## 🏗️ Architecture- *"What are the diagnostic criteria for OCD?"*

  - **Answer:** Detailed ICD-10 information about Obsessive-Compulsive Disorder

```

┌─────────────────┐- *"Tell me about bipolar disorder"*

│   Web Browser   │  - **Answer:** Classification and diagnostic information

│   (Dashboard)   │

└────────┬────────┘- *"Explain schizophrenia classification"*

         │  - **Answer:** ICD-10 Chapter V relevant sections

         ▼

┌─────────────────┐---

│  Flask Server   │

│  (run_server.py)│## 🏗️ Architecture

└────────┬────────┘

         │```

         ▼┌─────────────────┐

┌─────────────────┐│   Web Browser   │

│  RAG Pipeline   ││   (Dashboard)   │

│ ┌─────────────┐ │└────────┬────────┘

│ │   FAISS     │ │  ← Vector Database         │

│ │  Embeddings │ │         ▼

│ └─────────────┘ │┌─────────────────┐

│ ┌─────────────┐ ││  Flask Server   │

│ │ Falcon-1B   │ │  ← Language Model│  (run_server.py)│

│ │   (LLM)     │ │└────────┬────────┘

│ └─────────────┘ │         │

└─────────────────┘         ▼

```┌─────────────────┐

│  RAG Pipeline   │

---│ ┌─────────────┐ │

│ │   FAISS     │ │  ← Vector Database

## 🛠️ Tech Stack│ │  Embeddings │ │

│ └─────────────┘ │

| Component | Technology |│ ┌─────────────┐ │

|-----------|-----------|│ │ Falcon-1B   │ │  ← Language Model

| **Frontend** | HTML5, CSS3, JavaScript |│ │   (LLM)     │ │

| **Backend** | Flask (Python) |│ └─────────────┘ │

| **RAG Framework** | LangChain |└─────────────────┘

| **Vector DB** | FAISS |```

| **Embeddings** | all-MiniLM-L6-v2 |

| **LLM** | Falcon-RW-1B / Phi-3-Mini (optional) |---

| **Data** | ICD-10 Chapter V |

## 🛠️ Tech Stack

---

| Component | Technology |

## 📁 Project Structure|-----------|-----------|

| **Frontend** | HTML5, CSS3, JavaScript |

```| **Backend** | Flask (Python) |

NEURO-RAG/| **RAG Framework** | LangChain |

├── 📁 src/                      # Core source code| **Vector DB** | FAISS |

│   ├── rag_pipeline.py         # RAG logic & AI ⭐| **Embeddings** | all-MiniLM-L6-v2 |

│   ├── utils.py                # Utility functions| **LLM** | Falcon-RW-1B |

│   └── __init__.py| **Data** | ICD-10 Chapter V |

│

├── 📁 tests/                    # Test suite---

│   ├── test_complete.py        # Comprehensive tests

│   ├── test_system.py          # System tests## 📁 Project Structure

│   └── __init__.py

│```

├── 📁 docs/                     # DocumentationNEURO-RAG/

│   ├── DEPLOYMENT_GUIDE.md     # Deployment instructions├── run_server.py          # Main Flask server ⭐

│   ├── QUICKSTART.md           # Quick start guide├── rag_pipeline.py        # RAG logic & AI

│   ├── QUICKSTART_AI_MODE.md   # AI mode setup├── templates/

│   ├── QUICKSTART_DEPLOY.md    # Deployment quickstart│   └── index.html         # Dashboard UI

│   ├── ML_RAG_TECHNICAL_GUIDE.md  # Technical details├── static/

│   ├── PHI3_INTEGRATION_SUMMARY.md│   ├── style.css          # Styling

│   ├── SYSTEM_STATUS.md│   └── script.js          # Interactivity

│   ├── TEST_RESULTS_REAL_WORLD.md├── data/

│   └── README_HF.md            # Hugging Face README│   └── icd10_text.txt     # Mental health data

│├── faiss_index/

├── 📁 scripts/                  # Utility scripts│   └── index.faiss        # Vector database

│   ├── START_SERVER.bat        # Quick start script ⭐├── START_SERVER.bat       # Quick start script

│   ├── push_to_hf.bat          # HuggingFace deployment├── test_complete.py       # Test suite

│   └── deploy_to_huggingface.ps1└── requirements.txt       # Dependencies

│```

├── 📁 templates/                # HTML templates

│   └── index.html              # Dashboard UI---

│

├── 📁 static/                   # Static assets## 🧪 Testing

│   ├── style.css               # Styling

│   └── script.js               # Frontend logicRun the comprehensive test suite:

│

├── 📁 data/                     # Data files```bash

│   └── icd10_text.txt          # Mental health datapython test_complete.py

│```

├── 📁 faiss_index/              # Vector database

│   ├── index.faiss             # FAISS index**Expected Output:**

│   └── index.pkl               # Index metadata```

│✅ Imports             PASS

├── 📄 run_server.py             # Main Flask server ⭐✅ Files               PASS

├── 📄 app.py                    # HuggingFace entry point✅ RAG Pipeline        PASS

├── 📄 app_streamlit.py          # Streamlit interface✅ Flask Routes        PASS

├── 📄 requirements.txt          # Dependencies

├── 📄 Dockerfile                # Docker configuration🎉 ALL TESTS PASSED!

├── 📄 README.md                 # This file```

├── 📄 .gitignore

├── 📄 .gitattributes---

└── 📄 .dockerignore

```## 🎯 API Endpoints



---### `GET /`

Dashboard homepage

## 🧪 Testing

### `GET /health`

Run the comprehensive test suite:Health check endpoint

```json

```bash{

# From project root  "status": "healthy",

python -m tests.test_complete  "message": "NeuroRAG server is running",

  "version": "1.0.0"

# Or run system tests}

python -m tests.test_system```

```

### `POST /api/search`

**Expected Output:**Search for mental health information

``````json

✅ Imports             PASS{

✅ Files               PASS  "query": "What is depression?"

✅ RAG Pipeline        PASS}

✅ Flask Routes        PASS```



🎉 ALL TESTS PASSED!### `GET /api/stats`

```Get system statistics

```json

---{

  "vectorstore_loaded": true,

## 🎯 API Endpoints  "data_file_exists": true,

  "index_exists": true,

### `GET /`  "status": "online"

Dashboard homepage}

```

### `GET /health`

Health check endpoint---

```json

{## ⚙️ Configuration

  "status": "healthy",

  "message": "NeuroRAG server is running",### Environment Variables

  "version": "1.0.0"

}- `USE_TF=0` - Disable TensorFlow

```- `TRANSFORMERS_NO_TF=1` - Use PyTorch only

- `PYTHONUNBUFFERED=1` - Real-time output

### `POST /api/search`

Search for mental health information### Server Settings

```json

{Edit `run_server.py`:

  "query": "What is depression?",- Change port: `port=5000`

  "use_ai": false- Change host: `host='127.0.0.1'`

}- Toggle debug: `debug=True`

```

---

### `GET /api/stats`

Get system statistics## 🐛 Troubleshooting

```json

{### Issue: Connection Refused

  "vectorstore_loaded": true,

  "data_file_exists": true,**Solution:** Make sure the server is running

  "index_exists": true,- Check for green terminal window

  "status": "online"- Look for: "Running on http://127.0.0.1:5000"

}- Restart with `START_SERVER.bat`

```

### Issue: Port Already in Use

### `GET /api/database`

Get database content (for searching and viewing)**Solution:** Close other apps on port 5000 or change the port



---### Issue: Import Errors



## ⚙️ Configuration**Solution:** Reinstall dependencies

```bash

### Environment Variablespip install -r requirements.txt --force-reinstall

```

- `USE_TF=0` - Disable TensorFlow

- `TRANSFORMERS_NO_TF=1` - Use PyTorch only### Issue: Vector Store Not Found

- `PYTHONUNBUFFERED=1` - Real-time output

**Solution:** Build the vector store

### Server Settings```bash

python test_system.py

Edit `run_server.py`:```

- Change port: `port=5000`

- Change host: `host='127.0.0.1'`---

- Toggle debug: `debug=True`

## 📊 Performance

---

- **Search Speed:** < 2 seconds

## 🐛 Troubleshooting- **Memory Usage:** ~2GB RAM

- **Accuracy:** High (RAG-based retrieval)

### Issue: Import Errors After Restructuring- **Offline:** 100% local execution



**Solution:** Make sure you're running from the project root:---

```bash

cd NEURO-RAG## 🤝 Contributing

python run_server.py

```Contributions are welcome! Please:



### Issue: Connection Refused1. Fork the repository

2. Create a feature branch

**Solution:** Make sure the server is running3. Make your changes

- Check for green terminal window4. Submit a pull request

- Look for: "Running on http://127.0.0.1:5000"

- Restart with `scripts\START_SERVER.bat`---



### Issue: Port Already in Use## 📄 License



**Solution:** Close other apps on port 5000 or change the portThis project is open-source under the MIT License.



### Issue: Module Not Found---



**Solution:** Reinstall dependencies## 🙏 Acknowledgments

```bash

pip install -r requirements.txt --force-reinstall- **HuggingFace** 🤗 for open-source models

```- **LangChain** for RAG framework

- **WHO ICD-10** for medical classification data

### Issue: Vector Store Not Found- **FAISS** for efficient vector search



**Solution:** Build the vector store---

```bash

python -m tests.test_system## 👨‍💻 Author

```

**Gaurav Patil**  

---B.Tech Computer Engineering  

📍 India  

## 📊 Performance

**Contact:**

- **Search Speed:** < 2 seconds- GitHub: [@GauravPatil2515](https://github.com/GauravPatil2515)

- **Memory Usage:** ~2GB RAM- Project: [NEURO-RAG](https://github.com/GauravPatil2515/NEURO-RAG)

- **Accuracy:** High (RAG-based retrieval)

- **Offline:** 100% local execution---



---## 📈 Future Enhancements



## 🤝 Contributing- [ ] Multi-language support

- [ ] Export search results to PDF

Contributions are welcome! Please:- [ ] Advanced filtering options

- [ ] User authentication

1. Fork the repository- [ ] API rate limiting

2. Create a feature branch- [ ] Docker containerization

3. Make your changes- [ ] Cloud deployment guide

4. Submit a pull request

---

See `CONTRIBUTING.md` for details.

## ⭐ Star This Project

---

If you find NeuroRAG helpful, please give it a star! ⭐

## 📄 License

---

This project is open-source under the MIT License.

**Built with ❤️ using AI & Open Source Technology**

---

*Last Updated: October 25, 2025*

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
- HuggingFace: [neuro-rag](https://huggingface.co/spaces/GauravPatil2515/neuro-rag)

---

## 📈 Recent Updates (v1.0.0)

- ✅ **Restructured project** for better organization
- ✅ **Separated concerns**: src/, tests/, docs/, scripts/
- ✅ **Improved imports** with proper package structure
- ✅ **Enhanced documentation** and guides
- ✅ **Cleaned up** redundant files

---

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Export search results to PDF
- [ ] Advanced filtering options
- [ ] User authentication
- [ ] API rate limiting
- [ ] Enhanced Docker containerization
- [ ] Cloud deployment automation

---

## ⭐ Star This Project

If you find NeuroRAG helpful, please give it a star! ⭐

---

**Built with ❤️ using AI & Open Source Technology**

*Last Updated: November 1, 2025*
