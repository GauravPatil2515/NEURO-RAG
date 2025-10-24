# ✅ NeuroRAG - WORKING VERSION

## 🎉 Status: FULLY FUNCTIONAL

The dashboard is now working correctly! All connection issues have been resolved.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Open the Project Folder
Navigate to:
```
c:\Users\GAURAV PATIL\Desktop\gaurav's code\Project\neuro-rag\NEURO-RAG
```

### Step 2: Start the Server
**Double-click:** `START_SERVER.bat`

### Step 3: Open Your Browser
Go to: **http://127.0.0.1:5000**

✅ **That's it! The dashboard is now running!**

---

## 🌐 Access URLs

- **Primary:** http://127.0.0.1:5000
- **Alternative:** http://localhost:5000

Both URLs work identically.

---

## ✨ What You Can Do

### 🔍 Search Mental Health Information
Ask questions like:
- "What is the code for Recurrent depressive disorder in remission?"
- "Tell me about bipolar disorder"
- "What are the diagnostic criteria for OCD?"
- "Explain schizophrenia"

### 📊 View System Stats
- Vector store status
- AI model information
- Real-time system health

### 💡 Quick Examples
Click any suggestion chip on the dashboard to instantly try sample queries!

---

## ⚠️ Important Notes

1. **Keep the terminal window OPEN** - Don't close it while using the app
2. The window shows server logs and activity
3. To stop: Press `Ctrl+C` in the terminal window
4. If you see errors, check that port 5000 is not in use by another program

---

## 🔧 Files in This Repository

### Main Application Files
- `run_server.py` - **Main server (USE THIS)** ✅
- `templates/index.html` - Dashboard HTML
- `static/style.css` - Beautiful styling
- `static/script.js` - Interactive features
- `rag_pipeline.py` - AI/RAG logic
- `data/icd10_text.txt` - ICD-10 mental health data
- `faiss_index/` - Vector database

### Startup Files
- `START_SERVER.bat` - **Double-click this to start** ✅
- `START_FLASK.bat` - Alternative startup
- `START_FLASK.ps1` - PowerShell version

### Other Files
- `app.py` - CLI version
- `app_streamlit.py` - Streamlit version
- `test_system.py` - System tests
- `requirements.txt` - Python dependencies

---

## ✅ What Was Fixed

### Previous Issues:
- ❌ Connection refused errors
- ❌ Server binding to 0.0.0.0 causing issues
- ❌ Import errors with spacy/langchain
- ❌ Server exiting unexpectedly

### Solutions Applied:
- ✅ Changed server to bind to 127.0.0.1 (localhost)
- ✅ Fixed import paths for langchain_text_splitters
- ✅ Created robust `run_server.py` with error handling
- ✅ Added lazy loading for RAG pipeline
- ✅ Improved startup scripts

---

## 🧪 Test Results

**Server Health Check:** ✅ PASSING
```json
{
  "status": "healthy",
  "message": "NeuroRAG server is running",
  "version": "1.0.0"
}
```

**Search Test:** ✅ WORKING
Query: "What is the code for Recurrent depressive disorder in remission?"
Result: Correctly returns F33.4 with full diagnostic information

**Dashboard:** ✅ RESPONSIVE
All features working:
- Search interface ✅
- Results display ✅
- Statistics cards ✅
- Navigation ✅
- Styling ✅

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** Flask (Python)
- **AI Engine:** LangChain + FAISS + HuggingFace
- **LLM:** Falcon-RW-1B
- **Embeddings:** all-MiniLM-L6-v2
- **Database:** ICD-10 Chapter V (Mental Health)

---

## 📞 Support

If you encounter any issues:

1. Check that port 5000 is available
2. Verify Python 3.10+ is installed
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check terminal window for error messages

---

## 👨‍💻 Developer

**Gaurav Patil**  
B.Tech Computer Engineering  
📍 India  
🔗 GitHub: [GauravPatil2515/NEURO-RAG](https://github.com/GauravPatil2515/NEURO-RAG)

---

## 🎉 Enjoy Your NeuroRAG Dashboard!

Built with ❤️ using AI & Open Source Technology

---

**Last Updated:** October 25, 2025  
**Status:** ✅ Production Ready
