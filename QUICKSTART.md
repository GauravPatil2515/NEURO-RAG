# 🚀 Quick Start Guide - NeuroRAG Dashboard

## ✅ FIXED: Connection Issue Resolved!

The dashboard now runs on **localhost** instead of 0.0.0.0 to avoid connection issues.

---

## 🎯 How to Start the Dashboard (2 Simple Steps)

### Option 1: Double-Click Method (Easiest!)

1. **Navigate to the project folder:**
   ```
   c:\Users\GAURAV PATIL\Desktop\gaurav's code\Project\neuro-rag\NEURO-RAG
   ```

2. **Double-click:** `START_SERVER.bat`

3. **Open your browser to:** http://localhost:5000

✅ That's it! The dashboard is now running!

---

### Option 2: Command Line Method

1. **Open PowerShell or Command Prompt**

2. **Run this command:**
   ```powershell
   cd "c:\Users\GAURAV PATIL\Desktop\gaurav's code\Project\neuro-rag\NEURO-RAG"
   python app_simple.py
   ```

3. **Open your browser to:** http://localhost:5000

---

## 🌐 Dashboard URLs

- **Primary:** http://localhost:5000
- **Alternative:** http://127.0.0.1:5000

Both URLs work the same!

---

## ⚠️ Important Notes

1. **Keep the terminal/command window OPEN** while using the dashboard
2. The window will show server logs and activity
3. To stop the server: Press `Ctrl+C` in the terminal window
4. If port 5000 is busy, the server will show an error - close other apps using that port

---

## 🔍 Features of the Dashboard

### 🎨 Beautiful Interface
- Modern green-themed design
- Professional sidebar navigation
- Responsive layout for all devices

### 🔎 Smart Search
- Ask questions in natural language
- Get instant AI-powered answers from ICD-10 database
- Quick example queries to get started

### 📊 Live Statistics
- Vector store status
- System health monitoring
- Real-time updates

### 💡 Example Queries
Try these sample questions:
- "What is the code for Recurrent depressive disorder in remission?"
- "What are the diagnostic criteria for Obsessive-Compulsive Disorder?"
- "Tell me about bipolar disorder"
- "What is schizophrenia?"
- "Anxiety disorders classification"

---

## 🛠️ Troubleshooting

### Problem: "Connection Refused" Error

**Solution:** Make sure the Flask server is running!
- Check if you see the green terminal window
- Look for: "Running on http://localhost:5000"
- If not visible, run `START_SERVER.bat` again

### Problem: Port Already in Use

**Solution:**
1. Close any other programs using port 5000
2. Or change the port in `app_simple.py` (line: `port=5000`)
3. Restart the server

### Problem: Page Not Loading

**Solution:**
1. Refresh the browser (F5)
2. Clear browser cache (Ctrl+Shift+Delete)
3. Try the alternative URL: http://127.0.0.1:5000

---

## 📝 What Changed?

### Before (Issue):
- Server ran on `0.0.0.0:5000` 
- Connection refused errors
- Complex dependency loading

### After (Fixed):
- Server runs on `localhost:5000` ✅
- Stable connection ✅
- Lazy loading for better performance ✅
- Simplified app (`app_simple.py`) ✅

---

## 🎓 Technical Details

**Files:**
- `app_simple.py` - Simplified Flask server (more stable)
- `app_flask.py` - Original Flask server (still available)
- `templates/index.html` - Beautiful dashboard HTML
- `static/style.css` - Professional styling
- `static/script.js` - Interactive features

**Stack:**
- Frontend: HTML5, CSS3, Vanilla JavaScript
- Backend: Flask (Python)
- AI: LangChain + FAISS + HuggingFace
- Database: ICD-10 Mental Health (Chapter V)

---

## 🆘 Need Help?

If you're still having issues:

1. **Check the terminal output** for error messages
2. **Ensure Python is installed** (Python 3.10+)
3. **Verify dependencies:** `pip install -r requirements.txt`
4. **Contact:** [Your contact info]

---

## 🎉 Enjoy Your NeuroRAG Dashboard!

Built with ❤️ by **Gaurav Patil**  
Powered by AI & Open Source Technology

---
