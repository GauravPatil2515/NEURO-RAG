# 🚀 Quick Start: Enable Phi-3-Mini AI Mode

## Step 1: Verify Installation

Make sure all required packages are installed:

```powershell
pip install transformers>=4.50.0
pip install torch>=2.0.0
pip install accelerate>=1.0.0
```

## Step 2: Enable AI Mode in Server

Edit `run_server.py` - Change line 11:

```python
# BEFORE:
use_ai_mode = False

# AFTER:
use_ai_mode = True
```

## Step 3: Restart Server

```powershell
cd NEURO-RAG
python run_server.py
```

**First Time:** Model will download (~7.6GB, takes 5-10 min)

**Subsequent Runs:** Model loads in ~30 seconds

## Step 4: Test AI Mode

Open your browser to `http://127.0.0.1:5000`

Try these queries:
- "What is major depressive disorder?"
- "Explain the symptoms of bipolar disorder"
- "What treatments are available for anxiety?"

You should see:
- 🤖 **AI Mode** badge (instead of ⚡ Fast Mode)
- Natural language answers (instead of raw text)
- Response time: 2-3 seconds (instead of 30ms)

## Optional: Test Without Web Server

```powershell
cd NEURO-RAG
python test_phi3_integration.py
```

When prompted "Do you want to test AI mode? (y/n):" type `y`

---

## 🎛️ Configuration Options

### Adjust Answer Length
Edit `rag_pipeline.py` → `generate_answer_with_phi3()`:

```python
max_new_tokens=500,  # Increase to 800 for longer answers
```

### Adjust Temperature (Creativity)
```python
temperature=0.7,  # Lower (0.3) = more factual, Higher (0.9) = more creative
```

### Force CPU (if GPU issues)
Edit `rag_pipeline.py` → `setup_phi3_mini()`:

```python
device = "cpu"  # Replace the auto-detect line
```

---

## 📊 Performance Expectations

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| ⚡ Fast | 20-50ms | Raw docs | Batch processing, API lookups |
| 🤖 AI | 2-3s | Natural language | User Q&A, explanations |

---

## ✅ Verification

Server console should show:

```
🤖 Attempting to load Phi-3-Mini...
   Using device: CPU (or GPU)
✅ AI Mode: ENABLED with microsoft/Phi-3-mini-4k-instruct
```

If you see:
```
⚡ Fast Mode: ENABLED (AI mode disabled)
```

Then `use_ai_mode` is still `False` - check Step 2.

---

## 🆘 Troubleshooting

**Problem:** "Out of memory"
**Solution:** Need 8GB+ RAM. Close other apps or use Fast Mode.

**Problem:** "Model download fails"
**Solution:** Check internet connection. Download is 7.6GB.

**Problem:** "AI mode not working"
**Solution:** Check console logs. Ensure `use_ai_mode = True`.

**Problem:** "Very slow responses (10+ seconds)"
**Solution:** Normal for CPU. GPU recommended for production.

---

## 🔄 Switch Back to Fast Mode

Edit `run_server.py`:

```python
use_ai_mode = False  # Disable AI mode
```

Restart server. All queries will use fast retrieval-only mode (~30ms).

---

**That's it!** Your NeuroRAG system now has AI-powered answer generation! 🎉
