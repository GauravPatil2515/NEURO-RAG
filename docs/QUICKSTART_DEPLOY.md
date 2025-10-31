# 🚀 Quick Start: Deploy to Hugging Face in 5 Minutes

## Before You Start

1. **Create Hugging Face Account** (if you don't have one)
   - Go to: https://huggingface.co/join
   - Sign up (it's free!)
   - Verify your email

2. **Create Your Space**
   - Go to: https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `neuro-rag`
   - SDK: **Docker** (important!)
   - Visibility: Public
   - Click "Create Space"

3. **Get Access Token**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: `neuro-rag-deploy`
   - Role: **Write**
   - Click "Generate"
   - **COPY THE TOKEN** (you won't see it again!)

---

## 🎯 Deploy in 3 Steps

### Method 1: Automated Script (Easiest!)

```powershell
# Open PowerShell in your project folder
cd "c:\Users\GAURAV PATIL\Desktop\gaurav's code\Project\neuro-rag\NEURO-RAG"

# Run deployment script
.\deploy_to_huggingface.ps1
```

Follow the prompts:
1. Enter your HF username
2. Enter space name (or press Enter for 'neuro-rag')
3. Confirm push
4. When asked for password, paste your **Access Token**

Done! 🎉

---

### Method 2: Manual Steps

```powershell
# 1. Add Hugging Face remote
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag

# 2. Add and commit files
git add .
git commit -m "Deploy to Hugging Face Spaces"

# 3. Push to Hugging Face
git push huggingface main
```

When asked:
- **Username**: Your HF username
- **Password**: Your HF Access Token (NOT your HF password!)

---

## 📊 Monitor Deployment

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag`
2. Click **"Logs"** tab
3. Watch the build (takes 3-5 minutes)
4. Look for: "Running on port 7860" ✅

---

## ✅ Test Your Deployment

1. Click **"App"** tab
2. Try a search: "What is schizophrenia?"
3. Verify results appear

**Success!** Your app is live! 🚀

---

## 🎨 Customize Your Space

### Update Title & Description

Edit the top of `README_HF.md`:

```yaml
---
title: NeuroRAG - Mental Health Assistant
emoji: 🧠
colorFrom: green
colorTo: blue
---
```

### Add Custom Domain

1. Space Settings → Custom domain
2. Enter: `neuro-rag.yourdomain.com`
3. Follow DNS instructions

---

## 🔄 Update Your Deployment

Made changes? Update with:

```powershell
git add .
git commit -m "Update: description"
git push huggingface main
```

HF automatically rebuilds! ⚡

---

## 🆘 Troubleshooting

### Authentication Failed?
- Use Access Token as password (NOT your HF password)
- Generate new token if expired

### Build Failed?
- Check Logs tab for errors
- Verify Dockerfile exists
- Check requirements.txt

### App Won't Load?
- Check port is 7860 in app.py
- Verify all files are present
- Check Logs for errors

### Large Files?
```powershell
git lfs install
git lfs track "*.faiss"
git add .gitattributes
```

---

## 📈 After Deployment

**Share Your Work:**
- [ ] Add to LinkedIn Projects
- [ ] Update Resume with URL
- [ ] Share on Twitter/X
- [ ] Add badge to GitHub README
- [ ] Write blog post
- [ ] Create demo video

**Add Badge to GitHub README:**
```markdown
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag)
```

---

## 🎯 Your Live URLs

After deployment:

- **Hugging Face Space**: `https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag`
- **Direct App URL**: `https://YOUR_USERNAME-neuro-rag.hf.space`
- **API Endpoint**: `https://YOUR_USERNAME-neuro-rag.hf.space/api/search`

---

## 💡 Pro Tips

1. **Test locally first**: `python app.py`
2. **Keep commits small**: easier to debug
3. **Monitor logs**: check daily for errors
4. **Engage users**: respond to discussions
5. **Update regularly**: show active development

---

## 📞 Need Help?

- **Full Guide**: See `DEPLOYMENT_GUIDE.md`
- **HF Docs**: https://huggingface.co/docs/hub/spaces
- **Community**: https://discuss.huggingface.co/

---

## ✨ Success Checklist

- [ ] HF account created
- [ ] Space created with Docker SDK
- [ ] Access token generated
- [ ] Files committed to git
- [ ] Pushed to huggingface remote
- [ ] Build completed successfully
- [ ] App loads and works
- [ ] Shared on social media
- [ ] Added to resume/portfolio

---

**Total Time: ~5 minutes** ⏱️

**Cost: $0 (100% Free!)** 💰

**Impact: Impressive portfolio piece!** 🌟

