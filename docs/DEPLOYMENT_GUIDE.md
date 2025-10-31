# 🚀 Deploying NeuroRAG to Hugging Face Spaces

Complete step-by-step guide to deploy your NeuroRAG project to Hugging Face Spaces.

---

## 📋 Prerequisites

- ✅ GitHub account
- ✅ Hugging Face account (free - create at https://huggingface.co/join)
- ✅ Git installed on your computer
- ✅ Your NeuroRAG project code

---

## 🎯 Step-by-Step Deployment

### Step 1: Create Hugging Face Account

1. Go to https://huggingface.co/join
2. Sign up with email or GitHub
3. Verify your email
4. Complete your profile

### Step 2: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"** button
3. Fill in the details:
   - **Space name**: `neuro-rag` (or any name you like)
   - **License**: MIT
   - **Select SDK**: Choose **Docker** (important!)
   - **Hardware**: CPU basic (free tier)
   - **Visibility**: Public (for portfolio)

4. Click **"Create Space"**

### Step 3: Get Your Space URL

After creation, you'll see your Space URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag
```

Copy this URL - you'll need it!

### Step 4: Clone Your Space Repository

Open PowerShell in your project folder and run:

```powershell
# Navigate to your project
cd "c:\Users\GAURAV PATIL\Desktop\gaurav's code\Project\neuro-rag\NEURO-RAG"

# Add Hugging Face as a remote (replace YOUR_USERNAME)
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag

# Verify remotes
git remote -v
```

You should see both `origin` (GitHub) and `huggingface` remotes.

### Step 5: Prepare Files for Deployment

The following files have been created for you:
- ✅ `app.py` - Main entry point for HF
- ✅ `Dockerfile` - Container configuration
- ✅ `README_HF.md` - Space documentation
- ✅ `.dockerignore` - Files to exclude

### Step 6: Update README for Hugging Face

```powershell
# Copy the HF README to replace main README temporarily
cp README_HF.md README.md
```

### Step 7: Commit Changes

```powershell
# Add all files
git add .

# Commit
git commit -m "feat: Add Hugging Face Spaces deployment configuration

- Add Dockerfile for containerized deployment
- Update app.py as HF entry point
- Add comprehensive README for Space
- Configure for port 7860 (HF standard)
- Add health check endpoint
"
```

### Step 8: Push to Hugging Face

```powershell
# Push to Hugging Face Spaces
git push huggingface main

# If it asks for credentials, use:
# Username: YOUR_HF_USERNAME
# Password: YOUR_HF_ACCESS_TOKEN (create at https://huggingface.co/settings/tokens)
```

### Step 9: Create Hugging Face Access Token (if needed)

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `neuro-rag-deploy`
4. Role: **Write**
5. Click **"Generate"**
6. Copy the token (you won't see it again!)

Use this token as your password when pushing.

### Step 10: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag`
2. Click on **"Logs"** tab
3. Watch the build process (takes 3-5 minutes)
4. Look for:
   - ✅ "Building Docker image..."
   - ✅ "Installing dependencies..."
   - ✅ "Starting application..."
   - ✅ "Running on port 7860"

### Step 11: Test Your Deployment

Once deployment is complete:

1. Click on **"App"** tab
2. Your NeuroRAG interface should load
3. Try a search query: "What is schizophrenia?"
4. Verify results appear correctly

---

## 🔧 Troubleshooting

### Build Failed?

**Check Dockerfile:**
```powershell
# View build logs in HF Space > Logs tab
# Look for error messages
```

**Common issues:**
- Missing dependencies in `requirements.txt`
- Port not set to 7860
- FAISS index path incorrect

### App Not Loading?

**Check app.py:**
```python
# Make sure port is 7860
port = int(os.environ.get("PORT", 7860))
```

### Authentication Failed?

```powershell
# Use access token, not password
# Token from: https://huggingface.co/settings/tokens
```

### Large Files?

If you have files > 10MB:

```powershell
# Install git-lfs
git lfs install

# Track large files
git lfs track "*.faiss"
git lfs track "*.bin"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

---

## 🎨 Customizing Your Space

### Change Space Settings

1. Go to your Space
2. Click **Settings** (top right)
3. You can change:
   - Title
   - Emoji
   - Description
   - Custom domain
   - Hardware (upgrade to GPU if needed)

### Add Custom Domain

1. In Space Settings
2. Click **"Custom domain"**
3. Enter your domain: `neuro-rag.yourdomain.com`
4. Follow DNS configuration instructions

### Enable Community Features

1. Go to Space Settings
2. Enable:
   - ✅ Discussions
   - ✅ Pull requests
   - ✅ Community contributions

---

## 📊 Post-Deployment Checklist

- [ ] Space is publicly accessible
- [ ] Search functionality works
- [ ] UI loads correctly
- [ ] FAISS index loads (check logs)
- [ ] All routes work (/api/search, /api/stats, etc.)
- [ ] Health check passes
- [ ] README looks good
- [ ] Add Space to your resume/portfolio
- [ ] Share on LinkedIn/Twitter

---

## 🔄 Updating Your Deployment

To update your deployed app:

```powershell
# Make changes to your code
# Commit changes
git add .
git commit -m "Update: description of changes"

# Push to both GitHub and Hugging Face
git push origin main
git push huggingface main
```

Hugging Face will automatically rebuild and redeploy!

---

## 📈 Monitoring Your Space

### View Analytics

1. Go to your Space
2. Click **"Analytics"** tab
3. See:
   - Unique visitors
   - Page views
   - Geographic distribution
   - Usage over time

### Check Logs

1. Click **"Logs"** tab
2. See real-time application logs
3. Monitor errors and performance

---

## 💡 Tips for Success

1. **Test Locally First**: Always test with `python app.py` locally before deploying
2. **Small Commits**: Make small, incremental changes
3. **Clear README**: Good documentation gets more users
4. **Add Examples**: Include example queries in your README
5. **Monitor Logs**: Check logs regularly for errors
6. **Engage Community**: Respond to discussions and issues
7. **Share Your Work**: Post on social media with #HuggingFace

---

## 🎯 Next Steps

After successful deployment:

1. ✅ Add Space to your portfolio website
2. ✅ Add badge to GitHub README
3. ✅ Share on LinkedIn
4. ✅ Write a blog post about your project
5. ✅ Add to your resume
6. ✅ Create a demo video
7. ✅ Engage with users in Discussions

---

## 📞 Need Help?

- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Community Forum**: https://discuss.huggingface.co/
- **Discord**: https://discord.gg/hugging-face

---

## 🎉 Success!

Once deployed, your NeuroRAG will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/neuro-rag
```

Share this link on:
- LinkedIn
- GitHub README
- Resume
- Portfolio website
- Twitter/X

**Congratulations on deploying your AI application!** 🚀

---

