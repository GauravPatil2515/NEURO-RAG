# NeuroRAG Deployment Script for Hugging Face Spaces
# Run this script to deploy your project

Write-Host "🚀 NeuroRAG Deployment to Hugging Face Spaces" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Step 1: Check if git is installed
Write-Host "Step 1: Checking Git installation..." -ForegroundColor Cyan
if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "✅ Git is installed" -ForegroundColor Green
} else {
    Write-Host "❌ Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Step 2: Get Hugging Face username
Write-Host ""
Write-Host "Step 2: Enter your Hugging Face username" -ForegroundColor Cyan
$hf_username = Read-Host "Hugging Face Username"

if ([string]::IsNullOrWhiteSpace($hf_username)) {
    Write-Host "❌ Username cannot be empty" -ForegroundColor Red
    exit 1
}

# Step 3: Get Space name
Write-Host ""
Write-Host "Step 3: Enter your Space name (default: neuro-rag)" -ForegroundColor Cyan
$space_name = Read-Host "Space Name (press Enter for 'neuro-rag')"

if ([string]::IsNullOrWhiteSpace($space_name)) {
    $space_name = "neuro-rag"
}

# Build HF URL
$hf_url = "https://huggingface.co/spaces/$hf_username/$space_name"

Write-Host ""
Write-Host "📍 Your Space will be at: $hf_url" -ForegroundColor Yellow
Write-Host ""

# Step 4: Check if remote exists
Write-Host "Step 4: Configuring Git remotes..." -ForegroundColor Cyan

$remotes = git remote -v
if ($remotes -match "huggingface") {
    Write-Host "⚠️  Hugging Face remote already exists. Removing old remote..." -ForegroundColor Yellow
    git remote remove huggingface
}

# Add HF remote
git remote add huggingface $hf_url
Write-Host "✅ Added Hugging Face remote" -ForegroundColor Green

# Step 5: Verify files
Write-Host ""
Write-Host "Step 5: Verifying deployment files..." -ForegroundColor Cyan

$required_files = @("app.py", "Dockerfile", "requirements.txt", "run_server.py", "rag_pipeline.py")
$missing_files = @()

foreach ($file in $required_files) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
        $missing_files += $file
    }
}

if ($missing_files.Count -gt 0) {
    Write-Host ""
    Write-Host "❌ Missing required files. Please ensure all files are present." -ForegroundColor Red
    exit 1
}

# Step 6: Commit changes
Write-Host ""
Write-Host "Step 6: Committing changes..." -ForegroundColor Cyan

git add .

$commit_msg = @"
feat: Deploy NeuroRAG to Hugging Face Spaces

- Add Dockerfile for containerized deployment
- Update app.py as HF entry point  
- Configure for port 7860 (HF standard)
- Add comprehensive documentation
- Ready for production deployment
"@

git commit -m $commit_msg

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Changes committed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Nothing to commit (already up to date)" -ForegroundColor Yellow
}

# Step 7: Push to Hugging Face
Write-Host ""
Write-Host "Step 7: Pushing to Hugging Face..." -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  IMPORTANT: When asked for password, use your HF Access Token" -ForegroundColor Yellow
Write-Host "   Get it from: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
Write-Host ""

$confirm = Read-Host "Ready to push? (y/n)"

if ($confirm -eq "y" -or $confirm -eq "Y") {
    git push huggingface main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=============================================" -ForegroundColor Green
        Write-Host "🎉 SUCCESS! Deployment initiated!" -ForegroundColor Green
        Write-Host "=============================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your Space is being built at:" -ForegroundColor Cyan
        Write-Host $hf_url -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Go to: $hf_url" -ForegroundColor White
        Write-Host "2. Click 'Logs' tab to watch build progress" -ForegroundColor White
        Write-Host "3. Wait 3-5 minutes for build to complete" -ForegroundColor White
        Write-Host "4. Click 'App' tab to see your live application" -ForegroundColor White
        Write-Host ""
        Write-Host "Share your Space:" -ForegroundColor Cyan
        Write-Host "- LinkedIn: Add to your projects" -ForegroundColor White
        Write-Host "- Resume: Include the URL" -ForegroundColor White
        Write-Host "- GitHub: Add badge to README" -ForegroundColor White
        Write-Host ""
        
        # Offer to open in browser
        $open_browser = Read-Host "Open Space in browser? (y/n)"
        if ($open_browser -eq "y" -or $open_browser -eq "Y") {
            Start-Process $hf_url
        }
    } else {
        Write-Host ""
        Write-Host "❌ Push failed. Common issues:" -ForegroundColor Red
        Write-Host "1. Invalid credentials - use HF Access Token as password" -ForegroundColor Yellow
        Write-Host "2. Space doesn't exist - create it first at huggingface.co/spaces" -ForegroundColor Yellow
        Write-Host "3. Network issues - check your internet connection" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Deployment cancelled" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
