"""
Simple Phi-3-Mini Downloader
Downloads the model to HuggingFace cache automatically
"""

print("=" * 70)
print("🤖 Phi-3-Mini Model Downloader")
print("=" * 70)
print()
print("📦 Model: microsoft/Phi-3-mini-4k-instruct")
print("📏 Size: ~7.6 GB")
print("⏱️  Time: 10-15 minutes (depending on internet speed)")
print()
print("=" * 70)
print()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    print("Step 1/3: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True
    )
    print("✅ Tokenizer downloaded successfully")
    print()
    
    print("Step 2/3: Detecting device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}")
    print()
    
    print("Step 3/3: Downloading model (this will take several minutes)...")
    print("   Progress will be shown below:")
    print()
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        low_cpu_mem_usage=True
    )
    
    print()
    print("=" * 70)
    print("✅ SUCCESS! Phi-3-Mini downloaded and loaded!")
    print("=" * 70)
    print()
    print("📍 Model cached at:")
    print("   C:\\Users\\GAURAV PATIL\\.cache\\huggingface\\hub\\")
    print()
    print("Next steps:")
    print("  1. Edit run_server.py: set use_ai_mode = True")
    print("  2. Restart the Flask server")
    print("  3. Enable 'AI Mode' toggle on the website")
    print()
    
    # Quick test
    print("🧪 Quick Test:")
    print("   Generating a test response...")
    
    inputs = tokenizer("What is depression?", return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Response preview: {response[:100]}...")
    print()
    print("✅ Model is working correctly!")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ ERROR")
    print("=" * 70)
    print(f"Failed to download model: {e}")
    print()
    print("Common solutions:")
    print("  - Check internet connection")
    print("  - Free up disk space (~10GB needed)")
    print("  - Try again (download will resume)")
    print()
    import traceback
    traceback.print_exc()
