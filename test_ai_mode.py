"""
Quick test to verify AI Mode is working
Makes a request with AI mode enabled
"""

import requests
import json
import time

print("=" * 70)
print("🧪 Testing AI Mode Integration")
print("=" * 70)
print()

# Test query
query = "What is major depressive disorder?"

print(f"📝 Query: {query}")
print(f"🤖 Mode: AI Mode (LLM-powered)")
print()

# Make request with AI mode enabled
print("🔄 Sending request to server...")
start_time = time.time()

try:
    response = requests.post(
        'http://127.0.0.1:5000/api/search',
        json={'query': query, 'use_ai': True},
        timeout=120  # 2 minute timeout for first LLM load
    )
    
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print(f"⏱️  Response Time: {elapsed:.2f} seconds")
        print(f"🎯 Mode: {data.get('mode', 'unknown').upper()}")
        print()
        print("─" * 70)
        print("ANSWER:")
        print("─" * 70)
        print(data.get('answer', 'No answer'))
        print("─" * 70)
        print()
        
        sources = data.get('sources', [])
        if sources:
            print(f"📚 Sources: {len(sources)} documents")
            for i, src in enumerate(sources[:3], 1):
                print(f"   {i}. {src[:100]}...")
        print()
        
        if data.get('mode') == 'llm':
            print("🎉 AI MODE IS WORKING!")
            print("   Phi-3-Mini successfully generating natural language answers!")
        else:
            print("⚠️  Fell back to retrieval mode (LLM might not be loaded yet)")
        
    else:
        print(f"❌ Error: HTTP {response.status_code}")
        print(response.text)
        
except requests.exceptions.Timeout:
    print("⏱️  Request timed out (LLM might be loading - this is normal on first request)")
    print("   Try again in 1-2 minutes")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
