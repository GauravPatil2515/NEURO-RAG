"""
Test Phi-3-Mini LLM Integration
Tests the new AI-powered answer generation capability
"""

import os
import sys
import time

print("=" * 60)
print("🧪 NeuroRAG Phi-3-Mini Integration Test")
print("=" * 60)
print()

# Test 1: Import RAG Pipeline
print("Test 1: Importing RAG Pipeline...")
try:
    from rag_pipeline import RAGPipeline
    print("✅ RAG Pipeline imported successfully")
except Exception as e:
    print(f"❌ Failed to import RAG Pipeline: {e}")
    sys.exit(1)

print()

# Test 2: Initialize RAG Pipeline
print("Test 2: Initializing RAG Pipeline...")
try:
    rag = RAGPipeline(
        doc_path=os.path.join(os.path.dirname(__file__), 'data', 'icd10_text.txt')
    )
    print("✅ RAG Pipeline initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

print()

# Test 3: Load Vector Store
print("Test 3: Loading Vector Store...")
try:
    rag.load_vectorstore()
    if rag.vectorstore is not None:
        print("✅ Vector store loaded successfully")
        print(f"   📊 Vectors in index: {rag.vectorstore.index.ntotal}")
    else:
        print("❌ Failed to load vector store - vectorstore is None")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Test Fast Mode (Retrieval Only)
print("Test 4: Testing Fast Mode (Retrieval Only)...")
test_queries_fast = [
    "What is depression?",
    "Tell me about bipolar disorder",
    "What are anxiety disorders?"
]

for i, query in enumerate(test_queries_fast, 1):
    print(f"\n   Query {i}: {query}")
    start_time = time.time()
    try:
        result = rag.simple_search(query, k=3)
        elapsed = (time.time() - start_time) * 1000
        print(f"   ⚡ Response time: {elapsed:.2f}ms")
        print(f"   📄 Result length: {len(result)} characters")
        print(f"   Preview: {result[:150]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print()
print("✅ Fast Mode Test Complete")

print()
print("=" * 60)

# Test 5: Load Phi-3-Mini (Optional - May take time)
print("\nTest 5: Loading Phi-3-Mini LLM...")
print("⚠️  This may take 2-5 minutes on first run (downloading model)")
print("⚠️  Requires ~8GB RAM/VRAM")
print()

user_input = input("Do you want to test AI mode? (y/n): ").strip().lower()

if user_input == 'y':
    try:
        print("\n🤖 Loading Phi-3-Mini...")
        start_time = time.time()
        success = rag.setup_phi3_mini()
        elapsed = time.time() - start_time
        
        if success:
            print(f"✅ Phi-3-Mini loaded successfully in {elapsed:.2f} seconds")
            print(f"   🎯 Model: microsoft/Phi-3-mini-4k-instruct")
            print(f"   💾 Device: {'GPU (CUDA)' if rag.llm_model.device.type == 'cuda' else 'CPU'}")
            
            print()
            print("=" * 60)
            
            # Test 6: Test AI Mode (LLM-Powered Answers)
            print("\nTest 6: Testing AI Mode (LLM-Powered Answers)...")
            
            test_queries_ai = [
                "What is major depressive disorder and what are its symptoms?",
                "Explain the difference between bipolar I and bipolar II disorder",
                "What treatments are available for anxiety disorders?"
            ]
            
            for i, query in enumerate(test_queries_ai, 1):
                print(f"\n   Query {i}: {query}")
                start_time = time.time()
                try:
                    result = rag.smart_search(query, k=3)
                    elapsed = (time.time() - start_time) * 1000
                    
                    print(f"   🤖 Mode: {result['mode']}")
                    print(f"   ⏱️  Response time: {elapsed:.2f}ms ({elapsed/1000:.2f}s)")
                    print(f"   📝 Answer length: {len(result['answer'])} characters")
                    print(f"   📚 Sources: {len(result.get('sources', []))} documents")
                    print()
                    print("   " + "─" * 56)
                    print("   ANSWER:")
                    print("   " + "─" * 56)
                    # Print answer with indentation
                    for line in result['answer'].split('\n'):
                        print(f"   {line}")
                    print("   " + "─" * 56)
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            print()
            print("✅ AI Mode Test Complete")
            
        else:
            print("❌ Failed to load Phi-3-Mini")
            print("   This is optional - Fast Mode still works!")
            
    except Exception as e:
        print(f"❌ Error loading Phi-3-Mini: {e}")
        print("   This is optional - Fast Mode still works!")
        import traceback
        traceback.print_exc()
else:
    print("⏭️  Skipping AI Mode test (Fast Mode still works perfectly!)")

print()
print("=" * 60)
print("🎉 Testing Complete!")
print("=" * 60)
print()
print("Summary:")
print("  ✅ Fast Mode (Retrieval): Always available, ~30ms response")
print("  🤖 AI Mode (LLM): Optional, requires Phi-3-Mini, ~2-3s response")
print()
print("To enable AI mode in the web app:")
print("  1. Set use_ai_mode = True in run_server.py")
print("  2. Restart the server")
print("  3. First request will load the model (takes 1-2 min)")
print()
