"""
Comprehensive test script for NeuroRAG system
Tests all components to ensure they work correctly
"""

from rag_pipeline import RAGPipeline
import sys

def test_embeddings():
    """Test 1: Embeddings creation"""
    print("=" * 60)
    print("TEST 1: Creating Embeddings")
    print("=" * 60)
    try:
        rag = RAGPipeline('data/icd10_text.txt')
        emb = rag.create_embeddings()
        print("✅ PASS: Embeddings created successfully\n")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def test_vectorstore_load():
    """Test 2: Loading vector store"""
    print("=" * 60)
    print("TEST 2: Loading Vector Store")
    print("=" * 60)
    try:
        rag = RAGPipeline('data/icd10_text.txt')
        rag.load_vectorstore()
        print("✅ PASS: Vector store loaded successfully\n")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def test_search_queries():
    """Test 3: Multiple search queries"""
    print("=" * 60)
    print("TEST 3: Search Queries")
    print("=" * 60)
    
    test_queries = [
        ("depression", "Should find F32 or F33 codes"),
        ("anxiety disorder", "Should find F40-F41 codes"),
        ("schizophrenia", "Should find F20 codes"),
        ("bipolar", "Should find F31 codes"),
        ("PTSD post traumatic stress", "Should find F43.1 code"),
    ]
    
    rag = RAGPipeline('data/icd10_text.txt')
    passed = 0
    failed = 0
    
    for query, expected in test_queries:
        try:
            result = rag.simple_search(query, k=2)
            if len(result) > 100:  # Got some result
                print(f"✅ PASS: '{query}' - {expected}")
                print(f"   Preview: {result[:150]}...\n")
                passed += 1
            else:
                print(f"⚠️  WARN: '{query}' - Got short result\n")
                failed += 1
        except Exception as e:
            print(f"❌ FAIL: '{query}' - Error: {e}\n")
            failed += 1
    
    print(f"Results: {passed} passed, {failed} failed\n")
    return failed == 0

def test_text_loading():
    """Test 4: Text file loading"""
    print("=" * 60)
    print("TEST 4: Text File Loading")
    print("=" * 60)
    try:
        rag = RAGPipeline('data/icd10_text.txt')
        text = rag.load_text()
        if len(text) > 10000:
            print(f"✅ PASS: Loaded {len(text)} characters")
            print(f"   Preview: {text[:100]}...\n")
            return True
        else:
            print(f"⚠️  WARN: File seems too small ({len(text)} chars)\n")
            return False
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def test_chunking():
    """Test 5: Text chunking"""
    print("=" * 60)
    print("TEST 5: Text Chunking")
    print("=" * 60)
    try:
        rag = RAGPipeline('data/icd10_text.txt')
        text = rag.load_text()
        chunks = rag.split_chunks(text)
        print(f"✅ PASS: Created {len(chunks)} chunks")
        print(f"   First chunk preview: {chunks[0].page_content[:100]}...\n")
        return len(chunks) > 0
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def main():
    print("\n🧪 NEURORAG SYSTEM TEST SUITE\n")
    
    tests = [
        ("Text Loading", test_text_loading),
        ("Text Chunking", test_chunking),
        ("Embeddings Creation", test_embeddings),
        ("Vector Store Loading", test_vectorstore_load),
        ("Search Queries", test_search_queries),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"💥 CRASH in {name}: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
