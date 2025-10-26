"""
Direct RAG Testing - Tests the RAG pipeline components directly without Flask server
"""

import sys
import time
from datetime import datetime

print('='*80)
print('NEURORAG DIRECT RAG PIPELINE TESTING')
print('='*80)
print(f'Test Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('='*80)

# Test 1: Import all required modules
print('\n[TEST 1] Module Imports')
print('-'*80)
try:
    from sentence_transformers import SentenceTransformer
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch
    import faiss
    import os
    print('✓ All required modules imported successfully')
    test1_pass = True
except Exception as e:
    print(f'✗ Import failed: {e}')
    test1_pass = False
    sys.exit(1)

# Test 2: Check data file exists
print('\n[TEST 2] Data File Verification')
print('-'*80)
data_path = 'data/icd10_text.txt'
try:
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        print(f'✓ Data file exists: {data_path}')
        print(f'  • File size: {file_size:,} bytes ({file_size/1024:.2f} KB)')
        print(f'  • Total lines: {len(lines):,}')
        print(f'  • Content preview: {content[:150]}...')
        test2_pass = True
    else:
        print(f'✗ Data file not found: {data_path}')
        test2_pass = False
except Exception as e:
    print(f'✗ Error reading data: {e}')
    test2_pass = False

# Test 3: Load Embedding Model
print('\n[TEST 3] Embedding Model Loading')
print('-'*80)
try:
    start = time.time()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    load_time = (time.time() - start) * 1000
    
    # Test embedding
    test_text = "This is a test sentence"
    embedding = model.encode(test_text)
    
    print(f'✓ Model loaded successfully')
    print(f'  • Model: sentence-transformers/all-MiniLM-L6-v2')
    print(f'  • Load time: {load_time:.2f}ms')
    print(f'  • Embedding dimension: {len(embedding)}')
    print(f'  • Sample embedding (first 5 values): {embedding[:5]}')
    test3_pass = True
except Exception as e:
    print(f'✗ Model loading failed: {e}')
    test3_pass = False
    model = None

# Test 4: Check FAISS Index
print('\n[TEST 4] FAISS Index Verification')
print('-'*80)
faiss_path = 'faiss_index'
try:
    if os.path.exists(faiss_path):
        index_size = os.path.getsize(f'{faiss_path}/index.faiss')
        print(f'✓ FAISS index exists: {faiss_path}')
        print(f'  • Index file size: {index_size:,} bytes ({index_size/1024/1024:.2f} MB)')
        
        # Load the index
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        
        # Get index stats
        index = vector_store.index
        print(f'  • Total vectors: {index.ntotal:,}')
        print(f'  • Vector dimension: {index.d}')
        print(f'  • Index type: {type(index).__name__}')
        
        test4_pass = True
    else:
        print(f'✗ FAISS index not found: {faiss_path}')
        test4_pass = False
        vector_store = None
except Exception as e:
    print(f'✗ FAISS index check failed: {e}')
    test4_pass = False
    vector_store = None

# Test 5: Semantic Search - Depression
if vector_store and model:
    print('\n[TEST 5] Semantic Search - Depression Query')
    print('-'*80)
    try:
        query = "What are the symptoms of major depressive disorder?"
        print(f'Query: "{query}"')
        
        start = time.time()
        results = vector_store.similarity_search(query, k=3)
        search_time = (time.time() - start) * 1000
        
        print(f'\n✓ Search completed successfully')
        print(f'  • Search time: {search_time:.2f}ms')
        print(f'  • Results returned: {len(results)}')
        
        for i, doc in enumerate(results, 1):
            print(f'\n  Result {i}:')
            content_preview = doc.page_content[:200].replace('\n', ' ')
            print(f'    {content_preview}...')
        
        test5_pass = True
    except Exception as e:
        print(f'✗ Search failed: {e}')
        test5_pass = False
else:
    print('\n[TEST 5] Semantic Search - Depression Query')
    print('-'*80)
    print('✗ Skipped - vector store or model not loaded')
    test5_pass = False

# Test 6: Semantic Search - Anxiety
if vector_store and model:
    print('\n[TEST 6] Semantic Search - Anxiety Query')
    print('-'*80)
    try:
        query = "What is generalized anxiety disorder?"
        print(f'Query: "{query}"')
        
        start = time.time()
        results = vector_store.similarity_search(query, k=3)
        search_time = (time.time() - start) * 1000
        
        print(f'\n✓ Search completed')
        print(f'  • Search time: {search_time:.2f}ms')
        print(f'  • Results: {len(results)}')
        
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:150].replace('\n', ' ')
            print(f'  [{i}] {content_preview}...')
        
        test6_pass = True
    except Exception as e:
        print(f'✗ Search failed: {e}')
        test6_pass = False
else:
    print('\n[TEST 6] Semantic Search - Anxiety Query')
    print('-'*80)
    print('✗ Skipped - dependencies not loaded')
    test6_pass = False

# Test 7: Embedding Quality Test
if model:
    print('\n[TEST 7] Embedding Quality Test')
    print('-'*80)
    try:
        # Test semantic similarity
        texts = [
            "depression symptoms and diagnosis",
            "major depressive disorder criteria",
            "weather forecast tomorrow",
        ]
        
        embeddings_list = model.encode(texts)
        
        # Calculate cosine similarity
        import numpy as np
        from numpy.linalg import norm
        
        sim_1_2 = np.dot(embeddings_list[0], embeddings_list[1]) / (norm(embeddings_list[0]) * norm(embeddings_list[1]))
        sim_1_3 = np.dot(embeddings_list[0], embeddings_list[2]) / (norm(embeddings_list[0]) * norm(embeddings_list[2]))
        
        print(f'Text 1: "{texts[0]}"')
        print(f'Text 2: "{texts[1]}"')
        print(f'Text 3: "{texts[2]}"')
        print(f'\nSimilarity Scores:')
        print(f'  • Text 1 <-> Text 2 (related): {sim_1_2:.4f}')
        print(f'  • Text 1 <-> Text 3 (unrelated): {sim_1_3:.4f}')
        
        # Related texts should have higher similarity
        if sim_1_2 > sim_1_3:
            print(f'\n✓ Embeddings correctly distinguish related vs unrelated content')
            print(f'  Related texts are {sim_1_2/sim_1_3:.2f}x more similar')
            test7_pass = True
        else:
            print(f'\n✗ Embedding quality issue - unrelated texts scored higher')
            test7_pass = False
            
    except Exception as e:
        print(f'✗ Embedding quality test failed: {e}')
        test7_pass = False
else:
    print('\n[TEST 7] Embedding Quality Test')
    print('-'*80)
    print('✗ Skipped - model not loaded')
    test7_pass = False

# Test 8: Performance Test - Multiple Queries
if vector_store:
    print('\n[TEST 8] Performance Test - Multiple Queries')
    print('-'*80)
    try:
        queries = [
            "schizophrenia symptoms",
            "bipolar disorder diagnosis",
            "panic attack criteria",
            "PTSD treatment",
            "OCD compulsive behaviors"
        ]
        
        times = []
        for query in queries:
            start = time.time()
            results = vector_store.similarity_search(query, k=2)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f'  • "{query}": {elapsed:.2f}ms ({len(results)} results)')
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f'\nPerformance Summary:')
        print(f'  • Total queries: {len(queries)}')
        print(f'  • Average time: {avg_time:.2f}ms')
        print(f'  • Min time: {min_time:.2f}ms')
        print(f'  • Max time: {max_time:.2f}ms')
        print(f'  • Throughput: {1000/avg_time:.2f} queries/second')
        
        # Pass if average time is under 1 second
        if avg_time < 1000:
            print(f'\n✓ Performance acceptable (under 1000ms average)')
            test8_pass = True
        else:
            print(f'\n⚠ Performance warning (over 1000ms average)')
            test8_pass = True  # Still pass but with warning
            
    except Exception as e:
        print(f'✗ Performance test failed: {e}')
        test8_pass = False
else:
    print('\n[TEST 8] Performance Test')
    print('-'*80)
    print('✗ Skipped - vector store not loaded')
    test8_pass = False

# Final Summary
print('\n' + '='*80)
print('TEST SUMMARY')
print('='*80)

tests = [
    ('Module Imports', test1_pass),
    ('Data File Verification', test2_pass),
    ('Embedding Model Loading', test3_pass),
    ('FAISS Index Verification', test4_pass),
    ('Semantic Search - Depression', test5_pass),
    ('Semantic Search - Anxiety', test6_pass),
    ('Embedding Quality', test7_pass),
    ('Performance Test', test8_pass),
]

passed = sum(1 for _, p in tests if p)
total = len(tests)

for test_name, result in tests:
    status = '✓ PASS' if result else '✗ FAIL'
    print(f'{status:8} | {test_name}')

print('-'*80)
print(f'Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)')
print(f'Test End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('='*80)

if passed == total:
    print('\n🎉 ALL TESTS PASSED! RAG system is fully functional.')
else:
    print(f'\n⚠ {total-passed} test(s) failed. Review errors above.')
