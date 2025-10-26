"""
NeuroRAG Comprehensive Testing Suite
Tests RAG pipeline, embeddings, FAISS, API endpoints, and real-world scenarios
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = 'http://127.0.0.1:5000'

def print_header(title):
    print('\n' + '='*80)
    print(f' {title}')
    print('='*80)

def print_test(test_name):
    print(f'\n[TEST] {test_name}')
    print('-'*80)

def print_result(passed, message=''):
    status = '✓ PASSED' if passed else '✗ FAILED'
    print(f'{status} {message}')

# =============================================================================
# TEST 1: SERVER HEALTH & AVAILABILITY
# =============================================================================
def test_server_health():
    print_test('Server Health Check')
    try:
        start = time.time()
        response = requests.get(f'{BASE_URL}/health', timeout=5)
        elapsed = (time.time() - start) * 1000
        
        print(f'Status Code: {response.status_code}')
        print(f'Response Time: {elapsed:.2f}ms')
        print(f'Response: {json.dumps(response.json(), indent=2)}')
        
        passed = response.status_code == 200
        print_result(passed, f'Response time: {elapsed:.2f}ms')
        return passed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False

# =============================================================================
# TEST 2: SYSTEM STATISTICS & METADATA
# =============================================================================
def test_system_stats():
    print_test('System Statistics API')
    try:
        start = time.time()
        response = requests.get(f'{BASE_URL}/api/stats', timeout=5)
        elapsed = (time.time() - start) * 1000
        stats = response.json()
        
        print(f'Status Code: {response.status_code}')
        print(f'Response Time: {elapsed:.2f}ms')
        print(f'\nSystem Information:')
        print(f'  • Database Size: {stats.get("database_size", "N/A")}')
        print(f'  • Total Documents: {stats.get("total_documents", "N/A")}')
        print(f'  • Index Type: {stats.get("index_type", "N/A")}')
        print(f'  • Embedding Model: {stats.get("embedding_model", "N/A")}')
        
        passed = response.status_code == 200 and stats.get('database_size') is not None
        print_result(passed, f'All stats retrieved successfully')
        return passed, stats
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False, {}

# =============================================================================
# TEST 3: RAG SEMANTIC SEARCH - DEPRESSION
# =============================================================================
def test_rag_depression():
    print_test('RAG Query: Major Depressive Disorder')
    query = 'What are the diagnostic criteria for major depressive disorder?'
    
    try:
        print(f'Query: "{query}"')
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/api/search',
            json={'query': query},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        result = response.json()
        
        print(f'\nResponse Details:')
        print(f'  • Status Code: {response.status_code}')
        print(f'  • Response Time: {elapsed:.2f}ms')
        print(f'  • Answer Length: {len(result.get("answer", ""))} characters')
        
        answer = result.get('answer', '')
        print(f'\nAnswer Preview (first 300 chars):')
        print(f'{answer[:300]}...')
        
        # Check for relevant keywords
        keywords = ['depressive', 'disorder', 'mood', 'depression', 'symptom']
        found_keywords = [kw for kw in keywords if kw.lower() in answer.lower()]
        print(f'\nRelevant Keywords Found: {found_keywords}')
        
        passed = (response.status_code == 200 and 
                 len(answer) > 50 and 
                 len(found_keywords) >= 2)
        print_result(passed, f'Query time: {elapsed:.2f}ms, Keywords: {len(found_keywords)}/5')
        return passed, elapsed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False, 0

# =============================================================================
# TEST 4: RAG SEMANTIC SEARCH - ANXIETY
# =============================================================================
def test_rag_anxiety():
    print_test('RAG Query: Anxiety Disorders')
    query = 'What is generalized anxiety disorder?'
    
    try:
        print(f'Query: "{query}"')
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/api/search',
            json={'query': query},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        result = response.json()
        
        print(f'\nResponse Details:')
        print(f'  • Status Code: {response.status_code}')
        print(f'  • Response Time: {elapsed:.2f}ms')
        print(f'  • Answer Length: {len(result.get("answer", ""))} characters')
        
        answer = result.get('answer', '')
        print(f'\nAnswer Preview:')
        print(f'{answer[:250]}...')
        
        keywords = ['anxiety', 'worry', 'disorder', 'fear', 'generalized']
        found_keywords = [kw for kw in keywords if kw.lower() in answer.lower()]
        print(f'\nRelevant Keywords Found: {found_keywords}')
        
        passed = (response.status_code == 200 and 
                 len(answer) > 50 and 
                 len(found_keywords) >= 2)
        print_result(passed, f'Query time: {elapsed:.2f}ms')
        return passed, elapsed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False, 0

# =============================================================================
# TEST 5: RAG SEMANTIC SEARCH - SCHIZOPHRENIA
# =============================================================================
def test_rag_schizophrenia():
    print_test('RAG Query: Schizophrenia')
    query = 'What are the symptoms of schizophrenia?'
    
    try:
        print(f'Query: "{query}"')
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/api/search',
            json={'query': query},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        result = response.json()
        
        print(f'\nResponse Details:')
        print(f'  • Status Code: {response.status_code}')
        print(f'  • Response Time: {elapsed:.2f}ms')
        print(f'  • Answer Length: {len(result.get("answer", ""))} characters')
        
        answer = result.get('answer', '')
        print(f'\nFull Answer:')
        print(f'{answer}')
        
        keywords = ['schizophrenia', 'psychotic', 'hallucination', 'delusion', 'symptom']
        found_keywords = [kw for kw in keywords if kw.lower() in answer.lower()]
        print(f'\nRelevant Keywords Found: {found_keywords}')
        
        passed = (response.status_code == 200 and 
                 len(answer) > 50 and 
                 len(found_keywords) >= 1)
        print_result(passed, f'Query time: {elapsed:.2f}ms')
        return passed, elapsed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False, 0

# =============================================================================
# TEST 6: RAG SEMANTIC SEARCH - OCD
# =============================================================================
def test_rag_ocd():
    print_test('RAG Query: Obsessive-Compulsive Disorder')
    query = 'What is OCD and how is it diagnosed?'
    
    try:
        print(f'Query: "{query}"')
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/api/search',
            json={'query': query},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        result = response.json()
        
        print(f'\nResponse Details:')
        print(f'  • Status Code: {response.status_code}')
        print(f'  • Response Time: {elapsed:.2f}ms')
        print(f'  • Answer Length: {len(result.get("answer", ""))} characters')
        
        answer = result.get('answer', '')
        print(f'\nAnswer:')
        print(f'{answer}')
        
        keywords = ['obsessive', 'compulsive', 'disorder', 'ocd', 'ritual']
        found_keywords = [kw for kw in keywords if kw.lower() in answer.lower()]
        print(f'\nRelevant Keywords Found: {found_keywords}')
        
        passed = (response.status_code == 200 and 
                 len(answer) > 50 and 
                 len(found_keywords) >= 1)
        print_result(passed, f'Query time: {elapsed:.2f}ms')
        return passed, elapsed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False, 0

# =============================================================================
# TEST 7: EDGE CASE - EMPTY QUERY
# =============================================================================
def test_edge_empty_query():
    print_test('Edge Case: Empty Query')
    try:
        response = requests.post(
            f'{BASE_URL}/api/search',
            json={'query': ''},
            timeout=10
        )
        result = response.json()
        
        print(f'Status Code: {response.status_code}')
        print(f'Response: {json.dumps(result, indent=2)}')
        
        # Should handle gracefully
        passed = response.status_code in [200, 400]
        print_result(passed, 'Empty query handled gracefully')
        return passed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False

# =============================================================================
# TEST 8: EDGE CASE - VERY LONG QUERY
# =============================================================================
def test_edge_long_query():
    print_test('Edge Case: Very Long Query')
    long_query = 'What are the symptoms ' * 50  # Very long repetitive query
    
    try:
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/api/search',
            json={'query': long_query},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        
        print(f'Query Length: {len(long_query)} characters')
        print(f'Status Code: {response.status_code}')
        print(f'Response Time: {elapsed:.2f}ms')
        
        passed = response.status_code == 200
        print_result(passed, 'Long query handled successfully')
        return passed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False

# =============================================================================
# TEST 9: PERFORMANCE - MULTIPLE RAPID QUERIES
# =============================================================================
def test_performance_rapid_queries():
    print_test('Performance: Rapid Consecutive Queries')
    queries = [
        'depression symptoms',
        'anxiety treatment',
        'bipolar disorder'
    ]
    
    times = []
    try:
        for i, query in enumerate(queries, 1):
            print(f'\nQuery {i}/{len(queries)}: "{query}"')
            start = time.time()
            response = requests.post(
                f'{BASE_URL}/api/search',
                json={'query': query},
                timeout=30
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f'  Time: {elapsed:.2f}ms, Status: {response.status_code}')
        
        avg_time = sum(times) / len(times)
        print(f'\nPerformance Summary:')
        print(f'  • Total Queries: {len(queries)}')
        print(f'  • Average Time: {avg_time:.2f}ms')
        print(f'  • Min Time: {min(times):.2f}ms')
        print(f'  • Max Time: {max(times):.2f}ms')
        
        passed = all(t < 5000 for t in times)  # All queries under 5 seconds
        print_result(passed, f'Average query time: {avg_time:.2f}ms')
        return passed, avg_time
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False, 0

# =============================================================================
# TEST 10: DATABASE ENDPOINT
# =============================================================================
def test_database_endpoint():
    print_test('Database Content Retrieval')
    try:
        start = time.time()
        response = requests.get(f'{BASE_URL}/api/database', timeout=10)
        elapsed = (time.time() - start) * 1000
        result = response.json()
        
        print(f'Status Code: {response.status_code}')
        print(f'Response Time: {elapsed:.2f}ms')
        
        content = result.get('content', '')
        print(f'Database Content Length: {len(content)} characters')
        print(f'Preview: {content[:200]}...')
        
        passed = response.status_code == 200 and len(content) > 100
        print_result(passed, f'Database retrieved: {len(content)} chars')
        return passed
    except Exception as e:
        print_result(False, f'Error: {e}')
        return False

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================
def run_all_tests():
    print_header('NEURORAG COMPREHENSIVE TESTING SUITE')
    print(f'Test Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Server URL: {BASE_URL}')
    
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'response_times': []
    }
    
    # Run all tests
    tests = [
        ('Server Health', test_server_health),
        ('System Stats', test_system_stats),
        ('RAG - Depression', test_rag_depression),
        ('RAG - Anxiety', test_rag_anxiety),
        ('RAG - Schizophrenia', test_rag_schizophrenia),
        ('RAG - OCD', test_rag_ocd),
        ('Edge - Empty Query', test_edge_empty_query),
        ('Edge - Long Query', test_edge_long_query),
        ('Performance - Rapid', test_performance_rapid_queries),
        ('Database Endpoint', test_database_endpoint),
    ]
    
    for test_name, test_func in tests:
        results['total'] += 1
        try:
            test_result = test_func()
            if isinstance(test_result, tuple):
                passed = test_result[0]
                if len(test_result) > 1:
                    results['response_times'].append(test_result[1])
            else:
                passed = test_result
            
            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            print(f'\n✗ Test "{test_name}" crashed: {e}')
            results['failed'] += 1
        
        time.sleep(0.5)  # Small delay between tests
    
    # Final Summary
    print_header('TEST SUMMARY')
    print(f'Total Tests: {results["total"]}')
    print(f'Passed: {results["passed"]} ✓')
    print(f'Failed: {results["failed"]} ✗')
    print(f'Success Rate: {(results["passed"]/results["total"]*100):.1f}%')
    
    if results['response_times']:
        avg_response = sum(results['response_times']) / len(results['response_times'])
        print(f'\nAverage Response Time: {avg_response:.2f}ms')
        print(f'Fastest Response: {min(results["response_times"]):.2f}ms')
        print(f'Slowest Response: {max(results["response_times"]):.2f}ms')
    
    print('\n' + '='*80)
    print(f'Test End Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*80 + '\n')
    
    return results

if __name__ == '__main__':
    run_all_tests()
