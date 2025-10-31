"""
NeuroRAG System Test Script
Tests all components of the application
"""

import os
import sys

def test_imports():
    """Test if all required imports work"""
    print("=" * 60)
    print("🧪 Testing Imports...")
    print("=" * 60)
    
    try:
        print("✓ Testing Flask...")
        from flask import Flask, jsonify
        
        print("✓ Testing LangChain components...")
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        print("✓ Testing RAG pipeline...")
        from src.rag_pipeline import RAGPipeline
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_files_exist():
    """Test if all required files exist"""
    print("\n" + "=" * 60)
    print("📁 Testing File Structure...")
    print("=" * 60)
    
    # Navigate to parent directory for correct paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    os.chdir(base_dir)
    
    required_files = [
        "run_server.py",
        "src/rag_pipeline.py",
        "data/icd10_text.txt",
        "faiss_index/index.faiss",
        "templates/index.html",
        "static/style.css",
        "static/script.js",
        "requirements.txt"
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files exist!")
    else:
        print("\n⚠️ Some files are missing!")
    
    return all_exist

def test_rag_pipeline():
    """Test if RAG pipeline can be loaded"""
    print("\n" + "=" * 60)
    print("🤖 Testing RAG Pipeline...")
    print("=" * 60)
    
    try:
        from src.rag_pipeline import RAGPipeline
        
        print("Creating RAG instance...")
        rag = RAGPipeline(doc_path="data/icd10_text.txt")
        
        print("Loading vector store...")
        rag.load_vectorstore()
        
        print("Testing simple search...")
        result = rag.simple_search("depression", k=1)
        
        print(f"\nSearch result preview (first 100 chars):")
        print(result[:100] + "...")
        
        print("\n✅ RAG Pipeline working!")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_routes():
    """Test Flask application routes"""
    print("\n" + "=" * 60)
    print("🌐 Testing Flask Routes...")
    print("=" * 60)
    
    try:
        # Import the Flask app
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        
        from run_server import app
        
        with app.test_client() as client:
            # Test health endpoint
            print("Testing /health endpoint...")
            response = client.get('/health')
            assert response.status_code == 200
            print(f"✓ Health check: {response.json}")
            
            # Test stats endpoint
            print("\nTesting /api/stats endpoint...")
            response = client.get('/api/stats')
            assert response.status_code == 200
            print(f"✓ Stats: {response.json}")
            
            # Test index page
            print("\nTesting / (dashboard) endpoint...")
            response = client.get('/')
            assert response.status_code == 200
            print("✓ Dashboard loads successfully")
        
        print("\n✅ All Flask routes working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Flask routes error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    # Navigate to project root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    os.chdir(base_dir)
    sys.path.insert(0, base_dir)
    
    print("\n" + "=" * 60)
    print("🧠 NeuroRAG System Test Suite")
    print("=" * 60)
    print()
    
    results = {
        "Imports": test_imports(),
        "Files": test_files_exist(),
        "RAG Pipeline": test_rag_pipeline(),
        "Flask Routes": test_flask_routes()
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ NeuroRAG is ready to use!")
        print("\nTo start the server, run: python run_server.py")
        print("Or double-click: scripts\\START_SERVER.bat")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please check the errors above and fix them.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
