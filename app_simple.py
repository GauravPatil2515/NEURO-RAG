"""
Simple Flask Backend for NeuroRAG - Minimal Version
Tests basic functionality without loading heavy dependencies
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurorag-secret-key'

# Lazy loading to avoid startup issues
rag = None

def get_rag():
    """Lazy load RAG pipeline only when needed"""
    global rag
    if rag is None:
        try:
            # Only import when actually needed
            from rag_pipeline import RAGPipeline
            rag = RAGPipeline(doc_path="data/icd10_text.txt")
            print("🔍 Loading vector store...")
            rag.load_vectorstore()
            print("✅ Vector store loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading RAG: {e}")
            return None
    return rag

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for searching ICD-10 information"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get RAG instance and perform search
        rag_instance = get_rag()
        if rag_instance is None:
            return jsonify({
                'success': False,
                'error': 'RAG pipeline not initialized. Please check server logs.'
            }), 500
        
        result = rag_instance.simple_search(query, k=3)
        
        return jsonify({
            'success': True,
            'query': query,
            'result': result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        global rag
        # Check if vector store is loaded
        is_loaded = rag is not None and rag.vectorstore is not None
        
        # Get document stats
        text_file_exists = os.path.exists('data/icd10_text.txt')
        index_exists = os.path.exists('faiss_index/index.faiss')
        
        return jsonify({
            'vectorstore_loaded': is_loaded,
            'data_file_exists': text_file_exists,
            'index_exists': index_exists,
            'status': 'online'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'NeuroRAG is running!'})

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 NeuroRAG Flask Server Starting...")
    print("=" * 60)
    print("📍 Dashboard URL: http://localhost:5000")
    print("📍 Alternative:   http://127.0.0.1:5000")
    print("=" * 60)
    print("⚠️  Keep this window open!")
    print("⚠️  Press CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    # Set environment variables to avoid issues
    os.environ['USE_TF'] = '0'
    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Run on localhost only, port 5000
        app.run(
            debug=True,
            host='127.0.0.1',  # Using 127.0.0.1 explicitly
            port=5000,
            use_reloader=False,  # Disable reloader to avoid double startup
            threaded=True
        )
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nPress Enter to exit...")
        input()
