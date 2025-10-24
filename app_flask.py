"""
Flask Backend for NeuroRAG
Simple API for ICD-10 Mental Health Question Answering
"""

from flask import Flask, render_template, request, jsonify
from rag_pipeline import RAGPipeline
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurorag-secret-key'

# Initialize RAG pipeline (lazy loading)
rag = None

def get_rag():
    """Lazy load RAG pipeline"""
    global rag
    if rag is None:
        rag = RAGPipeline(doc_path="data/icd10_text.txt")
        try:
            print("🔍 Loading vector store...")
            rag.load_vectorstore()
            print("✅ Vector store loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading vector store: {e}")
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
        result = rag_instance.simple_search(query, k=3)
        
        return jsonify({
            'success': True,
            'query': query,
            'result': result
        })
        
    except Exception as e:
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

if __name__ == '__main__':
    print("🧠 Starting NeuroRAG Flask Server...")
    app.run(debug=False, host='0.0.0.0', port=5000)
