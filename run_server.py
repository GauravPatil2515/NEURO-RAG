"""
NeuroRAG Flask Application - Production Ready
Fixed version without spacy import issues
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Prevent spacy from being imported during langchain initialization
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurorag-secret-key-2025'

# Global RAG instance
rag = None
use_ai_mode = False  # Toggle for Phi-3-Mini (disabled - use Fast Mode for stability)

def get_rag():
    """Lazy load RAG pipeline only when needed"""
    global rag
    if rag is None:
        try:
            print("🔄 Initializing RAG pipeline...")
            from src.rag_pipeline import RAGPipeline
            rag = RAGPipeline(doc_path="data/icd10_text.txt")
            
            # Check if vector store exists
            if os.path.exists("faiss_index/index.faiss"):
                print("📂 Loading existing vector store...")
                rag.load_vectorstore()
                print("✅ Vector store loaded successfully!")
                
                # Try to load Phi-3-Mini if requested
                global use_ai_mode
                if use_ai_mode:
                    print("\n🤖 Attempting to load Phi-3-Mini for AI-powered answers...")
                    success = rag.setup_phi3_mini()
                    if success:
                        print("✅ AI Mode: ENABLED (Phi-3-Mini)")
                    else:
                        print("⚠️  AI Mode: DISABLED (using fast retrieval)")
                else:
                    print("⚡ Fast Mode: ENABLED (retrieval-only, instant results)")
            else:
                print("⚠️ No vector store found. Please build it first using test_system.py")
                return None
                
        except Exception as e:
            print(f"❌ Error loading RAG: {e}")
            import traceback
            traceback.print_exc()
            return None
    return rag

@app.route('/')
def home():
    """Render home page"""
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for searching ICD-10 information"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        use_ai = data.get('use_ai', use_ai_mode)  # Allow per-request AI toggle
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get RAG instance
        rag_instance = get_rag()
        if rag_instance is None:
            return jsonify({
                'success': False,
                'error': 'RAG pipeline not initialized. Vector store may not be built yet.'
            }), 500
        
        # Perform search
        print(f"🔍 Searching for: {query}")
        
        # Use smart search if AI is enabled and available
        if use_ai and rag_instance.use_phi3:
            result_data = rag_instance.smart_search(query, k=3)
            return jsonify({
                'success': True,
                'query': query,
                'answer': result_data['answer'],
                'sources': result_data.get('sources', []),
                'mode': result_data.get('mode', 'retrieval')
            })
        else:
            # Fast retrieval-only mode
            result = rag_instance.simple_search(query, k=3)
            return jsonify({
                'success': True,
                'query': query,
                'answer': result,
                'mode': 'retrieval'
            })
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Search failed: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        global rag
        
        # Check if vector store is loaded
        is_loaded = rag is not None and hasattr(rag, 'vectorstore') and rag.vectorstore is not None
        
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

@app.route('/api/database', methods=['GET'])
def database():
    """Get database content"""
    try:
        # Read the ICD-10 text file
        data_path = 'data/icd10_text.txt'
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'error': 'Database file not found'
            }), 404
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Return full content for searching and viewing
        return jsonify({
            'success': True,
            'content': content,
            'total_length': len(content),
            'lines': len(content.split('\n'))
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'NeuroRAG server is running',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🧠 NeuroRAG - Mental Health AI Assistant")
    print("=" * 70)
    print("📍 Dashboard URL:  http://localhost:5000")
    print("📍 Alternative:    http://127.0.0.1:5000")
    print("=" * 70)
    print("⚠️  Keep this terminal window OPEN while using the app")
    print("⚠️  Press CTRL+C to stop the server")
    print("=" * 70)
    print()
    
    try:
        app.run(
            debug=True,
            host='127.0.0.1',
            port=5000,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
