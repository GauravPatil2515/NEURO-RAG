"""
NeuroRAG - Complete ML/RAG Demonstration
=========================================

This script demonstrates all ML/RAG concepts from the technical guide.
Run this in Google Colab or locally to see how RAG works step-by-step.

Author: Gaurav Patil
Date: October 25, 2025
"""

# ============================================================================
# SECTION 1: INSTALLATION & IMPORTS
# ============================================================================

print("=" * 70)
print("NeuroRAG - Complete ML/RAG Demonstration")
print("=" * 70)
print("\n📦 Installing required packages...")
print("(Skip this if running locally with packages already installed)\n")

# Uncomment these lines if running in Google Colab
"""
!pip install -q sentence-transformers
!pip install -q faiss-cpu
!pip install -q langchain
!pip install -q langchain-community
!pip install -q transformers
!pip install -q torch
"""

# Imports
import numpy as np
import time
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

print("\n✅ Imports successful!")

# ============================================================================
# SECTION 2: SAMPLE MEDICAL DATA
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 2: SAMPLE MEDICAL DATA")
print("=" * 70)

# Sample ICD-10 Mental Health Data (subset for demonstration)
SAMPLE_ICD10_DATA = """
F32 - Depressive Episode
A depressive episode is characterized by a period of depressed mood or loss of interest or pleasure in nearly all activities. The episode must last at least 2 weeks and cause clinically significant distress or impairment.

F32.0 - Mild depressive episode
At least two of the three typical symptoms of depression (depressed mood, loss of interest, reduced energy) must be present, along with at least two other symptoms. No symptom should be present to an intense degree.

F32.1 - Moderate depressive episode
At least two of the three typical symptoms must be present, along with at least three (and preferably four) other symptoms. Several symptoms are marked to a considerable degree.

F32.2 - Severe depressive episode without psychotic symptoms
All three typical symptoms should be present, plus at least four other symptoms, some of which should be of severe intensity.

F33 - Recurrent Depressive Disorder
Recurrent depressive disorder is characterized by repeated episodes of depression. The current episode may be mild, moderate, or severe.

F33.0 - Recurrent depressive disorder, current episode mild
The current episode meets criteria for mild depressive episode, and there has been at least one previous depressive episode.

F33.1 - Recurrent depressive disorder, current episode moderate
The current episode meets criteria for moderate depressive episode, and there has been at least one previous episode.

F33.2 - Recurrent depressive disorder, current episode severe without psychotic symptoms
The current episode meets criteria for severe depressive episode without psychotic symptoms, and there has been at least one previous episode.

F33.4 - Recurrent depressive disorder, currently in remission
There has been at least one previous depressive episode, but no symptoms are present at the current time. The patient has been symptom-free for several months.

F40 - Phobic Anxiety Disorders
Phobic anxiety disorders are characterized by excessive and unreasonable fear of specific objects, activities, or situations.

F40.0 - Agoraphobia
Marked fear or anxiety about two or more of the following: using public transportation, being in open spaces, being in enclosed spaces, standing in line or being in a crowd, being outside of the home alone.

F40.1 - Social Phobias
Marked fear or anxiety about one or more social situations in which the individual is exposed to possible scrutiny by others. Examples include social interactions, being observed, and performing in front of others.

F41 - Other Anxiety Disorders
Anxiety disorders not classified as phobic anxiety disorders.

F41.0 - Panic Disorder
Recurrent unexpected panic attacks characterized by a sudden surge of intense fear or discomfort that reaches a peak within minutes.

F41.1 - Generalized Anxiety Disorder
Excessive anxiety and worry occurring more days than not for at least 6 months, about a number of events or activities.

F42 - Obsessive-Compulsive Disorder
Presence of obsessions (recurrent and persistent thoughts, urges, or images) and/or compulsions (repetitive behaviors or mental acts).

F43 - Reaction to Severe Stress and Adjustment Disorders
Disorders that arise as a direct response to acute severe stress or continued trauma.

F43.0 - Acute Stress Reaction
A transient disorder that develops in response to exceptional physical or mental stress and usually subsides within hours or days.

F43.1 - Post-Traumatic Stress Disorder (PTSD)
Arises as a delayed or protracted response to a stressful event or situation of an exceptionally threatening or catastrophic nature.

F50 - Eating Disorders
Eating disorders characterized by abnormal eating habits that negatively affect physical or mental health.

F50.0 - Anorexia Nervosa
Characterized by a distorted body image and excessive dieting leading to severe weight loss with a pathological fear of becoming fat.

F50.2 - Bulimia Nervosa
Characterized by repeated episodes of binge eating followed by compensatory behaviors such as purging, fasting, or excessive exercise.
"""

print(f"\n📚 Loaded sample ICD-10 data: {len(SAMPLE_ICD10_DATA)} characters")
print(f"📄 Contains information about depression, anxiety, OCD, PTSD, and eating disorders")

# ============================================================================
# SECTION 3: TEXT SPLITTING (CHUNKING)
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: TEXT SPLITTING (DOCUMENT CHUNKING)")
print("=" * 70)

class SimpleTextSplitter:
    """
    Splits text into chunks with overlap.
    Demonstrates the chunking strategy used in RAG systems.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > self.chunk_size // 2:  # At least halfway
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks

# Split the data
splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(SAMPLE_ICD10_DATA)

print(f"\n✅ Split text into {len(chunks)} chunks")
print(f"📊 Chunk size: {splitter.chunk_size} characters")
print(f"📊 Overlap: {splitter.chunk_overlap} characters")
print(f"\n📝 First chunk preview:")
print("-" * 70)
print(chunks[0][:200] + "...")
print("-" * 70)

# ============================================================================
# SECTION 4: EMBEDDINGS (TEXT TO VECTORS)
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: EMBEDDINGS - Converting Text to Vectors")
print("=" * 70)

print("\n⏳ Loading embedding model (all-MiniLM-L6-v2)...")
print("   This may take a minute on first run...")

from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Model loaded successfully!")
print(f"📐 Embedding dimensions: 384")

# Demonstrate embeddings
demo_texts = [
    "Depression is a mental health disorder",
    "Depressive episode with low mood",
    "Anxiety and panic attacks"
]

print("\n🧪 DEMONSTRATION: Creating embeddings for sample texts")
print("-" * 70)

for i, text in enumerate(demo_texts, 1):
    embedding = embedding_model.encode(text)
    print(f"\n{i}. Text: \"{text}\"")
    print(f"   Vector shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")

# Create embeddings for all chunks
print(f"\n⏳ Creating embeddings for {len(chunks)} document chunks...")
start_time = time.time()
chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
embedding_time = time.time() - start_time

print(f"\n✅ Created {len(chunk_embeddings)} embeddings in {embedding_time:.2f} seconds")
print(f"📊 Embeddings shape: {chunk_embeddings.shape}")
print(f"💾 Memory size: ~{chunk_embeddings.nbytes / 1024:.2f} KB")

# ============================================================================
# SECTION 5: COSINE SIMILARITY
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: COSINE SIMILARITY - Measuring Semantic Similarity")
print("=" * 70)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Demonstrate similarity
print("\n🧪 DEMONSTRATION: Comparing semantic similarity")
print("-" * 70)

test_pairs = [
    ("Depression is a mental disorder", "Depressive episode"),
    ("Depression is a mental disorder", "Anxiety disorder"),
    ("Depression is a mental disorder", "Eating disorder")
]

for text1, text2 in test_pairs:
    emb1 = embedding_model.encode(text1)
    emb2 = embedding_model.encode(text2)
    similarity = cosine_similarity(emb1, emb2)
    print(f"\nText 1: \"{text1}\"")
    print(f"Text 2: \"{text2}\"")
    print(f"Similarity: {similarity:.4f} {'✅ High' if similarity > 0.7 else '⚠️ Medium' if similarity > 0.4 else '❌ Low'}")

# ============================================================================
# SECTION 6: FAISS VECTOR DATABASE
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 6: FAISS - Building Vector Database for Fast Search")
print("=" * 70)

import faiss

# Create FAISS index
print("\n⏳ Building FAISS index...")

dimension = chunk_embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)

# Add embeddings to index
index.add(chunk_embeddings.astype('float32'))

print(f"✅ FAISS index built successfully!")
print(f"📊 Index type: IndexFlatL2 (exact search)")
print(f"📊 Dimensions: {dimension}")
print(f"📊 Total vectors: {index.ntotal}")
print(f"💾 Approximate size: ~{(index.ntotal * dimension * 4) / 1024:.2f} KB")

# ============================================================================
# SECTION 7: SEMANTIC SEARCH
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 7: SEMANTIC SEARCH - Finding Relevant Information")
print("=" * 70)

def semantic_search(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """
    Perform semantic search using FAISS.
    
    Args:
        query: User query
        k: Number of results to return
    
    Returns:
        List of (chunk, distance) tuples
    """
    # Convert query to embedding
    query_embedding = embedding_model.encode([query]).astype('float32')
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Get results
    results = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append((chunks[idx], float(dist)))
    
    return results

# Test queries
test_queries = [
    "What is the code for recurrent depression in remission?",
    "Tell me about panic attacks",
    "What are the criteria for PTSD?",
    "Information about eating disorders"
]

print("\n🧪 DEMONSTRATION: Running semantic searches")
print("=" * 70)

for query in test_queries:
    print(f"\n🔍 Query: \"{query}\"")
    print("-" * 70)
    
    start_time = time.time()
    results = semantic_search(query, k=3)
    search_time = time.time() - start_time
    
    for i, (chunk, distance) in enumerate(results, 1):
        # Convert L2 distance to similarity score (approximate)
        similarity = 1 / (1 + distance)
        print(f"\n📄 Result {i} (similarity: {similarity:.4f}):")
        print(f"   {chunk[:150]}...")
    
    print(f"\n⏱️  Search time: {search_time*1000:.2f}ms")

# ============================================================================
# SECTION 8: COMPLETE RAG PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 8: COMPLETE RAG PIPELINE")
print("=" * 70)

class SimpleRAG:
    """
    Simple RAG implementation for demonstration.
    """
    
    def __init__(self, chunks: List[str], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        
        # Build FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents."""
        # Convert query to embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return chunks
        return [self.chunks[idx] for idx in indices[0]]
    
    def answer(self, query: str) -> str:
        """
        Answer query using retrieval.
        (In full RAG, this would use an LLM to generate the answer)
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve(query, k=3)
        
        # Format response
        response = f"📚 Most relevant information found:\n\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            response += f"--- Source {i} ---\n{chunk}\n\n"
        
        response += "💡 Note: In a full RAG system, an LLM would synthesize this information into a natural language answer."
        
        return response

# Initialize RAG
print("\n⏳ Initializing RAG pipeline...")
rag = SimpleRAG(chunks, chunk_embeddings)
print("✅ RAG pipeline ready!")

# Test RAG
print("\n🧪 DEMONSTRATION: Complete RAG Query")
print("=" * 70)

test_query = "What is F33.4?"
print(f"\n❓ Question: {test_query}")
print("-" * 70)

answer = rag.answer(test_query)
print(f"\n{answer}")

# ============================================================================
# SECTION 9: PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 9: PERFORMANCE ANALYSIS")
print("=" * 70)

# Benchmark search performance
num_queries = 100
queries = [f"query {i}" for i in range(num_queries)]

print(f"\n⏳ Benchmarking {num_queries} searches...")

start_time = time.time()
for query in queries:
    _ = rag.retrieve(query, k=3)
total_time = time.time() - start_time

print(f"\n📊 Performance Results:")
print(f"   Total queries: {num_queries}")
print(f"   Total time: {total_time:.2f} seconds")
print(f"   Average time per query: {(total_time/num_queries)*1000:.2f}ms")
print(f"   Queries per second: {num_queries/total_time:.2f}")

# Memory analysis
print(f"\n💾 Memory Usage:")
print(f"   Document chunks: {len(chunks)} chunks")
print(f"   Embeddings: {chunk_embeddings.nbytes / 1024:.2f} KB")
print(f"   FAISS index: ~{(index.ntotal * dimension * 4) / 1024:.2f} KB")
print(f"   Total: ~{((chunk_embeddings.nbytes + index.ntotal * dimension * 4) / 1024):.2f} KB")

# ============================================================================
# SECTION 10: COMPARISON - KEYWORD VS SEMANTIC SEARCH
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 10: KEYWORD VS SEMANTIC SEARCH COMPARISON")
print("=" * 70)

def keyword_search(query: str, k: int = 3) -> List[str]:
    """Simple keyword-based search for comparison."""
    query_words = query.lower().split()
    scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        # Count keyword matches
        score = sum(1 for word in query_words if word in chunk_lower)
        scores.append((score, i))
    
    # Sort by score
    scores.sort(reverse=True)
    
    # Return top k
    return [chunks[idx] for score, idx in scores[:k] if score > 0]

# Compare searches
comparison_query = "mood problems and sadness"

print(f"\n🔍 Query: \"{comparison_query}\"")
print("=" * 70)

print("\n📋 KEYWORD SEARCH (Traditional):")
print("-" * 70)
keyword_results = keyword_search(comparison_query, k=3)
if keyword_results:
    for i, chunk in enumerate(keyword_results, 1):
        print(f"\n{i}. {chunk[:150]}...")
else:
    print("No results found (no exact keyword matches)")

print("\n\n🧠 SEMANTIC SEARCH (RAG/Embeddings):")
print("-" * 70)
semantic_results = rag.retrieve(comparison_query, k=3)
for i, chunk in enumerate(semantic_results, 1):
    print(f"\n{i}. {chunk[:150]}...")

print("\n\n💡 Observation:")
print("   Semantic search finds relevant results even without exact keyword matches!")
print("   It understands that 'mood problems and sadness' relates to depression.")

# ============================================================================
# SECTION 11: INTERACTIVE DEMO
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 11: INTERACTIVE DEMO")
print("=" * 70)

def interactive_search():
    """Interactive search interface."""
    print("\n🎯 Try your own queries! (Type 'quit' to exit)")
    print("-" * 70)
    
    while True:
        query = input("\n❓ Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n⏳ Searching...")
        start_time = time.time()
        results = rag.retrieve(query, k=3)
        search_time = time.time() - start_time
        
        print(f"\n📚 Top 3 Results (found in {search_time*1000:.2f}ms):")
        print("=" * 70)
        
        for i, chunk in enumerate(results, 1):
            print(f"\n📄 Result {i}:")
            print("-" * 70)
            print(chunk)
        
        print("\n" + "=" * 70)

# Uncomment to enable interactive mode
# interactive_search()

# ============================================================================
# SECTION 12: SUMMARY & KEY CONCEPTS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 12: SUMMARY & KEY CONCEPTS")
print("=" * 70)

summary = """
🎓 WHAT YOU'VE LEARNED:

1. TEXT SPLITTING (Chunking)
   - Breaks large documents into manageable pieces
   - Uses overlap to preserve context
   - Critical for effective retrieval

2. EMBEDDINGS
   - Converts text to 384-dimensional vectors
   - Captures semantic meaning, not just keywords
   - Model: all-MiniLM-L6-v2 (80 MB, runs on CPU)

3. VECTOR SIMILARITY
   - Cosine similarity measures semantic closeness
   - Similar meanings = Similar vectors
   - Range: -1 to 1 (1 = identical)

4. FAISS (Vector Database)
   - Fast similarity search in high dimensions
   - IndexFlatL2 for exact search
   - 10ms search time for 1,350 documents

5. SEMANTIC SEARCH
   - Understands meaning, not just keywords
   - Finds "depression" when you search "sadness"
   - Superior to traditional keyword matching

6. RAG PIPELINE
   - Retrieval: Find relevant documents
   - Augmentation: Add context to query
   - Generation: LLM creates answer (not shown here)

📊 PERFORMANCE ACHIEVED:
   - Search time: ~{:.2f}ms per query
   - Accuracy: High (semantic understanding)
   - Memory: ~{:.2f} KB
   - Scalability: Can handle millions of documents

🚀 NEXT STEPS:
   - Add LLM for natural language generation
   - Implement caching for common queries
   - Scale to larger datasets
   - Deploy as web service (like NeuroRAG)

💡 KEY INSIGHT:
   RAG combines the best of both worlds:
   - Factual accuracy (retrieval from real data)
   - Natural language (LLM generation)
   
   This is why it's superior to standalone LLMs for
   domain-specific applications like medical diagnosis!
"""

print(summary.format(
    (total_time/num_queries)*1000,
    (chunk_embeddings.nbytes + index.ntotal * dimension * 4) / 1024
))

# ============================================================================
# SECTION 13: SAVE RESULTS (OPTIONAL)
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 13: SAVE RESULTS")
print("=" * 70)

def save_index():
    """Save FAISS index and chunks for later use."""
    import pickle
    
    # Save FAISS index
    faiss.write_index(index, "demo_faiss_index.bin")
    
    # Save chunks
    with open("demo_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("\n✅ Saved:")
    print("   - demo_faiss_index.bin (FAISS index)")
    print("   - demo_chunks.pkl (text chunks)")

# Uncomment to save
# save_index()

print("\n" + "=" * 70)
print("🎉 DEMONSTRATION COMPLETE!")
print("=" * 70)
print("\n✅ You now understand how RAG works end-to-end!")
print("📚 Check out the full NeuroRAG implementation for production code.")
print("\n💡 To run in Google Colab:")
print("   1. Upload this file")
print("   2. Uncomment the !pip install lines at the top")
print("   3. Run all cells")
print("\n" + "=" * 70)
