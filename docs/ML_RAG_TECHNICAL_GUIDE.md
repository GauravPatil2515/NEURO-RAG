# 🧠 NeuroRAG - ML/RAG Technical Deep Dive

> **Complete guide to the Machine Learning and RAG architecture powering NeuroRAG**

## 📋 Table of Contents

- [Overview](#overview)
- [RAG Architecture](#rag-architecture)
- [Embeddings & Semantic Search](#embeddings--semantic-search)
- [FAISS Vector Database](#faiss-vector-database)
- [Text Splitting Strategy](#text-splitting-strategy)
- [Language Models](#language-models)
- [LangChain Framework](#langchain-framework)
- [PyTorch Backend](#pytorch-backend)
- [Mathematical Foundations](#mathematical-foundations)
- [Performance Analysis](#performance-analysis)
- [Data Flow Example](#data-flow-example)
- [Production Considerations](#production-considerations)

---

## 🎯 Overview

**NeuroRAG** implements a **Retrieval-Augmented Generation (RAG)** system for ICD-10 mental health information retrieval. Unlike traditional chatbots that rely solely on pre-trained knowledge, RAG combines:

1. **Retrieval** - Finding relevant information from a curated knowledge base
2. **Generation** - Using AI to present information in a helpful format

### Why RAG?

| Traditional LLMs | RAG (NeuroRAG) |
|------------------|----------------|
| ❌ Knowledge cutoff dates | ✅ Always up-to-date (you control data) |
| ❌ Hallucinations | ✅ Factual (retrieves actual documents) |
| ❌ No private data access | ✅ Uses custom knowledge base |
| ❌ No source citations | ✅ Traceable to source |

---

## 🏗️ RAG Architecture

### Complete Pipeline

```
┌─────────────────────────────────────────────────┐
│              RAG PIPELINE                        │
├─────────────────────────────────────────────────┤
│                                                  │
│  STAGE 1: RETRIEVAL (Finding Information)       │
│  ────────────────────────────────────────       │
│  User Query: "What is depression?"              │
│      ↓                                           │
│  [Embedding Model] → Query Vector (384-dim)     │
│      ↓                                           │
│  [FAISS Vector DB] → Similarity Search          │
│      ↓                                           │
│  Top K Most Relevant Documents                  │
│                                                  │
│  STAGE 2: GENERATION (Creating Answer)          │
│  ────────────────────────────────────────       │
│  Retrieved Documents + Original Query           │
│      ↓                                           │
│  [LLM (Falcon-1B)] → Context-Aware Generation  │
│      ↓                                           │
│  Final Answer                                   │
│                                                  │
└─────────────────────────────────────────────────┘
```

### System Flow

```python
# High-level workflow
def rag_search(query):
    # 1. Convert query to vector
    query_vector = embedding_model.encode(query)  # [384 dimensions]
    
    # 2. Find similar documents
    docs = faiss_index.search(query_vector, k=3)
    
    # 3. (Optional) Generate answer with LLM
    answer = llm.generate(query, context=docs)
    
    return answer
```

---

## 🧮 Embeddings & Semantic Search

### What are Embeddings?

**Embeddings** convert text into numerical vectors that capture semantic meaning.

```python
# Example
text1 = "Depression"
text2 = "Depressive disorder"
text3 = "Diabetes"

# After embedding (simplified to 3D)
vector1 = [0.8, 0.3, 0.1]  # Depression
vector2 = [0.7, 0.4, 0.1]  # Depressive disorder (similar to vector1)
vector3 = [0.1, 0.1, 0.9]  # Diabetes (very different)
```

**Key Insight:** Similar meanings = Similar vectors

### Model: all-MiniLM-L6-v2

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

**Specifications:**
- **Architecture:** Sentence-BERT (SBERT)
- **Dimensions:** 384 (each text → 384 numbers)
- **Training Data:** 1 billion sentence pairs
- **Speed:** ~2000 sentences/second on CPU
- **Size:** ~80 MB
- **Normalization:** L2 normalized (unit vectors)

### How Embeddings Work

```
Input: "What is depression?"
    ↓
Tokenization: ["what", "is", "depression", "?"]
    ↓
Token IDs: [2054, 2003, 6438, 1029]
    ↓
Transformer Neural Network (6 layers)
    ↓
Mean Pooling (average token embeddings)
    ↓
384-dimensional vector: [0.023, -0.145, 0.892, ..., 0.334]
    ↓
L2 Normalization (unit length)
    ↓
Final Embedding: [0.012, -0.073, 0.446, ..., 0.167]
```

### Semantic Search vs Keyword Search

**Keyword Search (Traditional):**
```
Query: "depression"
Matches: Only documents containing "depression"
Misses: "depressive disorder", "mood disorder"
```

**Semantic Search (NeuroRAG):**
```
Query: "depression"
Embedding: [0.8, 0.3, 0.1, ...]
Finds:
  - "depression" (exact match) - similarity: 1.0
  - "depressive disorder" (synonym) - similarity: 0.92
  - "mood disorder" (related) - similarity: 0.78
  - "F32" (ICD code) - similarity: 0.85
```

---

## 🔍 FAISS Vector Database

### What is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library for efficient similarity search in high-dimensional vector spaces.

**Creator:** Meta AI Research  
**Purpose:** Find similar vectors FAST (even with millions of vectors)

### The Problem FAISS Solves

**Naive Approach:**
```python
# Compare query to EVERY document (slow)
def naive_search(query_vector, all_vectors, k=3):
    similarities = []
    for i, doc_vector in enumerate(all_vectors):
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((sim, i))
    
    similarities.sort(reverse=True)
    return similarities[:k]

# Time Complexity: O(N × D)
# N = number of documents (1,350)
# D = dimensions (384)
```

**FAISS Approach:**
- Uses optimized index structures
- SIMD vectorization (8 numbers at once)
- Multi-threading
- Time Complexity: O(N) but 50x faster in practice

### FAISS in NeuroRAG

```python
from langchain_community.vectorstores import FAISS

# Building the index
def build_vectorstore(documents):
    embeddings = create_embeddings()
    
    # FAISS converts documents to vectors and builds index
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save to disk (2.11 MB file)
    vectorstore.save_local("faiss_index/")
```

**What happens internally:**

```
Step 1: Document Processing
──────────────────────────
1,350 text chunks
    ↓
For each chunk:
    text → embedding_model → 384-dim vector
    
Total: 1,350 × 384 = 518,400 numbers

Step 2: Index Creation (IndexFlatL2)
────────────────────────────────────
- Stores all vectors in contiguous memory
- Creates metadata mapping (vector_id → document)
- Optimizes for L2 distance calculations

Step 3: Serialization
─────────────────────
- index.faiss (2.11 MB) - vector data
- index.pkl - metadata
```

### Search Algorithm

```python
def similarity_search(query, k=3):
    # 1. Convert query to vector
    query_vector = embeddings.embed_query(query)
    
    # 2. FAISS finds k nearest neighbors
    docs = vectorstore.similarity_search_by_vector(query_vector, k=k)
    
    return docs
```

**Internal FAISS Process:**

```python
def faiss_l2_search(query_vector, k=3):
    """
    Find k most similar vectors using L2 distance
    """
    distances = []
    
    # Compare query to every document vector (optimized with SIMD)
    for i, doc_vector in enumerate(all_vectors):
        # L2 Distance (Euclidean)
        dist = sqrt(sum((query_vector[j] - doc_vector[j])**2 
                       for j in range(384)))
        distances.append((dist, i))
    
    # Sort by distance (smaller = more similar)
    distances.sort()
    
    # Return top k document IDs
    return [doc_id for (dist, doc_id) in distances[:k]]
```

### FAISS Index Types

```python
# 1. IndexFlatL2 (what we use)
# - Exact search (100% accuracy)
# - Compares to all vectors
# - Best for: < 1 million vectors
# - Speed: O(N) but highly optimized

# 2. IndexIVFFlat (for larger datasets)
# - Partitions vectors into clusters
# - Approximate search
# - Speed: O(√N)
# - Best for: 1M - 10M vectors

# 3. IndexIVFPQ (for huge datasets)
# - Compresses vectors with quantization
# - Approximate search
# - Speed: Very fast
# - Best for: 10M+ vectors
```

### Performance Metrics

**NeuroRAG Statistics:**
- **Documents:** 1,350 chunks
- **Index Size:** 2.11 MB
- **Search Time:** ~10ms per query
- **Memory Usage:** ~3 MB (loaded in RAM)

---

## 📄 Text Splitting Strategy

### Why Split Documents?

**Problem:** ICD-10 file is 660 KB (too large for single embedding)

**Solution:** Split into manageable chunks with overlap

### RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Maximum characters per chunk
    chunk_overlap=50     # Overlap between chunks
)

documents = splitter.create_documents([text])
```

### Recursive Splitting Algorithm

```python
def split_text(text, max_size=500):
    """
    Try to split at natural boundaries
    """
    separators = [
        "\n\n",    # Paragraph breaks (best)
        "\n",      # Line breaks
        ". ",      # Sentences
        " ",       # Words
        ""         # Characters (worst case)
    ]
    
    for separator in separators:
        if len(text) <= max_size:
            return [text]
        
        chunks = text.split(separator)
        if all(len(c) <= max_size for c in chunks):
            return chunks
    
    # Fallback: hard split
    return [text[i:i+max_size] for i in range(0, len(text), max_size)]
```

### Overlap Strategy

```
Document: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
Chunk Size: 10, Overlap: 3

Chunk 1: ABCDEFGHIJ
Chunk 2:       HIJKLMNOPQ  ← overlaps "HIJ"
Chunk 3:              OPQRSTUVWX
Chunk 4:                    VWXYZ

Why overlap?
✅ Preserves context at boundaries
✅ Ensures no information is lost
✅ Helps with queries spanning chunks
```

### Statistics

```yaml
Total Documents: 1,350 chunks
Average Chunk Size: 486 characters
Distribution:
  < 400 chars: 15%
  400-500 chars: 70%
  500+ chars: 15%
```

---

## 🤖 Language Models

### Falcon-RW-1B Specifications

```yaml
Model Name: Falcon-RW-1B (Refined Web)
Creator: Technology Innovation Institute (TII), UAE
Parameters: 1.1 billion
Architecture: Decoder-only Transformer
Training Data: 350B tokens (RefinedWeb dataset)
Context Length: 2048 tokens
Precision: FP32 (32-bit floating point)
Size on Disk: ~4.5 GB
License: Apache 2.0
```

### Model Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def setup_llm(model_id="tiiuae/falcon-rw-1b"):
    # 1. Load tokenizer (text → token IDs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Load model weights (~4.5 GB)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 3. Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=torch.device("cpu")
    )
    
    # 4. Wrap in LangChain interface
    return HuggingFacePipeline(pipeline=pipe)
```

### Transformer Architecture

```
Input Tokens
    ↓
Embedding Layer (tokens → vectors)
    ↓
┌──────────────────────┐
│ Transformer Block 1  │  ← 24 blocks total
│ ├─ Multi-Head Attn   │  (self-attention)
│ ├─ Layer Norm        │
│ ├─ Feed Forward      │  (MLP)
│ └─ Layer Norm        │
├──────────────────────┤
│ Transformer Block 2  │
│ ...                  │
│ Transformer Block 24 │
└──────────────────────┘
    ↓
Output Layer (predicts next token)
    ↓
Generated Text
```

### Auto-Regressive Generation

```python
def generate_text(prompt, max_tokens=256):
    """
    Generate text one token at a time
    """
    tokens = tokenize(prompt)
    
    for i in range(max_tokens):
        # 1. Forward pass through transformer
        logits = model(tokens)  # [vocab_size] probabilities
        
        # 2. Sample next token (temperature, top-k, nucleus)
        next_token = sample(logits)
        
        # 3. Append to sequence
        tokens.append(next_token)
        
        # 4. Stop if end token
        if next_token == EOS_TOKEN:
            break
    
    return detokenize(tokens)
```

### Performance

```yaml
Speed on CPU: ~5-10 tokens/second
Memory: ~6 GB RAM (model + activations)
Latency: ~20 seconds for 256 token response
Inference Mode: Weights frozen (no training)
```

---

## 🔗 LangChain Framework

### What is LangChain?

LangChain is a **framework for building LLM applications**. It provides:
- Pre-built chains (workflows)
- Vector store integrations
- Retriever abstractions
- Prompt templates

### LangChain Components in NeuroRAG

#### 1. Vector Store Integration

```python
from langchain_community.vectorstores import FAISS

# Unified interface across all vector stores
vectorstore = FAISS.from_documents(documents, embeddings)

# Standardized methods
vectorstore.similarity_search(query, k=3)
vectorstore.max_marginal_relevance_search(query, k=5)
vectorstore.as_retriever()
```

#### 2. Retriever with MMR

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",              # Maximal Marginal Relevance
    search_kwargs={
        "k": 5,                     # Retrieve 5 documents
        "lambda_mult": 0.7          # Diversity parameter
    }
)
```

**MMR (Maximal Marginal Relevance):**

Balances **relevance** and **diversity** to avoid redundant results.

```python
def mmr_search(query_vector, k=5, lambda_mult=0.7):
    """
    lambda_mult:
        1.0 = pure relevance (may return duplicates)
        0.0 = pure diversity (may miss relevant docs)
        0.7 = balanced (recommended)
    """
    selected = []
    candidates = all_documents
    
    # Select first (most relevant)
    first_doc = most_similar(query_vector, candidates)
    selected.append(first_doc)
    candidates.remove(first_doc)
    
    # Select remaining k-1 documents
    for i in range(k - 1):
        best_score = -inf
        best_doc = None
        
        for doc in candidates:
            # Relevance to query
            relevance = similarity(query_vector, doc.vector)
            
            # Similarity to already selected
            max_sim = max(similarity(doc.vector, s.vector) 
                         for s in selected)
            
            # MMR score
            score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            
            if score > best_score:
                best_score = score
                best_doc = doc
        
        selected.append(best_doc)
        candidates.remove(best_doc)
    
    return selected
```

#### 3. RetrievalQA Chain

```python
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Usage
answer = qa_chain.run("What is depression?")
```

**Internal Workflow:**

```python
def retrieval_qa(user_query):
    # Step 1: Retrieve relevant documents
    docs = retriever.get_relevant_documents(user_query)
    
    # Step 2: Build prompt with context
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Use the following context to answer the question.
    
    Context:
    {context}
    
    Question: {user_query}
    
    Answer:
    """
    
    # Step 3: Generate answer with LLM
    answer = llm(prompt)
    
    return answer
```

---

## ⚙️ PyTorch Backend

### Role of PyTorch

PyTorch powers the neural networks:
1. **Embedding Model** (MiniLM)
2. **Language Model** (Falcon-1B)

### Tensor Operations

```python
import torch

# Neural network computations use tensors
input_tensor = torch.tensor([[1, 2, 3, 4]])  # Shape: [1, 4]

# Matrix multiplication (core of transformers)
weights = torch.randn(4, 384)
output = torch.matmul(input_tensor, weights)  # [1, 384]

# Activation functions
activated = torch.nn.functional.gelu(output)

# Normalization
normalized = torch.nn.functional.layer_norm(activated, [384])
```

### Forward Pass

```python
def forward_pass(tokens):
    """
    Single forward pass through transformer
    """
    # 1. Embedding lookup
    embeddings = embedding_table[tokens]  # [seq_len, hidden_dim]
    
    # 2. Add positional encoding
    positions = torch.arange(len(tokens))
    embeddings += positional_encoding[positions]
    
    # 3. Pass through transformer blocks
    hidden_states = embeddings
    for layer in transformer_blocks:
        # Multi-head attention
        attention_out = layer.attention(hidden_states)
        
        # Feed-forward network
        ffn_out = layer.ffn(attention_out)
        
        # Residual connections + layer norm
        hidden_states = layer_norm(ffn_out + hidden_states)
    
    # 4. Output projection
    logits = output_layer(hidden_states)
    
    return logits  # [seq_len, vocab_size]
```

### CPU vs GPU

```python
# NeuroRAG uses CPU
device = torch.device("cpu")

# Why CPU?
# ✅ More accessible (no GPU required)
# ✅ 1B parameter model small enough for CPU
# ✅ Good enough for low-traffic applications

# Performance comparison:
# CPU: ~5 tokens/second
# GPU (RTX 3090): ~50 tokens/second (10x faster)
```

---

## 🔬 Mathematical Foundations

### Cosine Similarity

```python
def cosine_similarity(a, b):
    """
    Measures angle between vectors
    Range: [-1, 1]
        1 = identical
        0 = orthogonal (unrelated)
       -1 = opposite
    """
    dot_product = sum(a[i] * b[i] for i in range(len(a)))
    magnitude_a = sqrt(sum(a[i]**2 for i in range(len(a))))
    magnitude_b = sqrt(sum(b[i]**2 for i in range(len(b))))
    
    return dot_product / (magnitude_a * magnitude_b)
```

**Geometric Interpretation:**

```
Vector Space (2D visualization):

    depression ↗
              /
             / 30°
            /____→ depressive_disorder
           
cos(30°) = 0.87 (high similarity)


    depression ↗
              |
              | 90°
              ↓
            diabetes
            
cos(90°) = 0.0 (no similarity)
```

### L2 Distance (Euclidean)

```python
def l2_distance(a, b):
    """
    Euclidean distance in high-dimensional space
    """
    return sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

# For normalized vectors:
# l2_distance(a, b) = sqrt(2 - 2*cosine_similarity(a, b))
# So minimizing L2 = maximizing cosine similarity
```

### Attention Mechanism

```python
def attention(query, keys, values):
    """
    Core of transformer models
    Allows model to focus on relevant input parts
    """
    # 1. Calculate attention scores
    scores = matmul(query, keys.T) / sqrt(d_k)
    
    # 2. Softmax (convert to probabilities)
    weights = softmax(scores)
    
    # 3. Weighted sum of values
    output = matmul(weights, values)
    
    return output
```

**Example:**
```
Input: "What is recurrent depressive disorder?"

Attention weights when processing "disorder":
    what:      0.05  (low attention)
    is:        0.03
    recurrent: 0.35  (high - modifies disorder)
    depressive:0.45  (high - modifies disorder)
    disorder:  0.12
```

### Softmax Function

```python
def softmax(logits):
    """
    Converts raw scores to probabilities
    Output always sums to 1.0
    """
    exp_logits = [exp(x) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]

# Example:
logits = [2.0, 1.0, 0.1]
probs = softmax(logits)  # [0.66, 0.24, 0.10]
```

---

## 📈 Performance Analysis

### Memory Breakdown

```
Total RAM Usage: ~2 GB

┌─────────────────────────────────────┐
│ Component              │ Memory     │
├─────────────────────────────────────┤
│ Embedding Model        │            │
│   - Weights            │ 80 MB      │
│   - Inference Buffers  │ 20 MB      │
├─────────────────────────────────────┤
│ FAISS Vector Store     │            │
│   - Index Data         │ 2 MB       │
│   - Metadata           │ 1 MB       │
├─────────────────────────────────────┤
│ LLM (when loaded)      │            │
│   - Model Weights      │ 4.5 GB     │
│   - Activations        │ 500 MB     │
│   - KV Cache           │ 200 MB     │
├─────────────────────────────────────┤
│ Python Runtime         │            │
│   - Flask              │ 50 MB      │
│   - Libraries          │ 100 MB     │
│   - Overhead           │ 50 MB      │
└─────────────────────────────────────┘
```

### Computational Complexity

```python
# Embedding Generation
Time: O(L × D²)
    L = sequence length (tokens)
    D = hidden dimension (384)

For typical query (10 tokens):
    FLOPs ≈ 10 × 384² × 6 layers ≈ 8.8M operations
    Time: ~50ms on CPU

# FAISS Search
Time: O(N × D)
    N = number of documents (1,350)
    D = dimensions (384)

For similarity search:
    Operations: 1,350 × 384 ≈ 518K comparisons
    Time: ~10ms on CPU

# LLM Generation (if used)
Time: O(T × D² × L)
    T = tokens to generate (256)
    D = hidden dimension (2048)
    L = number of layers (24)

    FLOPs ≈ 256 × 2048² × 24 × 4 ≈ 51B operations
    Time: ~20 seconds on CPU
```

### Bottleneck Analysis

```
Search Request Breakdown:

┌─────────────────────────┬──────────┬─────────┐
│ Operation               │ Time     │ % Total │
├─────────────────────────┼──────────┼─────────┤
│ Query Embedding         │ 50ms     │ 3%      │
│ FAISS Search            │ 10ms     │ 1%      │
│ Document Retrieval      │ 5ms      │ <1%     │
│ Response Formatting     │ 15ms     │ 1%      │
│ Network (Flask)         │ 20ms     │ 1%      │
├─────────────────────────┼──────────┼─────────┤
│ TOTAL (Fast Mode)       │ ~100ms   │ 100%    │
└─────────────────────────┴──────────┴─────────┘

With LLM Generation:
┌─────────────────────────┬──────────┬─────────┐
│ Above operations        │ 100ms    │ <1%     │
│ LLM Generation          │ 20,000ms │ 99%     │
├─────────────────────────┼──────────┼─────────┤
│ TOTAL (Full RAG)        │ ~20s     │ 100%    │
└─────────────────────────┴──────────┴─────────┘

🔴 Bottleneck: LLM generation (when enabled)
```

### Optimization Strategies

#### 1. Lazy Loading
```python
# Don't load AI models until first search
def get_rag():
    global rag
    if rag is None:
        rag = RAGPipeline(...)  # Load only when needed
    return rag
```

#### 2. Fast Mode (Retrieval-Only)
```python
# Skip LLM generation, return docs directly
def simple_search(query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
```

#### 3. Normalized Embeddings
```python
# L2 normalization simplifies distance calculation
encode_kwargs={'normalize_embeddings': True}

# Distance becomes: sqrt(2 - 2*dot(a, b))
# Avoids computing vector magnitudes
```

#### 4. Potential: Quantization
```python
# Convert FP32 → INT8 (4x smaller, 3x faster)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True  # Requires bitsandbytes
)
```

#### 5. Potential: GPU Acceleration
```python
# Move FAISS to GPU (100x faster search)
import faiss
gpu_index = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(),
    0,  # GPU 0
    cpu_index
)
```

---

## 🔄 Data Flow Example

### Complete Query Execution

**User Query:** `"What is F33.4?"`

#### Step 1: Request Arrives
```python
@app.route('/api/search', methods=['POST'])
def search():
    query = "What is F33.4?"  # From JSON request
```

#### Step 2: Load RAG Pipeline
```python
rag_instance = get_rag()
# First time only:
# - Loads MiniLM model (80 MB) - 2s
# - Loads FAISS index (2 MB) - 1s
# Total: ~3 seconds

# Subsequent calls: instant (cached)
```

#### Step 3: Execute Search
```python
result = rag_instance.simple_search(query, k=3)

# Internal process:

# 3a. Convert query to embedding
query_embedding = embeddings.embed_query("What is F33.4?")
# Output: [0.023, -0.145, ..., 0.334] (384 numbers)
# Time: 50ms

# 3b. FAISS similarity search
docs = vectorstore.similarity_search_by_vector(query_embedding, k=3)
# Compares query to 1,350 document embeddings
# Uses optimized L2 distance
# Time: 10ms

# 3c. Retrieved documents:
doc1 = "F33.4 Recurrent depressive disorder, currently in remission..."
doc2 = "F33 Recurrent depressive disorder..."
doc3 = "F33.3 Recurrent depressive disorder, current episode severe..."

# 3d. Format response
context = "\n\n---\n\n".join([doc.page_content for doc in docs])
result = f"📚 Most relevant information found:\n\n{context}..."
# Time: 5ms
```

#### Step 4: Return Response
```python
return jsonify({
    'success': True,
    'query': query,
    'result': result
})
# Total time: ~100ms (after initial load)
```

### Visual Timeline

```
User clicks "Search"
    ↓ [0ms]
Flask receives request
    ↓ [5ms]
RAG pipeline check (cached)
    ↓ [10ms]
Query embedding generation
    ↓ [60ms]
FAISS vector search
    ↓ [70ms]
Document retrieval
    ↓ [75ms]
Response formatting
    ↓ [80ms]
JSON response sent
    ↓ [100ms]
User sees results
```

---

## 🎯 Production Considerations

### Scalability

#### Horizontal Scaling
```
NGINX Load Balancer
    ↓
┌─────────┬─────────┬─────────┐
│ Flask 1 │ Flask 2 │ Flask 3 │
└─────────┴─────────┴─────────┘
    ↓
Shared FAISS Index (NFS/Redis)
```

#### Vertical Scaling
```yaml
Current Setup:
  CPU: Any modern CPU
  RAM: 2 GB minimum
  Storage: 10 GB

Optimized Setup:
  CPU: 8+ cores for parallel search
  RAM: 16 GB for caching
  Storage: SSD for faster index loading
  GPU: Optional (RTX 3060+) for 10x LLM speed
```

### Monitoring Metrics

```python
metrics = {
    'query_latency': [],          # Response time (target: <200ms)
    'embedding_time': [],         # Embedding generation
    'search_time': [],            # FAISS search
    'memory_usage': [],           # RAM consumption
    'error_rate': [],             # Failed requests
    'throughput': []              # Queries per second
}
```

### Caching Strategy

```python
from functools import lru_cache

# Cache embeddings for common queries
@lru_cache(maxsize=1000)
def get_query_embedding(query):
    return embeddings.embed_query(query)

# Cache search results
@lru_cache(maxsize=500)
def cached_search(query):
    return vectorstore.similarity_search(query, k=3)
```

### Error Handling

```python
def robust_search(query):
    try:
        # Attempt search
        result = rag.simple_search(query)
        return result
        
    except Exception as e:
        # Log error
        logger.error(f"Search failed: {e}")
        
        # Fallback: keyword search
        return fallback_keyword_search(query)
```

---

## 📊 Key Concepts Summary

### Transfer Learning
```
Pre-trained Model (MiniLM, Falcon)
    ↓
Trained on billions of tokens
    ↓
Applied to medical domain
    ↓
No additional training needed
```

### Vector Space Model
```
High-dimensional space (384 dimensions)
- Each dimension = semantic feature
- Similar concepts cluster together
- Distance = semantic similarity

Example dimensions (hypothetical):
  dim[0] = "medical terminology" intensity
  dim[1] = "mental health" relevance
  dim[2] = "severity" level
  ...
  dim[383] = "treatment" related
```

### Inference vs Training

**Training (NOT done in NeuroRAG):**
- Requires labeled data
- Adjusts model weights
- Needs GPU cluster
- Takes days/weeks

**Inference (What NeuroRAG does):**
- Uses pre-trained weights
- Weights are frozen
- Runs on CPU
- Takes milliseconds

---

## 🛠️ Technology Stack

```yaml
ML/AI Stack:
  Embeddings: sentence-transformers (MiniLM)
  Vector DB: FAISS (Facebook AI)
  LLM: transformers (Falcon-RW-1B)
  Framework: LangChain
  Backend: PyTorch

Python Libraries:
  - langchain >= 1.0.0
  - langchain-community >= 0.4
  - sentence-transformers == 2.7.0
  - faiss-cpu >= 1.12.0
  - transformers >= 4.50.0
  - torch >= 2.0.0

Hardware Requirements:
  Minimum:
    - CPU: Any modern processor
    - RAM: 2 GB
    - Storage: 10 GB
  
  Recommended:
    - CPU: 4+ cores
    - RAM: 8 GB
    - Storage: 20 GB SSD
  
  Optional:
    - GPU: NVIDIA RTX 3060+ (10x faster LLM)
```

---

## 🎓 Further Reading

### Papers
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

### Documentation
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📝 Code Examples

### Building a Custom RAG Pipeline

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load your documents
with open('your_data.txt', 'r') as f:
    text = f.read()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = splitter.create_documents([text])

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

# 4. Build FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("my_index/")

# 5. Search
query = "Your question here"
results = vectorstore.similarity_search(query, k=3)

for doc in results:
    print(doc.page_content)
```

### Custom Similarity Search

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your documents
documents = [
    "Depression is a mental health disorder",
    "Anxiety can cause physical symptoms",
    "Bipolar disorder involves mood swings"
]

# Create embeddings
doc_embeddings = model.encode(documents)

# Query
query = "What is depression?"
query_embedding = model.encode([query])[0]

# Calculate similarities
similarities = []
for i, doc_emb in enumerate(doc_embeddings):
    similarity = np.dot(query_embedding, doc_emb)
    similarities.append((similarity, documents[i]))

# Sort by similarity
similarities.sort(reverse=True)

# Print results
for score, doc in similarities:
    print(f"Score: {score:.4f} | {doc}")
```

---

## 🙏 Acknowledgments

- **Meta AI** - FAISS vector search library
- **HuggingFace** - Pre-trained models and transformers library
- **LangChain** - RAG framework
- **TII UAE** - Falcon language models
- **UKPLab** - Sentence-BERT models

---

## 👨‍💻 Author

**Gaurav Patil**  
B.Tech Computer Engineering  
India 🇮🇳

**Project:** [NEURO-RAG](https://github.com/GauravPatil2515/NEURO-RAG)  
**GitHub:** [@GauravPatil2515](https://github.com/GauravPatil2515)

---

**Last Updated:** October 25, 2025  
**Version:** 1.0.0  
**License:** MIT

---

*Built with ❤️ using AI & Open Source Technology*
