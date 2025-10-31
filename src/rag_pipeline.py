

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

class RAGPipeline:
    def __init__(self, doc_path, embedding_model='all-MiniLM-L6-v2'):
        self.doc_path = doc_path
        self.embedding_model = embedding_model
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.index_path = "faiss_index/"
        self.vectorstore = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.use_phi3 = False  # Flag to enable Phi-3-Mini

    def load_text(self):
        with open(self.doc_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_chunks(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.create_documents([text])

    def create_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def build_vectorstore(self, documents):
        embeddings = self.create_embeddings()
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.vectorstore.save_local(self.index_path)

    def load_vectorstore(self):
        embeddings = self.create_embeddings()
        self.vectorstore = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)

    def setup_llm(self, model_id="tiiuae/falcon-rw-1b"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, device=torch.device("cpu"))
        return HuggingFacePipeline(pipeline=pipe)
    
    def setup_phi3_mini(self):
        """Load Phi-3-Mini for high-quality answer generation"""
        print("🔄 Loading Phi-3-Mini LLM (this may take a moment)...")
        try:
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {device.upper()}")
            
            # Load model with appropriate settings
            if device == "cuda":
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU mode - use float32 and lower memory
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.llm_model = self.llm_model.to(device)
            
            self.use_phi3 = True
            print("✅ Phi-3-Mini loaded successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not load Phi-3-Mini: {e}")
            print("   Falling back to retrieval-only mode.")
            self.use_phi3 = False
            return False
    
    def generate_answer_with_phi3(self, query, context):
        """Generate natural language answer using Phi-3-Mini"""
        if not self.use_phi3 or not self.llm_model:
            return None
        
        prompt = f"""<|system|>
You are a medical information assistant specializing in mental health diagnostics. Use ONLY the provided ICD-10 documentation to answer questions accurately and professionally.
<|end|>
<|user|>
ICD-10 Documentation:
{context}

Question: {query}

Instructions: Provide a clear, medically accurate answer based strictly on the documentation above. Include relevant ICD-10 codes when applicable. If the documentation doesn't contain enough information, say so clearly.
<|end|>
<|assistant|>"""

        try:
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()
            
            return answer
            
        except Exception as e:
            print(f"⚠️  Error generating answer: {e}")
            return None

    def get_qa_chain(self):
        if not self.vectorstore:
            self.load_vectorstore()
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.7})
        llm = self.setup_llm()
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    
    def simple_search(self, query, k=3):
        """Fast retrieval-only search without LLM (instant results)"""
        if not self.vectorstore:
            self.load_vectorstore()
        docs = self.vectorstore.similarity_search(query, k=k)
        # Combine retrieved documents
        if not docs:
            return "❌ No relevant information found. Try rephrasing your question."
        
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Add helpful note
        note = "\n\n💡 **Note:** This database contains ICD-10 Chapter V (Mental & Behavioural Disorders). For physical conditions, codes may be referenced but not fully described."
        
        return f"📚 Most relevant information found:\n\n{context}{note}"
    
    def smart_search(self, query, k=3):
        """Advanced search with Phi-3-Mini answer generation"""
        if not self.vectorstore:
            self.load_vectorstore()
        
        # Step 1: Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query, k=k)
        
        if not docs:
            return {
                "answer": "❌ No relevant information found in the database. Try rephrasing your question.",
                "sources": [],
                "mode": "retrieval"
            }
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Step 2: Try to generate answer with Phi-3-Mini
        if self.use_phi3:
            print("🤖 Generating AI-powered answer...")
            generated_answer = self.generate_answer_with_phi3(query, context)
            
            if generated_answer:
                return {
                    "answer": f"🤖 **AI-Generated Answer:**\n\n{generated_answer}",
                    "sources": [doc.page_content[:200] + "..." for doc in docs],
                    "mode": "llm"
                }
        
        # Step 3: Fallback to retrieval-only
        note = "\n\n💡 **Note:** This is direct retrieval from ICD-10 documentation. For AI-generated answers, enable Phi-3-Mini."
        return {
            "answer": f"📚 **Retrieved Information:**\n\n{context}{note}",
            "sources": [doc.page_content[:200] + "..." for doc in docs],
            "mode": "retrieval"
        }
