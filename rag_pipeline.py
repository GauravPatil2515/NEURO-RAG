

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
