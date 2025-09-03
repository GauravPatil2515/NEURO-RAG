

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

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
        return HuggingFaceEmbeddings(model_name=self.embedding_model)

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
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
