
from rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline(doc_path="data/icd10_text.txt")

    print("🔍 Loading and processing document...")
    raw_text = rag.load_text()
    chunks = rag.split_chunks(raw_text)
    rag.build_vectorstore(chunks)

    print("📚 Vector store built and saved!")

    qa_chain = rag.get_qa_chain()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        answer = qa_chain.invoke(query)
        # Extract result text
        if isinstance(answer, dict):
            answer = answer.get('result', str(answer))
        print("\n💡 Answer:", answer)

if __name__ == "__main__":
    main()
