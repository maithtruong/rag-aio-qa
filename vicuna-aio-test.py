from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys


def main():
    # (b) Read PDF file with UTF-8 encoding
    FILE_PATH = "YOLOv10_Tutorials.pdf"

    # Load PDF file with UTF-8 encoding
    with open(FILE_PATH, 'rb') as f:
        pdf_data = f.read()

    # Use PyPDFLoader to load the PDF data
    loader = PyPDFLoader(pdf_data)
    documents = loader.load()
    print("Number of sub-documents:", len(documents))
    print(documents[0])

    # (c) Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print("Number of sub-documents:", len(docs))
    print(docs[0])

    # (d) Initialize vectorization
    embedding = HuggingFaceEmbeddings()
    vectors = embedding.vectorize_documents(docs)

    # (e) Initialize vector database
    vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
    retriever = vector_db.as_retriever()
    query = "What is YOLO?"
    result = retriever.invoke(query)
    print("Number of relevant documents:", len(result))
    for doc in result:
        print(doc.page_content)

    # Initialize language model
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    llm = HuggingFacePipeline(pipeline=model_pipeline)

    # Run RAG program
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | hub.pull("rlm/rag-prompt")
        | llm
        | StrOutputParser()
    )
    USER_QUESTION = "What is YOLOv10?"
    output = rag_chain.invoke(USER_QUESTION)
    answer = output.split('Answer:')[1].strip()
    print("Answer:", answer)


if __name__ == "__main__":
    main()