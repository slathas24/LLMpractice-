from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def vectorize_docs(docs: dict):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for fname, content in docs.items():
        for doc in splitter.create_documents([content]):
            doc.metadata = {"source": fname}
            chunks.append(doc)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_documents(chunks, embeddings)

def query_themes(vectordb, questions: list):
    retriever = vectordb.as_retriever(search_type="similarity", k=5)
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return {q: chain.run(q) for q in questions}

def store_themes_in_vector_db(themes: list):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    docs = [Document(page_content=theme) for theme in themes]
    return FAISS.from_documents(docs, embeddings)