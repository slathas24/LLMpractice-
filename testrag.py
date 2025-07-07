import os
import openai
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Dummy embedding class to feed precomputed vectors into FAISS
class DummyOpenAIEmbedder(Embeddings):
    def __init__(self, precomputed_vectors):
        self._vectors = precomputed_vectors

    def embed_documents(self, texts):
        return self._vectors

    def embed_query(self, text):
        # Used only for retrieval. You can generate this properly if needed.
        response = client.embeddings.create(input=text, model="text-embedding-3-large")
        return response.data[0].embedding


# ✅ Step 1: Vectorize documents and create FAISS index
def vectorize_docs(docs: dict):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = []

    for fname, content in docs.items():
        for chunk in splitter.create_documents([content]):
            chunk.metadata = {"source": fname}
            all_docs.append(chunk)

    texts = [doc.page_content for doc in all_docs]
    metadatas = [doc.metadata for doc in all_docs]

    vectors = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        if len(text) > 12000:
            text = text[:12000]
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            vectors.append(response.data[0].embedding)
        except Exception as e:
            print(f"❌ Failed embedding at index {i}: {e}")
            vectors.append([0.0] * 1536)

    assert len(vectors) == len(texts), "Mismatch between texts and vectors"

    embedder = DummyOpenAIEmbedder(precomputed_vectors=vectors)
    return FAISS.from_documents(all_docs, embedder)


# ✅ Step 2: Query the FAISS DB with GPT-4o
def query_themes(vectordb, questions: list):
    results = {}

    for question in questions:
        docs = vectordb.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an AI assistant. Use the context below to answer the question as accurately as possible.

Context:
{context}

Question: {question}
Answer:
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            answer = response.choices[0].message.content.strip()
            results[question] = answer
        except Exception as e:
            results[question] = f"❌ Error querying LLM: {e}"

    return results


# ✅ Example usage
if __name__ == "__main__":
    # 🔹 Sample data (replace with PDF/doc loader if needed)
    docs = {
        "esg_risks.txt": """
            ESG (Environmental, Social, Governance) risk is emerging as a key factor in financial analysis.
            It includes climate risk, board accountability, social equity, and environmental compliance.
        """,
        "governance.txt": """
            Good corporate governance includes transparency, accountability, and fair treatment of stakeholders.
            Poor governance can increase financial risk and damage reputation.
        """
    }

    # 🔹 Build vector DB from documents
    vectordb = vectorize_docs(docs)

    # 🔹 Define questions
    questions = [
        "What are ESG risks?",
        "How does governance impact financial stability?"
    ]

    # 🔹 Query using GPT-4o with retrieval
    answers = query_themes(vectordb, questions)

    for q, a in answers.items():
        print(f"\n❓ {q}\n➡️ {a}")



#  ..creating themes 
def analyze_documents_for_themes(docs: dict, max_themes_per_chunk=5):
    all_themes = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for fname, content in docs.items():
        chunks = splitter.split_text(content)
        print(f"📄 Analyzing {fname} with {len(chunks)} chunks...")

        for chunk in chunks:
            themes = extract_themes_from_text(chunk, max_themes=max_themes_per_chunk)
            all_themes.extend(themes)

    # Deduplicate based on label
    unique_themes = {}
    for label, desc in all_themes:
        if label not in unique_themes:
            unique_themes[label] = desc

    return unique_themes
