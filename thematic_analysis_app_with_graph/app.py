import streamlit as st
from backend.zip_handler import extract_zip_and_read_documents
from backend.rag_engine import vectorize_docs, query_themes
from utils.constants import DEFAULT_QUESTIONS
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 GPT-4 Thematic Analysis of Financial Documents")

uploaded_zip = st.file_uploader("📁 Upload a ZIP file", type="zip")
questions = DEFAULT_QUESTIONS.copy()
custom_questions = []
for i, question in enumerate(questions):
    custom = st.text_input(f"Question {i+1}", value=question)
    custom_questions.append(custom)

add_more = st.checkbox("➕ Add more custom questions")
if add_more:
    num_additional = st.number_input("How many more questions?", min_value=1, max_value=10, value=1)
    for i in range(num_additional):
        q = st.text_input(f"Additional Question {i+1}")
        if q:
            custom_questions.append(q)

if uploaded_zip and st.button("Run Analysis"):
    with st.spinner("Extracting and analyzing..."):
        docs = extract_zip_and_read_documents(uploaded_zip)
        if not docs:
            st.error("No valid files found in ZIP.")
        else:
            vectordb = vectorize_docs(docs)
            questions = [q.strip() for q in custom_questions if q.strip()]
            results = query_themes(vectordb, questions)

            st.success("✅ Analysis complete!")

            for q, a in results.items():
                st.subheader(f"🟦 {q}")
                st.markdown(a)

            st.subheader("📈 Thematic Summary: Answer Lengths")
            lengths = {q: len(a.split()) for q, a in results.items()}
            fig, ax = plt.subplots()
            ax.barh(list(lengths.keys()), list(lengths.values()), color="skyblue")
            ax.set_xlabel("Answer Word Count")
            ax.set_title("Length of Thematic Answers")
            st.pyplot(fig)

            st.markdown("### 🔍 Similar Theme Suggestions")
            theme_vectordb = store_themes_in_vector_db(questions)
            if questions:
                similar = theme_vectordb.similarity_search(questions[0], k=3)
                for doc in similar:
                    st.markdown(f"- {doc.page_content}")




























