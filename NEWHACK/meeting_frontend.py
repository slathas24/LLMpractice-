VECTOR_DB_PATH = "faiss_meeting_index.bin"
META_DB_PATH = "faiss_meeting_meta.npy"

import streamlit as st
import openai
import os
import zipfile
import io
import pandas as pd
import faiss
import numpy as np
from datetime import datetime

st.title("Meeting Notes Summarizer")

# --- Load meeting types ---
csv_path = "meeting_types.csv"
if os.path.exists(csv_path):
    meeting_types_df = pd.read_csv(csv_path)
    meeting_types = meeting_types_df.iloc[:, 0].dropna().unique().tolist()
else:
    meeting_types = ["General", "Review", "Planning"]

# --- Initialize session state ---
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "points" not in st.session_state:
    st.session_state.points = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Meeting Form ---
with st.form("meeting_form"):
    project_id = st.text_input("Project ID", value=st.session_state.get("project_id", ""))
    meeting_type = st.selectbox("Meeting Type", meeting_types, 
                                index=meeting_types.index(st.session_state.get("meeting_type", meeting_types[0])))
    approver_name = st.text_input("Approver Name", value=st.session_state.get("approver_name", ""))

    uploaded_zip = st.file_uploader("Upload a ZIP file containing the meeting document", type=["zip"])
    uploaded_txt = st.file_uploader("Or upload a text file directly", type=["txt", "md"])

    notes = ""
    if uploaded_txt:
        notes = uploaded_txt.read().decode("utf-8")
    elif uploaded_zip:
        with zipfile.ZipFile(uploaded_zip) as z:
            doc_files = [f for f in z.namelist() if f.lower().endswith(('.txt', '.md', '.docx'))]
            if doc_files:
                selected_doc = st.selectbox("Select document from ZIP", doc_files)
                with z.open(selected_doc) as doc_file:
                    if selected_doc.lower().endswith(('.txt', '.md')):
                        notes = doc_file.read().decode("utf-8")
                    elif selected_doc.lower().endswith('.docx'):
                        import docx
                        doc = docx.Document(io.BytesIO(doc_file.read()))
                        notes = "\n".join([p.text for p in doc.paragraphs])
            else:
                st.warning("No supported document found in ZIP.")
    else:
        notes = st.text_area("Paste Meeting Notes Script/Doc", value=st.session_state.notes)

    submitted = st.form_submit_button("Extract High-Level Points")

# --- Handle form submission ---
if submitted:
    if not notes:
        st.error("No meeting notes found. Please upload a document or paste notes.")
    else:
        # Save in session state
        st.session_state.project_id = project_id
        st.session_state.meeting_type = meeting_type
        st.session_state.notes = notes
        st.session_state.approver_name = approver_name

        openai.api_key = "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"
        prompt = (
            f"Project ID: {project_id}\nMeeting Type: {meeting_type}\nMeeting Notes: {notes}\n"
            "Extract the high-level action points and decisions from these meeting notes as a bullet list."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            content = response.choices[0].message["content"]
            points = [line.strip('-• ').strip() for line in content.split('\n') if line.strip() and (line.startswith('-') or line.startswith('•'))]
            if not points:
                points = [content]
            st.session_state.points = points

        except Exception as e:
            st.error(f"Failed to connect to OpenAI: {e}")


# --- Display Results if Available ---
if st.session_state.points:
    st.subheader("High-Level Points for Approval:")
    for idx, point in enumerate(st.session_state.points, 1):
        st.write(f"{idx}. {point}")

    # --- Approve/Reject Buttons ---
    col1, col2 = st.columns(2)
    approved = col1.button("Approve", key="approve_button")
    rejected = col2.button("Reject", key="reject_button")

    # --- FAISS Vector DB Setup ---
    VECTOR_DB_PATH = "faiss_meeting_index.bin"
    META_DB_PATH = "faiss_meeting_meta.npy"
    def get_embedding(text):
        # Use OpenAI embedding API (or fallback to dummy vector for demo)
        try:
            emb_response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(emb_response["data"][0]["embedding"], dtype=np.float32)
        except Exception:
            # fallback: random vector
            return np.random.rand(1536).astype(np.float32)

    def save_to_faiss(meeting_id, meeting_type, approval_date, transcript, approver_name):
        vector = get_embedding(transcript)
        # Load or create FAISS index
        if os.path.exists(VECTOR_DB_PATH):
            index = faiss.read_index(VECTOR_DB_PATH)
            meta = np.load(META_DB_PATH, allow_pickle=True).tolist()
        else:
            index = faiss.IndexFlatL2(len(vector))
            meta = []
        index.add(np.expand_dims(vector, axis=0))
        meta.append({
            "meeting_id": meeting_id,
            "meeting_type": meeting_type,
            "approval_date": approval_date,
            "approver_name": approver_name,
            "transcript": transcript
        })
        faiss.write_index(index, VECTOR_DB_PATH)
        np.save(META_DB_PATH, np.array(meta, dtype=object))

    if approved:
        approval_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meeting_id = st.session_state.get("project_id", "")
        meeting_type = st.session_state.get("meeting_type", "")
        transcript = st.session_state.get("notes", "")
        approver_name = st.session_state.get("approver_name", "")
        save_to_faiss(meeting_id, meeting_type, approval_date, transcript, approver_name)
        st.success(f"Meeting approved and saved to vector DB! Meeting ID: {meeting_id}")
    if rejected:
        st.warning("Meeting rejected.")


    # --- Chat Facility ---
    st.markdown("---")
    st.subheader("Delve Deeper: Chat with Meeting Notes")

    user_question = st.text_input("Ask a question about the meeting notes:", key="chat_input")
    ask_clicked = st.button("Ask", key="ask_button")

    if ask_clicked and user_question:
        chat_prompt = (
            f"Meeting Transcript:\n{st.session_state.notes}\n"
            f"High-Level Points: {st.session_state.points}\n"
            f"User Question: {user_question}\n"
            "Answer the user's question or clarify the context."
        )
        try:
            chat_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": chat_prompt}],
                max_tokens=300
            )
            answer = chat_response.choices[0].message["content"]
        except Exception as e:
            answer = f"Error: {e}"

        st.session_state.chat_history.append((user_question, answer))

    # --- Show Chat History ---
    if st.session_state.chat_history:
        st.markdown("#### Chat History")
        for q, a in st.session_state.chat_history:
            st.write(f"**You:** {q}")
            st.write(f"**Assistant:** {a}")

# --- Project Category Chat (FAISS Search) ---
st.markdown("---")
st.subheader("Project Category Search & Chat")
project_query = st.text_input("Ask about a project category (meeting type):", key="project_category_input")
search_clicked = st.button("Search Meetings", key="search_button")
'''
def search_faiss_by_category(category):
    # Load meta info
    if not os.path.exists(META_DB_PATH):
        return []
    meta = np.load(META_DB_PATH, allow_pickle=True).tolist()
    # Filter meetings by type/category
    return [m for m in meta if category.lower() in m.get("meeting_type", "").lower()]

if search_clicked and project_query:
    results = search_faiss_by_category(project_query)
    if results:
        st.markdown(f"#### Meetings found for category '{project_query}':")
        for i, m in enumerate(results, 1):
            st.write(f"**Meeting {i}:**")
            st.write(f"- **ID:** {m['meeting_id']}")
            st.write(f"- **Type:** {m['meeting_type']}")
            st.write(f"- **Date Approved:** {m['approval_date']}")
            st.write(f"- **Approver:** {m['approver_name']}")
            st.write(f"- **Transcript:** {m['transcript'][:500]}{'...' if len(m['transcript'])>500 else ''}")
    else:
        st.info(f"No meetings found for category '{project_query}'.")
'''
