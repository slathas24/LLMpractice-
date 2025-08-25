import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/summarize_meeting"

st.title("Meeting Notes Summarizer")


import zipfile
import io

with st.form("meeting_form"):
    project_id = st.text_input("Project ID")
    meeting_type = st.text_input("Meeting Type")
    uploaded_zip = st.file_uploader("Upload a ZIP file containing the meeting document", type=["zip"])
    notes = ""
    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip) as z:
            doc_files = [f for f in z.namelist() if f.lower().endswith(('.txt', '.md', '.docx'))]
            if doc_files:
                selected_doc = st.selectbox("Select document from ZIP", doc_files)
                with z.open(selected_doc) as doc_file:
                    if selected_doc.lower().endswith('.txt') or selected_doc.lower().endswith('.md'):
                        notes = doc_file.read().decode("utf-8")
                    elif selected_doc.lower().endswith('.docx'):
                        import docx
                        doc = docx.Document(io.BytesIO(doc_file.read()))
                        notes = "\n".join([p.text for p in doc.paragraphs])
            else:
                st.warning("No supported document found in ZIP.")
    else:
        notes = st.text_area("Paste Meeting Notes Script/Doc")
    submitted = st.form_submit_button("Extract High-Level Points")

if submitted:
    if not notes:
        st.error("No meeting notes found. Please upload a document or paste notes.")
    else:
        payload = {
            "project_id": project_id,
            "meeting_type": meeting_type,
            "notes": notes
        }
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.subheader("High-Level Points for Approval:")
                for idx, point in enumerate(result["high_level_points"], 1):
                    st.write(f"{idx}. {point}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
