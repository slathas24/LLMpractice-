import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/analyze_transcript"
CHAT_URL = "http://localhost:8000/chat_validate"

st.title("Meeting Transcript Analyzer & Validator")

uploaded_file = st.file_uploader("Upload Meeting Transcript (Text File)", type=["txt"])
transcript = ""
if uploaded_file:
    transcript = uploaded_file.read().decode("utf-8")
    st.text_area("Transcript Preview", transcript, height=200)

if transcript:
    if st.button("Analyze Transcript"):
        with st.spinner("Analyzing transcript..."):
            response = requests.post(API_URL, json={"transcript": transcript})
            if response.status_code == 200:
                result = response.json()
                st.subheader("Summary:")
                st.write(result.get("summary", result.get("raw_output", "No summary found.")))
                st.subheader("Key Discussions:")
                st.write(result.get("discussions", []))
                st.subheader("Actions:")
                st.write(result.get("actions", []))
                st.subheader("Decisions:")
                st.write(result.get("decisions", []))
                st.subheader("Context:")
                st.write(result.get("context", ""))
                st.session_state["extracted"] = result
            else:
                st.error(f"Error: {response.text}")

if transcript and "extracted" in st.session_state:
    st.markdown("---")
    st.subheader("Validate or Clarify via Chat")
    question = st.text_input("Ask a question about the meeting or extracted points:")
    if st.button("Ask LLM") and question:
        with st.spinner("Getting answer..."):
            payload = {
                "transcript": transcript,
                "extracted": st.session_state["extracted"],
                "question": question
            }
            response = requests.post(CHAT_URL, json=payload)
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(f"**LLM Answer:** {answer}")
            else:
                st.error(f"Error: {response.text}")
