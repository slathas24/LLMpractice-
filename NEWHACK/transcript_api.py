import openai
import os


def analyze_transcript(transcript: str):
    #openai.api_key = OPENAI_API_KEY 
    prompt = (
        f"Meeting Transcript:\n{transcript}\n"
        "Summarize the meeting and extract key discussions, actions, decisions, and context as a JSON object."
        "\nExample output: { 'summary': '...', 'discussions': [...], 'actions': [...], 'decisions': [...], 'context': '...' }"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        import json
        content = response.choices[0].message["content"]
        try:
            result = json.loads(content)
        except Exception:
            result = {"raw_output": content}
        return result
    except Exception as e:
        import traceback
        print("OpenAI error in analyze_transcript:", e)
        traceback.print_exc()
        return {"error": str(e)}

def chat_validate(transcript: str, extracted: dict, question: str):
    #openai.api_key = OPENAI_API_KEY
    prompt = (
        f"Meeting Transcript:\n{transcript}\nExtracted Points: {extracted}\n"
        f"User Question: {question}\n"
        "Answer the user's question or clarify the context based on the transcript and extracted points."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return {"answer": response.choices[0].message["content"]}
    except Exception as e:
        import traceback
        print("OpenAI error in chat_validate:", e)
        traceback.print_exc()
        return {"error": str(e)}


