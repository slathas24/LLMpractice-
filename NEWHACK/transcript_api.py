import openai
import os
#OPENAI_API_KEY="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"
openai_api_key="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"

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

