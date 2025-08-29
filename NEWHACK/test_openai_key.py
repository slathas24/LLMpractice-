import openai

OPENAI_API_KEY = "sk-proj-"
def test_openai_key():
    openai.api_key = OPENAI_API_KEY
    try:
        models = openai.Model.list()
        print("API key is valid. Models available:", [m.id for m in models.data])
        return True
    except Exception as e:
        print("API key is invalid or there is a connection issue:", e)
        return False

if __name__ == "__main__":
    test_openai_key()

