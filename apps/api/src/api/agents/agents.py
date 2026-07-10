from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config

def run_llm(provider, model_name, messages, max_tokens=500):
    if provider == "openai":
        client = OpenAI(api_key=config.OPEN_API_KEY)
    elif provider == "google":
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
    else:
        client = Groq(api_key=config.GROQ_API_KEY)
    
    if provider == "google":
        return  client.models.generate_content(
                contents=[message["content"] for message in messages],
                model=model_name,
            ).text

    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
        ).choices[0].message.content