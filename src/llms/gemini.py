import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def llm_tool(query: str) -> str:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[f"Answer this query briefly: {query}"]
    )

    return response.text
