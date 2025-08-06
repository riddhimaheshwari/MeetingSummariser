import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_and_format(transcript: str) -> str:
    prompt = f"""
You are a professional meeting assistant. 
Read the transcript below and produce a structured meeting summary in this format:

📋 Summary: [Concise and accurate high-level summary 7-10 line pointers]

🗂️ Agenda Items:
- [item 1]
- [item 2]

✅ Decisions Made:
- [decision 1]
- [decision 2]

📌 Action Items:
- [Responsible Person]: [task] by [date if known]

Only include details that are explicitly stated in the transcript.

Transcript:
{transcript}
    """

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
