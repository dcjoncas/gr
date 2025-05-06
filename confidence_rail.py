from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hi"}]
)


def confidence_rail(
    input_query: str,
    ai_output: str,
    client_type: str = "CHATGPT",
    confidence_threshold: int = 90,
    criteria: str = ""
):
    try:
        query = (
            f"Respond with only a number from 0 to 100. No extra text.\n"
            f"Prompt: {input_query}\nResponse: {ai_output}"
        )
        if criteria:
            query += f"\nCriteria: {criteria}"

        if client_type.upper() == "CHATGPT":
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}]
            )
            confidence = result.choices[0].message["content"].strip()
            confidence_score = int(''.join(filter(str.isdigit, confidence)))
            return [confidence_score >= confidence_threshold, confidence_score]

        else:
            return [True, 0]

    except Exception as e:
        logger.error(f"confidence_rail error: {e}")
        raise
