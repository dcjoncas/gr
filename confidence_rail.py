from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client (do not export this directly to avoid circular import)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment.")
chatgpt_client = OpenAI(api_key=openai_api_key)

# W5 patented confidence evaluation function
def confidence_rail(
    input_query: str,
    ai_output: str,
    ai_client,
    client_type: str = "CHATGPT",
    confidence_threshold: int = 90,
    criteria: str = ""
):
    try:
        client_type = client_type.upper()
        responseSuccess = True
        numResponse = 0

        # Compose prompt
        query = (
            'Respond with only an integer (0-100). No extra text. '
            'How confident are you that the following AI output matches the user prompt?\n'
            f'User Prompt: "{input_query}"\nAI Response: "{ai_output}"'
        )
        if criteria:
            query += f"\nUse this evaluation criteria: {criteria}"

        if client_type == "CHATGPT":
            while responseSuccess:
                result = ai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}]
                )
                content = result.choices[0].message.content.strip()
                try:
                    numResponse = int(''.join(c for c in content if c.isdigit()))
                    responseSuccess = False
                except Exception:
                    continue
        else:
            raise ValueError("Unsupported client type for confidence rail.")

        return [numResponse >= confidence_threshold, numResponse]

    except Exception as e:
        raise Exception(f"Error in Confidence Rail: {str(e)}")
