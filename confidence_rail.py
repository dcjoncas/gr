import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
chatgpt_client = OpenAI()

def confidence_rail(input_query, ai_output, ai_client, client_type="CHATGPT", confidence_threshold=90, criteria=""):
    try:
        query = (
            'Respond with only an integer (0-100). No extra text. '
            'How confident are you that the following AI output matches the user prompt?\n'
            f'User Prompt: "{input_query}"\nAI Response: "{ai_output}"'
        )
        if criteria:
            query += f"\nUse this evaluation criteria: {criteria}"

        if client_type.upper() == "CHATGPT":
            result = ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}]
            )
            content = result.choices[0].message.content.strip()
            score = int(''.join(c for c in content if c.isdigit()))
            return [score >= confidence_threshold, score]
        else:
            raise ValueError("Unsupported client type for confidence rail.")
    except Exception as e:
        raise Exception(f"Error in Confidence Rail: {str(e)}")