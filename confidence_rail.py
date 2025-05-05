from openai import OpenAI
import os

# Function to initialize the OpenAI client
def get_chatgpt_client():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=openai_api_key)

# Function to check the quality of the AI's response.
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
        numResponse = ""

        queryAI = (
            f'Respond to the following question with only an integer number from 0 to 100. ADD NO ADDITIONAL TEXT. USE NO LETTERS. '
            f'How confident are you that the following AI output matches up to the corresponding user prompt?\n'
            f'User Prompt: "{input_query}"\nAI Response: "{ai_output}"'
        )

        if criteria:
            queryAI += f"\nUse this criteria to make your assessment: {criteria}"

        if client_type == "CHATGPT":
            while responseSuccess:
                response = ai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": queryAI}],
                )
                numResponse = str(response.choices[0].message.content)
                try:
                    numResponse = int(''.join(c for c in numResponse if c.isdigit()))
                    responseSuccess = False
                except Exception:
                    continue

        elif client_type == "GROK":
            while responseSuccess:
                response = ai_client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[{"role": "user", "content": queryAI}],
                )
                numResponse = str(response.choices[0].message.content)
                try:
                    numResponse = int(''.join(c for c in numResponse if c.isdigit()))
                    responseSuccess = False
                except Exception:
                    continue

        else:
            print("AI client type invalid.")
            return [True, 0]

        return [True, numResponse] if numResponse >= confidence_threshold else [False, numResponse]

    except Exception as err:
        raise Exception("Error in ConfidenceRail: " + str(err))
