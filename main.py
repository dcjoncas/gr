from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from openai import OpenAI
from confidence_rail import confidence_rail

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI()

class GenerateInput(BaseModel):
    prompt: str
    clientType: str

class TestInput(BaseModel):
    prompt: str
    response: str
    criteria: str
    clientType: str
    confidenceThreshold: int

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Could not load HTML")

@app.post("/generate")
async def generate(input: GenerateInput):
    try:
        if input.clientType.upper() == "CHATGPT":
            result = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input.prompt}]
            )
            return {"response": result.choices[0].message.content}
        else:
            raise HTTPException(status_code=400, detail="Invalid client type")
    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test")
async def test_confidence(input: TestInput):
    try:
        if input.clientType.upper() != "CHATGPT":
            raise HTTPException(status_code=400, detail="Invalid client type")

        passed, score = confidence_rail(
            input_query=input.prompt,
            ai_output=input.response,
            ai_client=openai_client,
            client_type=input.clientType,
            confidence_threshold=input.confidenceThreshold,
            criteria=input.criteria
        )
        return {"passed": passed, "score": score}
    except Exception as e:
        logger.error(f"Error in /test: {e}")
        raise HTTPException(status_code=500, detail=str(e))
