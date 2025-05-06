from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import logging
import os

from confidence_rail import confidence_rail

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

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
    except:
        raise HTTPException(status_code=500, detail="Could not load HTML")

@app.post("/generate")
async def generate(input: GenerateInput):
    try:
        if input.clientType.upper() == "CHATGPT":
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input.prompt}]
            )
            return {"response": result.choices[0].message["content"]}
        else:
            raise HTTPException(status_code=400, detail="Invalid clientType")
    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test")
async def test(input: TestInput):
    try:
        if input.clientType.upper() != "CHATGPT":
            raise HTTPException(status_code=400, detail="Invalid clientType")

        passed, score = confidence_rail(
            input_query=input.prompt,
            ai_output=input.response,
            client_type=input.clientType,
            confidence_threshold=input.confidenceThreshold,
            criteria=input.criteria
        )
        return {"passed": passed, "score": score}
    except Exception as e:
        logger.error(f"Error in /test: {e}")
        raise HTTPException(status_code=500, detail=str(e))
