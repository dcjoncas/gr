from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from confidence_rail import confidence_rail
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# FastAPI app
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Initialize OpenAI client (stable v1.30+)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")
openai_client = OpenAI(api_key=openai_api_key)

# Request models
class GenerateInput(BaseModel):
    prompt: str
    clientType: str

class TestInput(BaseModel):
    prompt: str
    response: str
    criteria: str
    clientType: str
    confidenceThreshold: int

# Serve HTML
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("index.html", "r") as f:
        return f.read()

# Generate endpoint
@app.post("/generate")
async def generate_response(input: GenerateInput):
    try:
        logger.info(f"Generating response for client: {input.clientType}")
        if input.clientType.upper() == "CHATGPT":
            result = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input.prompt}]
            )
            response_text = result.choices[0].message.content
        else:
            raise HTTPException(status_code=400, detail="Unsupported clientType")
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

# Confidence testing endpoint
@app.post("/test")
async def test_confidence(input: TestInput):
    try:
        logger.info(f"Testing confidence for client: {input.clientType}")
        if input.clientType.upper() != "CHATGPT":
            raise HTTPException(status_code=400, detail="Unsupported clientType")

        passed, score = confidence_rail(
            input_query=input.prompt,
            ai_output=input.response,
            ai_client=openai_client,
            client_type=input.clientType,
            confidence_threshold=input.confidenceThreshold,
            criteria=input.criteria,
        )
        return {"passed": passed, "score": score}
    except Exception as e:
        logger.error(f"Error in /test: {e}")
        raise HTTPException(status_code=500, detail=f"Confidence test error: {e}")
