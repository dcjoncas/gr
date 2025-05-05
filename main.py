from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from confidence_rail import confidence_rail, get_chatgpt_client
from dotenv import load_dotenv
import logging
import os

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Optional: Serve static files if you have them in a "static" folder
# Uncomment this if you're serving assets like .css or .js
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Attempt to get the OpenAI client
try:
    chatgpt_client = get_chatgpt_client()
except Exception as e:
    logger.warning(f"OpenAI client not initialized: {str(e)}")
    chatgpt_client = None

# Request schemas
class GenerateInput(BaseModel):
    prompt: str
    clientType: str

class TestInput(BaseModel):
    prompt: str
    response: str
    criteria: str
    clientType: str
    confidenceThreshold: int

# Serve HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Could not read index.html: {e}")
        raise HTTPException(status_code=500, detail="Could not load index.html")

# Endpoint for generating AI response
@app.post("/generate")
async def generate_response(input: GenerateInput):
    try:
        if not chatgpt_client:
            raise HTTPException(status_code=503, detail="OpenAI service unavailable.")

        logger.info(f"Generating response for client: {input.clientType}")
        if input.clientType.upper() == "CHATGPT":
            result = chatgpt_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input.prompt}]
            )
            response = result.choices[0].message.content
        else:
            raise HTTPException(status_code=400, detail="Invalid client type")

        return {"response": response}

    except Exception as e:
        logger.error(f"Error in /generate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# Endpoint for confidence checking
@app.post("/test")
async def test_confidence(input: TestInput):
    try:
        if not chatgpt_client:
            raise HTTPException(status_code=503, detail="OpenAI service unavailable.")

        logger.info(f"Testing confidence for client: {input.clientType}")
        if input.clientType.upper() != "CHATGPT":
            raise HTTPException(status_code=400, detail="Invalid client type")

        result, score = confidence_rail(
            input_query=input.prompt,
            ai_output=input.response,
            ai_client=chatgpt_client,
            client_type=input.clientType,
            confidence_threshold=input.confidenceThreshold,
            criteria=input.criteria
        )

        return {"passed": result, "score": score}

    except Exception as e:
        logger.error(f"Error in /test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test confidence: {str(e)}")
