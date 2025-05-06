from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from confidence_rail import confidence_rail, chatgpt_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

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
async def serve_html():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/generate")
async def generate_response(input: GenerateInput):
    try:
        logger.info(f"Generating response for client: {input.clientType}")
        if input.clientType.upper() == "CHATGPT":
            result = chatgpt_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input.prompt}]
            )
            return {"response": result.choices[0].message.content.strip()}
        else:
            raise HTTPException(status_code=400, detail="Invalid client type")
    except Exception as e:
        logger.error(f"Error in /generate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.post("/test")
async def test_confidence(input: TestInput):
    try:
        logger.info(f"Testing confidence for client: {input.clientType}")
        if input.clientType.upper() != "CHATGPT":
            raise HTTPException(status_code=400, detail="Invalid client type")
        passed, score = confidence_rail(
            input_query=input.prompt,
            ai_output=input.response,
            ai_client=chatgpt_client,
            client_type=input.clientType,
            confidence_threshold=input.confidenceThreshold,
            criteria=input.criteria
        )
        return {"passed": passed, "score": score}
    except Exception as e:
        logger.error(f"Error in /test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test confidence: {str(e)}")