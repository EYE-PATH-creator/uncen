from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from openai import OpenAI

app = FastAPI(
    title="Uncensored AI API",
    description="OpenAI-compatible proxy for https://uncensored.chat",
    version="1.0.0"
)

# Best practice: Use environment variable (set this in Vercel dashboard)
UNCENSORED_API_KEY = os.getenv(
    "UNCENSORED_API_KEY",
    "50cf193b3195f8ee773d49a41af20da00ddf1915b2f175c1ee68e5708f41fcf1"  # fallback only for testing
)

client = OpenAI(
    base_url="https://uncensored.chat/api/v1",
    api_key=UNCENSORED_API_KEY
)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "uncensored-v2"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upstream error: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Uncensored AI API is live on Vercel",
        "endpoint": "/v1/chat/completions",
        "provider": "uncensored.chat"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
