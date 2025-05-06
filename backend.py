from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
from ai_agent import get_response_from_ai_agent  # Ensure this module is available

# Pydantic Models for Schema Validation

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 30
    cfg_scale: float = 7.0
    samples: int = 1

# API Configuration
ALLOWED_MODEL_NAMES =  ["llama3-70b-8192", "llama-3.3-70b-versatile"]
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_API_HOST = "https://api.stability.ai"

# FastAPI App Initialization
app = FastAPI(title="LangGraph AI Agent")

# AI Chatbot Endpoint
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Create AI Agent and get response from it!
    try:
        response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Agent Error: {str(e)}")

# Image Generation Endpoint
@app.post("/generate-image")
async def generate_image(request: ImageGenerationRequest):
    """
    Generate images using Stability AI API.
    """
    if not STABILITY_API_KEY:
        raise HTTPException(status_code=500, detail="Stability API key not configured")
    
    try:
        engine_id = "stable-diffusion-xl-1024-v1-0"
        api_host = STABILITY_API_HOST
        
        # Send request to Stability AI API for image generation
        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            json={
                "text_prompts": [
                    {"text": request.prompt, "weight": 1},
                    {"text": request.negative_prompt, "weight": -1} if request.negative_prompt else None
                ],
                "cfg_scale": request.cfg_scale,
                "height": request.height,
                "width": request.width,
                "samples": request.samples,
                "steps": request.steps,
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        # Process the response and extract images
        data = response.json()
        images = []
        
        for image in data["artifacts"]:
            images.append({
                "base64": image["base64"],
                "seed": image["seed"],
                "mime_type": "image/png"
            })
        
        return {"images": images}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image Generation Error: {str(e)}")

# Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
