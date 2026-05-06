from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

from core.config import settings
from core.logging_config import logger
from api.schemas import QueryRequest, AgentResponse

# Create FastAPI app instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-Agent Research + Vision Assistant"
)

# CORS middleware - allows browser/frontend to talk to this API
# In production this would be restricted to your domain only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check - tells you the app is alive"""
    logger.info("Health check called")
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health():
    """Used by Docker and AWS to check if container is healthy"""
    return {"status": "healthy"}

@app.post("/query", response_model=AgentResponse)
async def query(
    query: str = Form(...),
    model: Optional[str] = Form("gpt-4o-mini"),
    image: Optional[UploadFile] = File(None)
):
    """
    Main endpoint - receives text query + optional image
    Runs through the multi-agent pipeline
    Returns final research report
    """
    logger.info(f"Query received | model={model} | has_image={image is not None}")

    try:
        # For now we return a placeholder
        # Tomorrow we wire in the real LangGraph agents
        return AgentResponse(
            status="success",
            query=query,
            result=f"Received your query: '{query}'. Agents coming in Day 2!",
            steps=["planner", "search", "summarizer"],
            model_used=model,
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return AgentResponse(
            status="error",
            query=query,
            error=str(e),
            model_used=model,
        )

# ─── Run ──────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)