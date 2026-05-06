from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional
import uvicorn

from core.config import settings
from core.logging_config import logger
from api.schemas import QueryRequest, AgentResponse
from agents.graph import run_agent_pipeline

# Create FastAPI app instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-Agent Research + Vision Assistant"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (our UI)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
async def serve_ui():
    return FileResponse("static/index.html")

# ─── Routes ───────────────────────────────────────────────

@app.get("/")
async def root():
    logger.info("Health check called")
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/query", response_model=AgentResponse)
async def query(
    query: str = Form(...),
    model: Optional[str] = Form("gpt-4o-mini"),
    image: Optional[UploadFile] = File(None)
):
    logger.info(f"Query received | model={model} | has_image={image is not None}")

    try:
        # Read image bytes if image was uploaded
        image_bytes = None
        if image is not None:
            image_bytes = await image.read()

        # Run the full 5-agent LangGraph pipeline
        pipeline_result = run_agent_pipeline(
            query=query,
            model=model,
            image_bytes=image_bytes
        )

        return AgentResponse(
            status=pipeline_result["status"],
            query=query,
            result=pipeline_result["final_report"],
            steps=pipeline_result["steps_completed"],
            model_used=model,
            error=pipeline_result.get("error")
        )

    except Exception as e:
        logger.error(f"Query endpoint failed: {str(e)}")
        return AgentResponse(
            status="error",
            query=query,
            error=str(e),
            model_used=model,
        )

# ─── Run ──────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)