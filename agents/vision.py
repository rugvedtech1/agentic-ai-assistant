from utils.helpers import format_agent_response
from core.logging_config import logger
from core.config import settings
from PIL import Image
import io
import base64
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def identify_person_with_serpapi(image_bytes: bytes) -> str:
    """
    Uses SerpApi Google Reverse Image Search to identify
    who is in the image.
    """
    try:
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # SerpApi reverse image search
        from serpapi import GoogleSearch
        
        # First upload image to get URL using imgur or use base64
        # We'll use SerpApi's image search with base64
        params = {
            "engine": "google_reverse_image",
            "image_base64": base64_image,
            "api_key": settings.SERPAPI_KEY
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract person name from results
        person_name = ""
        
        # Check knowledge graph
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            person_name = kg.get("title", "")
            logger.info(f"Person identified via knowledge graph: {person_name}")
        
        # Check image results title
        if not person_name and "image_results" in results:
            first_result = results["image_results"][0]
            person_name = first_result.get("title", "")
            logger.info(f"Person identified via image results: {person_name}")

        # Check organic results
        if not person_name and "organic_results" in results:
            first_organic = results["organic_results"][0]
            person_name = first_organic.get("title", "")
            logger.info(f"Person identified via organic results: {person_name}")

        return person_name if person_name else "Unknown person"

    except Exception as e:
        logger.error(f"SerpApi identification failed: {str(e)}")
        return "Unknown person"


def analyze_image_with_gpt4(image_bytes: bytes, query: str, person_name: str = "") -> str:
    """
    Uses GPT-4o to analyze image visually.
    Includes person name hint if identified.
    """
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        context = f"The person in this image is identified as: {person_name}. " if person_name and person_name != "Unknown person" else ""
        
        message = HumanMessage(content=[
            {
                "type": "text",
                "text": f"{context}Analyze this image in context of: {query}. Describe what you see including any sports equipment, jersey, setting, or other relevant visual details."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ])
        
        result = llm.invoke([message])
        return result.content
        
    except Exception as e:
        logger.error(f"GPT-4o vision failed: {str(e)}")
        return f"Visual analysis unavailable: {str(e)}"


def run_vision(query: str, llm, image_bytes: bytes = None) -> dict:
    """
    Vision Agent:
    1. Uses SerpApi to identify WHO is in the image
    2. Uses GPT-4o to describe what's in the image
    3. Returns both for Search Agent to use
    """
    logger.info(f"Vision Agent started | has_image={image_bytes is not None}")

    if image_bytes is None:
        logger.info("No image provided - Vision Agent skipping")
        return format_agent_response("vision", "No image provided - skipped")

    try:
        # Step 1 — Identify person using SerpApi
        logger.info("Identifying person using SerpApi reverse image search...")
        person_name = identify_person_with_serpapi(image_bytes)
        logger.info(f"Person identified: {person_name}")

        # Step 2 — Analyze image with GPT-4o
        visual_analysis = analyze_image_with_gpt4(image_bytes, query, person_name)

        # Step 3 — Combine results
        combined_result = f"""
IDENTIFIED PERSON: {person_name}

VISUAL ANALYSIS:
{visual_analysis}

SEARCH TARGET: {person_name}
"""
        logger.info("Vision Agent completed successfully")
        return format_agent_response("vision", combined_result)

    except Exception as e:
        logger.error(f"Vision Agent failed: {str(e)}")
        return format_agent_response("vision", f"Vision analysis failed: {str(e)}")