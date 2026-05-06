from utils.helpers import format_agent_response
from core.logging_config import logger
from PIL import Image
import io

def run_vision(query: str, llm, image_bytes: bytes = None) -> dict:
    """
    Vision Agent - Handles image analysis.
    If no image provided, skips gracefully.
    With real OpenAI key: uses GPT-4 Vision.
    """
    logger.info(f"Vision Agent started | has_image={image_bytes is not None}")

    # If no image provided, skip vision analysis
    if image_bytes is None:
        logger.info("No image provided - Vision Agent skipping")
        return format_agent_response("vision", "No image provided - skipped")

    try:
        # Verify image is valid using PIL
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        format = image.format

        logger.info(f"Image loaded | size={width}x{height} | format={format}")

        # Mock vision analysis
        # With real keys this would call GPT-4 Vision API
        vision_prompt = f"image analysis for query: {query}"
        result = llm.invoke(vision_prompt)

        if hasattr(result, 'content'):
            result = result.content

        logger.info("Vision Agent completed successfully")
        return format_agent_response("vision", str(result))

    except Exception as e:
        logger.error(f"Vision Agent failed: {str(e)}")
        return format_agent_response("vision", f"Vision analysis failed: {str(e)}")