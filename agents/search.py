from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from utils.helpers import format_agent_response
from core.logging_config import logger
from core.config import settings
import os

def run_search(query: str, llm, plan: str = "", vision_context: str = "") -> dict:
    """
    Search Agent - Uses Tavily to search the web.
    If vision_context is provided, searches based on what was seen in image.
    """
    logger.info(f"Search Agent started | query={query[:50]}")

    try:
        # Build smart search query
        if vision_context and vision_context != "No image provided - skipped":
            # Extract key subject from vision result for better search
            extract_prompt = f"""From this image analysis, extract ONLY the main subject name 
            (person name, object, place) in 3-5 words maximum:
            {vision_context[:300]}
            
            Return ONLY the search term, nothing else."""
            
            search_term = llm.invoke(extract_prompt)
            if hasattr(search_term, 'content'):
                search_term = search_term.content.strip()
            logger.info(f"Vision-based search term: {search_term}")
        else:
            search_term = query

        # Use Tavily for real web search
        os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
        tavily = TavilySearchResults(max_results=5)
        search_results = tavily.invoke(search_term)

        # Format results
        formatted = ""
        for i, result in enumerate(search_results):
            formatted += f"\nSource {i+1}: {result.get('url', '')}\n"
            formatted += f"Content: {result.get('content', '')}\n"

        logger.info(f"Search Agent found {len(search_results)} results")
        return format_agent_response("search", formatted)

    except Exception as e:
        logger.error(f"Search Agent failed: {str(e)}")
        # Fallback to LLM if Tavily fails
        prompt = PromptTemplate(
            input_variables=["query", "vision_context"],
            template="""Research this topic: {query}
            Additional context from image: {vision_context}
            Provide detailed information."""
        )
        chain = prompt | llm
        result = chain.invoke({"query": query, "vision_context": vision_context})
        if hasattr(result, 'content'):
            result = result.content
        return format_agent_response("search", str(result))