from langchain_core.prompts import PromptTemplate
from utils.helpers import format_agent_response, truncate_text
from core.logging_config import logger

def run_summarizer(query: str, llm, search_results: str = "") -> dict:
    """
    Summarizer Agent - Takes raw search results
    and creates a clean focused summary.
    """
    logger.info("Summarizer Agent started")

    prompt = PromptTemplate(
        input_variables=["query", "search_results"],
        template="""You are an expert summarizer.
Original query: {query}
Raw research data: {search_results}

Create a clear, concise summary of the most important findings.
Focus on what directly answers the query."""
    )

    try:
        chain = prompt | llm
        result = chain.invoke({
            "query": query,
            "search_results": truncate_text(search_results)
        })

        if hasattr(result, 'content'):
            result = result.content

        logger.info("Summarizer Agent completed successfully")
        return format_agent_response("summarizer", str(result))

    except Exception as e:
        logger.error(f"Summarizer Agent failed: {str(e)}")
        return format_agent_response("summarizer", f"Summarization failed: {str(e)}")