from langchain_core.prompts import PromptTemplate
from utils.helpers import format_agent_response
from core.logging_config import logger

def run_search(query: str, llm, plan: str = "") -> dict:
    """
    Search Agent - Second agent in the pipeline.
    In mock mode: simulates web search results.
    With real keys: uses Tavily to search the web.
    """
    logger.info(f"Search Agent started | query={query[:50]}")

    prompt = PromptTemplate(
        input_variables=["query", "plan"],
        template="""You are a web research agent.
Research plan: {plan}
Search query: {query}

Find and return the most relevant information.
Include key facts, recent developments, and important details."""
    )

    try:
        chain = prompt | llm
        result = chain.invoke({"query": query, "plan": plan})

        if hasattr(result, 'content'):
            result = result.content

        logger.info("Search Agent completed successfully")
        return format_agent_response("search", str(result))

    except Exception as e:
        logger.error(f"Search Agent failed: {str(e)}")
        return format_agent_response("search", f"Search failed: {str(e)}")