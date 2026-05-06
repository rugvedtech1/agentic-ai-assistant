from langchain_core.prompts import PromptTemplate
from utils.helpers import format_agent_response
from core.logging_config import logger

def run_planner(query: str, llm) -> dict:
    """
    Planner Agent - First agent in the pipeline.
    Takes the user query and creates a research plan.
    Decides what needs to be searched and analyzed.
    """
    logger.info(f"Planner Agent started | query={query[:50]}")

    prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are a research planner.
Given this query: {query}

Create a clear step by step research plan.
What should be searched? What should be analyzed?
Be specific and structured in your plan."""
    )

    try:
        chain = prompt | llm
        result = chain.invoke({"query": query})

        # Handle both string and object responses
        if hasattr(result, 'content'):
            result = result.content

        logger.info("Planner Agent completed successfully")
        return format_agent_response("planner", str(result))

    except Exception as e:
        logger.error(f"Planner Agent failed: {str(e)}")
        return format_agent_response("planner", f"Planning failed: {str(e)}")