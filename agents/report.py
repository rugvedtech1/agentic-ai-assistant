from langchain_core.prompts import PromptTemplate
from utils.helpers import format_agent_response, get_timestamp
from core.logging_config import logger

def run_report(query: str, llm, summary: str = "", vision_result: str = "") -> dict:
    """
    Report Generator Agent - Final agent in pipeline.
    Combines all previous agent outputs into
    one clean professional research report.
    """
    logger.info("Report Agent started")

    prompt = PromptTemplate(
        input_variables=["query", "summary", "vision_result", "timestamp"],
        template="""You are a professional report writer.
Query: {query}
Research Summary: {summary}
Visual Analysis: {vision_result}
Generated at: {timestamp}

Write a professional, well-structured research report.
Include: Overview, Key Findings, and Conclusion."""
    )

    try:
        chain = prompt | llm
        result = chain.invoke({
            "query": query,
            "summary": summary,
            "vision_result": vision_result,
            "timestamp": get_timestamp()
        })

        if hasattr(result, 'content'):
            result = result.content

        logger.info("Report Agent completed successfully")
        return format_agent_response("report", str(result))

    except Exception as e:
        logger.error(f"Report Agent failed: {str(e)}")
        return format_agent_response("report", f"Report generation failed: {str(e)}")