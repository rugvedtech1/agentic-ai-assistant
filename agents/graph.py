from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from agents.planner import run_planner
from agents.search import run_search
from agents.vision import run_vision
from agents.summarizer import run_summarizer
from agents.report import run_report
from core.logging_config import logger

class AgentState(TypedDict):
    query: str
    model: str
    image_bytes: Optional[bytes]
    plan: str
    search_results: str
    vision_result: str
    summary: str
    final_report: str
    steps_completed: list
    error: Optional[str]

def get_llm(model: str):
    return ChatOpenAI(model=model, temperature=0.7)

def planner_node(state: AgentState) -> AgentState:
    logger.info("Graph: entering planner node")
    llm = get_llm(state["model"])
    result = run_planner(state["query"], llm)
    state["plan"] = result["result"]
    state["steps_completed"].append("planner")
    return state

def vision_node(state: AgentState) -> AgentState:
    """Vision runs BEFORE search so results feed into search"""
    logger.info("Graph: entering vision node")
    llm = get_llm(state["model"])
    result = run_vision(state["query"], llm, state.get("image_bytes"))
    state["vision_result"] = result["result"]
    state["steps_completed"].append("vision")
    return state

def search_node(state: AgentState) -> AgentState:
    """Search uses vision results to find info about what's in image"""
    logger.info("Graph: entering search node")
    llm = get_llm(state["model"])
    result = run_search(
        state["query"],
        llm,
        state["plan"],
        state["vision_result"]  # ← pass vision results here!
    )
    state["search_results"] = result["result"]
    state["steps_completed"].append("search")
    return state

def summarizer_node(state: AgentState) -> AgentState:
    logger.info("Graph: entering summarizer node")
    llm = get_llm(state["model"])
    result = run_summarizer(state["query"], llm, state["search_results"])
    state["summary"] = result["result"]
    state["steps_completed"].append("summarizer")
    return state

def report_node(state: AgentState) -> AgentState:
    logger.info("Graph: entering report node")
    llm = get_llm(state["model"])
    result = run_report(
        state["query"], llm,
        state["summary"],
        state["vision_result"]
    )
    state["final_report"] = result["result"]
    state["steps_completed"].append("report")
    return state

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("vision", vision_node)    # ← vision runs first now!
    graph.add_node("search", search_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("report", report_node)

    # New order: planner → vision → search → summarizer → report
    graph.set_entry_point("planner")
    graph.add_edge("planner", "vision")
    graph.add_edge("vision", "search")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "report")
    graph.add_edge("report", END)
    return graph.compile()

def run_agent_pipeline(
    query: str,
    model: str = "gpt-4o-mini",
    image_bytes: bytes = None
) -> dict:
    logger.info(f"Pipeline starting | query={query[:50]}")
    graph = build_graph()
    initial_state = AgentState(
        query=query,
        model=model,
        image_bytes=image_bytes,
        plan="",
        search_results="",
        vision_result="",
        summary="",
        final_report="",
        steps_completed=[],
        error=None
    )
    try:
        final_state = graph.invoke(initial_state)
        logger.info(f"Pipeline completed | steps={final_state['steps_completed']}")
        return {
            "status": "success",
            "final_report": final_state["final_report"],
            "steps_completed": final_state["steps_completed"]
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return {
            "status": "error",
            "final_report": "",
            "steps_completed": [],
            "error": str(e)
        }