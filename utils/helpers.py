from datetime import datetime

def get_timestamp() -> str:
    """Returns current time as readable string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_agent_response(agent_name: str, result: str) -> dict:
    """Wraps any agent result in a standard format"""
    return {
        "agent": agent_name,
        "result": result,
        "timestamp": get_timestamp()
    }

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncates long text - prevents token overflow in LLM calls"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"