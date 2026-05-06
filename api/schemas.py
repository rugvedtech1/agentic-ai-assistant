from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    """What the user sends TO our API"""
    query: str                          # The text question
    model: Optional[str] = "gpt-4o-mini"  # Which AI model to use
    include_vision: Optional[bool] = False # Whether image is included

class AgentResponse(BaseModel):
    """What our API sends BACK to the user"""
    status: str                         # "success" or "error"
    query: str                          # Original question echoed back
    result: Optional[str] = None        # The final answer
    steps: Optional[list] = []          # Which agents ran
    model_used: str = ""                # Which model was used
    error: Optional[str] = None         # Error message if something failed