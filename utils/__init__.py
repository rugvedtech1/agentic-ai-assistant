from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from typing import Any, Iterator, List, Optional

class MockLLM(LLM):
    """
    Fake LLM for testing without API keys.
    Returns structured fake responses so we can test
    the full agent pipeline end to end.
    """

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Returns a fake but realistic response based on prompt content"""
        prompt_lower = prompt.lower()

        if "plan" in prompt_lower or "planner" in prompt_lower:
            return """RESEARCH PLAN:
1. Search for recent information on the topic
2. Analyze and extract key findings
3. Identify main themes and patterns
4. Generate comprehensive summary
5. Compile final research report"""

        elif "search" in prompt_lower or "find" in prompt_lower:
            return """SEARCH RESULTS:
- Found 5 relevant sources on the topic
- Source 1: Recent developments show significant progress
- Source 2: Experts agree on core principles
- Source 3: New applications emerging in 2024-2025
- Key finding: The field is rapidly evolving"""

        elif "vision" in prompt_lower or "image" in prompt_lower:
            return """IMAGE ANALYSIS:
- Image contains visual information relevant to query
- Detected: diagrams, text, or visual patterns
- Key visual elements identified and extracted
- Image context integrated with text query"""

        elif "summar" in prompt_lower:
            return """SUMMARY:
The research reveals important insights on this topic.
Multiple sources confirm the key findings.
The information is current and highly relevant.
Further investigation recommended in specific areas."""

        elif "report" in prompt_lower:
            return """FINAL REPORT:
# Research Findings

## Overview
Comprehensive analysis completed successfully.

## Key Findings
1. Topic has significant real-world applications
2. Recent developments show rapid advancement
3. Multiple expert sources confirm findings

## Conclusion
Research objective achieved with high confidence.
Ready for production use once API keys are configured."""

        else:
            return f"Mock response for: {prompt[:100]}..."

    @property
    def _identifying_params(self) -> dict:
        return {"model": "mock"}