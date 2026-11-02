# core/agents/__init__.py
"""
Package chứa 10 Agent chuyên biệt trong kiến trúc Multi-Agent LangGraph.
Nhóm 1 (Song song): IntentClassifier, KnowledgeBase, Calculator, DateTime, WebSearch, Analyst
Nhóm 2 (Tuần tự):   Planner, Validator, ResponseGenerator
Bộ điều phối:        Router (ThreadPool Fan-out / Fan-in)
"""

from .base import AgentBase
from .intent import IntentClassifierAgent
from .rag import KnowledgeBaseAgent
from .math_agent import CalculatorAgent
from .datetime_agent import DateTimeAgent
from .search import WebSearchAgent
from .analyst import AnalystAgent
from .planner import PlannerAgent
from .validator import ValidatorAgent
from .response import ResponseGeneratorAgent

__all__ = [
    "AgentBase",
    "IntentClassifierAgent",
    "KnowledgeBaseAgent",
    "CalculatorAgent",
    "DateTimeAgent",
    "WebSearchAgent",
    "AnalystAgent",
    "PlannerAgent",
    "ValidatorAgent",
    "ResponseGeneratorAgent",
]
