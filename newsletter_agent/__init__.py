"""Newsletter Agent: an autonomous LangGraph agent that researches, writes,
critiques and "sends" a newsletter from a single plain-English goal.

    >>> from newsletter_agent import run_newsletter_agent
    >>> result = run_newsletter_agent(
    ...     "Create a weekly newsletter on latest AI agent news and send it to our subscribers."
    ... )
"""

from newsletter_agent.agent import NewsletterAgent, NewsletterResult, run_newsletter_agent
from newsletter_agent.config import DEFAULT_GOAL, Settings, load_settings
from newsletter_agent.models import HumanDecision
from newsletter_agent.state import AgentEvent, AgentMode

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_GOAL",
    "AgentEvent",
    "AgentMode",
    "HumanDecision",
    "NewsletterAgent",
    "NewsletterResult",
    "Settings",
    "load_settings",
    "run_newsletter_agent",
]
