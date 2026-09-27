"""The agent's tool belt.

LLM-callable (the model chooses when and how to use them):
    * ``search_news``         Bing News RSS, falling back to Google News RSS
    * ``search_hacker_news``  Hacker News (Algolia API), ranked by community points
    * ``search_web``          Tavily web search (only when TAVILY_API_KEY is set)

Pipeline tools (invoked by graph nodes):
    * ``read_article``        fetch a page and extract its main text (trafilatura)
    * ``summarize_articles``  LLM summariser producing faithful, structured briefs
    * ``render_newsletter``   HTML / Markdown / plain-text generator (Jinja2)
    * ``send_newsletter``     MIME email builder that "sends" to the outbox
"""

from newsletter_agent.tools.mailer import send_newsletter
from newsletter_agent.tools.reader import read_article
from newsletter_agent.tools.renderer import render_newsletter
from newsletter_agent.tools.research import build_research_tools
from newsletter_agent.tools.summarizer import summarize_articles

__all__ = [
    "build_research_tools",
    "read_article",
    "render_newsletter",
    "send_newsletter",
    "summarize_articles",
]
