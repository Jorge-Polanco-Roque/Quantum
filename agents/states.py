"""State definitions for the portfolio analysis agent graph."""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class PortfolioDebateState(TypedDict):
    optimistic_history: Annotated[str, "Optimistic analyst conversation history"]
    cautious_history: Annotated[str, "Cautious analyst conversation history"]
    history: Annotated[str, "Full debate conversation history"]
    current_response: Annotated[str, "Latest response in debate"]
    count: Annotated[int, "Number of debate turns"]


class PortfolioAgentState(MessagesState):
    portfolio_data: dict  # weights, metrics, tickers from MC engine
    quant_report: str
    market_context: str
    portfolio_debate_state: PortfolioDebateState
    final_recommendation: str
    sentiment_data: dict  # news sentiment per ticker
    ensemble_data: dict   # results from all optimization methods
