import os
import certifi
from dotenv import load_dotenv

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from tools.tavliy_tool import tavily_search
from tools.flight_tool import search_flights

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

# =====================
# LLM
# =====================

llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=os.getenv("OPEN_AI_API_KEY"),
)

# =====================
# State
# =====================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# =====================
# Flight Agent
# =====================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =====================
# Search Agent
# =====================

def search_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_result = tavily_search(query)

    return {
        "hotel_results": hotel_result,
        "messages": [
            AIMessage(content="Hotel results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =====================
# Itinerary Agent
# =====================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query: {state['user_query']}
Flight Results: {state['flight_results']}
Hotel Results: {state['hotel_results']}

Make the itinerary practical, detailed, and user-friendly. Include day-by-day activities, travel tips, and any other relevant information.
"""

    response = llm.invoke([
        SystemMessage(content="You are a travel assistant that creates detailed itineraries."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =====================
# Final Response Agent
# =====================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate ta final travel response for the user.
User Query: {state['user_query']}
Flight Results: {state['flight_results']}
Hotel Results: {state['hotel_results']}
Itinerary: {state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavialable.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a travel assistant that creates detailed itineraries."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =====================
# Build Graph
# =====================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("search_agent", search_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "search_agent")
graph.add_edge("search_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =====================
# PostgreSQL Checkpointer
# =====================

DatabaseURL = get_database_url()

_conn = psycopg.connect(
    DatabaseURL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

# =====================
# Functin for FastAPI
# =====================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )
    final_answer = result["messages"][-1].pcontent
    return {
            "thread_id": thread_id,
            "answer": final_answer,
            "flight_results": result.get("flight_results", ""),
            "hotel_results": result.get("hotel_results", ""),
            "itinerary": result.get("itinerary", ""),
            "llm_calls": result.get("llm_calls", 0)
        }
    
    