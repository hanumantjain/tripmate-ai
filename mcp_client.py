import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

openweather_api_key = os.getenv("OPEN_WEATHER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
aviation_api_key = os.getenv("AVIATION_API_KEY")


client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp/",
            "headers": {"Authorization": f"Bearer {tavily_api_key}"},
        },
        "aviation":{
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "aviationstack-mcp",
            ],
            "env": {
                "AVIATIONSTACK_API_KEY": aviation_api_key,
            }
        },
        "weather":{
            "transport":"stdio",
            "command": "/Users/hanumant/Dev/TripPlanner/tripmate-ai/.venv/bin/python",
            "args":[
                "/Users/hanumant/Dev/TripPlanner/tripmate-ai/custom_weather_mcp.py"
            ],
            "env": {
                "OPEN_WEATHER_API_KEY": openweather_api_key
            }
        },
    }
)


async def get_all_tools():
    tools = await client.get_tools()
    print("Available tools:")
    for tool in tools:
        print(tool.name)
    return tools

search_tool = None
aviation_tool = {}

async def intialize_mcp():
    global search_tool, aviation_tool
    if search_tool is not None and aviation_tool:
        return

    tools = await client.get_tools()

    for tool in tools:
        print(tool.name)

    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    aviation_tool = {
        tool.name: tool
        for tool in tools 
        if tool.name != "tavil_search"
    }

async def tavily_mcp_search(query):
    await intialize_mcp()
    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )
    return result

async def aviation_mcp_call(
        tool_name: str,
        tool_args: dict = None
):
    tools = await client.get_tools()
    tool = next(
        t for t in tools
        if t.name == tool_name
    )
    result = await tool.ainvoke(
        tool_args or {}
    )
    return result

#######################
# Weather Tools
#######################

weather_tool = None
forecast_tool = None

async def initialize_weather_tool():
    global weather_tool, forecast_tool

    if weather_tool is not None:
        return

    tools = await client.get_tools()

    weather_tool = next(
        t for t in tools
        if t.name == "get_current_weather"
    )

    forecast_tool = next(
        t for t in tools
        if t.name == "get_forecast"
    )


async def weather_mcp_search(city:str):
    await initialize_weather_tool()
    return await weather_tool.ainvoke(
        {
            "city":city
        }
    )

async def forecast_mcp_search(city:str):
    await initialize_weather_tool()
    return await forecast_tool.ainvoke(
        {
            "city":city
        }
    )

#######################
# Destination Extractor 
#######################


llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=os.getenv("OPEN_AI_API_KEY"),
)

def extract_destination(query: str):
    prompt = f"""
    Extract only the destination city or country.

    Query: {query}
    Return only destination name.
    """

    response = llm.invoke(prompt)
    return response.content.strip()
    