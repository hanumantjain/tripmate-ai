import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp/",
            "headers": {"Authorization": f"Bearer {tavily_api_key}"},
        }
    }
)

async def get_all_tools():
    tools = await client.get_tools()
    print("Available tools:")
    for tool in tools:
        print(tool.name)
    return tools

tavily_search_tool = None

async def get_tavily_search_tool():
    global tavily_search_tool
    if tavily_search_tool is None:
        tools = await get_all_tools()
        for tool in tools:
            if tool.name == "tavily_search":
                tavily_search_tool = tool
                break
    return tavily_search_tool

async def tavily_mcp_search(query):
    await get_tavily_search_tool()
    result = await tavily_search_tool.ainvoke(
        {
            "query": query
        }
    )
    return result