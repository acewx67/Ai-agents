from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain.messages import HumanMessage
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tavily_client = TavilyClient()


@tool(
    "websearch_tool",
)
def web_search(query: str) -> Dict[str, Any]:
    """Search the internet for latest information using this tool

    Args:
        query (str): _description_

    Returns:
        Dict[str, Any]: _description_
    """
    return tavily_client.search(query)


config = {"configurable": {"thread_id": 1}}
system_prompt = "You are a Master Chef, well versed in all kinds of cooking, help the user by using latest knowledge from internet to help the user cook the best tasting dish possible with their limited resources in 100 words or less."
agent = create_agent(
    model=model,
    tools=[web_search],
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt,
)
# print(web_search.invoke("tomato recipies"))
# print(agent.invoke({"messages": ["tomatoes,potatoes"]}, config))

while True:
    user_query = input("User:")
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config,
    )
    print(response["messages"][-1].content[0]["text"])
