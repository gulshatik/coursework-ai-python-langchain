#!/usr/bin/env python3
"""
Hierarchical shopping assistant using LangChain 1.x and a local LLM.
The script demonstrates:
- A tool `get_price(product, city)` that internally creates a sub‑agent to generate a price table.
- A top‑level agent that uses the tool to answer a user query about a shopping list.
"""

import os
import sys
from dotenv import load_dotenv
from typing import Annotated, TypedDict

# LangChain imports (1.x)
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables (API key for the remote LLM endpoint)
load_dotenv()

# --------------------------------------------------------------------------- #
# 1. Configure the LLM
# --------------------------------------------------------------------------- #

llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://llm.brojs.ru/v1"),
    api_key=os.getenv("BROJS_PAT_TOKEN"),
    model="openai/gpt-oss-20b",
    temperature=0.1,
)

# --------------------------------------------------------------------------- #
# 2. Define the `get_price` tool with an internal sub‑agent
# --------------------------------------------------------------------------- #

@tool
def get_price(product: str, city: str) -> str:
    """
    Получить примерную цену продукта в указанном городе.

    Returns a markdown table with columns:
        | Продукт | Цена (руб.) | Магазин |
    The response contains only the table without any additional text.
    """
    # Sub‑agent that generates the price table
    def sub_agent_call(product: str, city: str) -> str:
        system_prompt = (
            f"Ты эксперт по ценам на продукты в {city}. "
            "Сгенерируй таблицу с колонками: Продукт, Цена (руб.), Магазин. "
            "Ответ должен быть только таблицей без лишних слов."
        )
        # Simple agent that just calls the LLM once
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Какова цена на {product} в {city}?",
                },
            ]
        )
        # The last message is the assistant's reply
        return response.content

    table = sub_agent_call(product, city)
    return table


# --------------------------------------------------------------------------- #
# 3. Build the top‑level agent graph
# --------------------------------------------------------------------------- #

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_llm(state: State) -> dict:
    """Node that calls the LLM with the current conversation."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Create the graph
builder = StateGraph(State)

# Agent node
builder.add_node("agent", call_llm)

# Tool node (contains our single tool)
tools_node = ToolNode(tools=[get_price])
builder.add_node("tools", tools_node)

# Graph edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# End state
builder.add_edge("agent", END)

# Set the entry point to the agent node (not START)
builder.set_entry_point("agent")

# Compile the graph with a memory checkpoint
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --------------------------------------------------------------------------- #
# 4. Run the agent with a predefined user query
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Predefined user question (no TTY interaction)
    user_query = (
        "Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."
    )

    # Configuration for the thread
    config = {"configurable": {"thread_id": "shopping_thread"}}

    # Invoke the graph
    result = graph.invoke(
        {"messages": [{"role": "human", "content": user_query}]},
        config=config,
    )

    # Helper to pretty‑print each message
    def format_message(msg):
        # If the message has plain content, return it
        if hasattr(msg, "content") and msg.content:
            return msg.content
        # Otherwise it's a tool call; format nicely
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            call = msg.tool_calls[0]
            name = call["name"]
            args = call.get("args", {})
            return f"{name}({args})"
        return ""

    sys.stdout.buffer.write(b"\n--- Conversation Trace ---\n\n")
    for m in result["messages"]:
        out = format_message(m)
        sys.stdout.buffer.write((out + "\n---\n").encode("utf-8"))
