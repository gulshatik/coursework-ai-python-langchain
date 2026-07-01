#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Простой AI‑агент на LangChain + LangGraph, который:
1) Принимает список продуктов и город.
2) Для каждого продукта запрашивает цену через инструмент get_price.
3) Считает итоговую стоимость корзины и выводит таблицу.

Требования к окружению:
- Python 3.10+
- LM Studio (локальный сервер LLM) доступен по адресу https://llm.brojs.ru/v1
"""

import os
from dotenv import load_dotenv

# Загружаем переменные окружения, если они есть
load_dotenv()

# ------------------------------------------------------------------
# Подключаемся к локальной модели через OpenAI‑совместимый API
# ------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="gpt-4o-mini",          # <-- замените на название вашей модели в LM Studio
    base_url="https://llm.brojs.ru/v1",
    api_key=SecretStr(os.getenv("BROJS_PAT_TOKEN") or "fake"),
    temperature=0.7,
)

# ------------------------------------------------------------------
# Определяем инструмент get_price с субагентом (простой вызов LLM)
# ------------------------------------------------------------------
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

@tool
def get_price(product: str, city: str) -> str:
    """
    Получить примерную цену продукта в указанном городе.
    Возвращает строку‑таблицу вида:

    | Продукт | Цена (руб.) | Магазин |
    """
    # Формируем запрос к модели
    prompt = (
        f"Сгенерируй таблицу с ценой для продукта '{product}' "
        f"в городе '{city}'. Таблица должна содержать три колонки: "
        "Продукт, Цена (руб.), Магазин. Используй реалистичные цены и магазины."
    )
    # Вызов LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# ------------------------------------------------------------------
# Создаём главный агент с инструментом get_price
# ------------------------------------------------------------------
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: State) -> dict:
    """
    Вызов основной модели для генерации ответа или вызова инструмента.
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Создаём граф
builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools=[get_price]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# ------------------------------------------------------------------
# Запускаем агента с примерным запросом
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Примерный вопрос пользователя
    user_question = (
        "Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."
    )

    # Запускаем граф
    config = {"configurable": {"thread_id": "shopping_agent"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_question)]},
        config=config,
    )

    # Функция форматирования сообщений для вывода
    def format_message(message: BaseMessage) -> str:
        if isinstance(message, AIMessage):
            return message.content or ""
        # Если сообщение содержит вызов инструмента
        if hasattr(message, "tool_calls") and message.tool_calls:
            call = message.tool_calls[0]
            name = call["name"]
            args = call.get("args", {})
            return f"{name}({args})"
        return ""

    # Выводим все сообщения (включая промежуточные вызовы)
    print("\n--- Цепочка сообщений ---\n")
    for msg in result["messages"]:
        formatted = format_message(msg).replace("\u202f", " ")
        if formatted:
            print(formatted)

    # Для удобства выводим итоговую таблицу и сумму
    import re

    def extract_price(table_text: str) -> int | None:
        """
        Извлекает первую числовую цену из строки таблицы.
        Возвращает None, если цена не найдена.
        """
        match = re.search(r"(\d+)", table_text)
        return int(match.group(1)) if match else None

    # Ищем все таблицы в ответе
    tables = re.findall(r"\|.*?\|\n\|.*?\|\n(?:\|.*?\|\n)+", result["messages"][-1].content or "")
    total = 0
    for tbl in tables:
        price = extract_price(tbl)
        if price is not None:
            total += price

    if total > 0:
        print(f"\n**Итого:** ~{total} руб.")
