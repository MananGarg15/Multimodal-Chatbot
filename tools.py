"""
Replaces Day4LLMCalling/tool_list.py + tool_calling.py.

Same two operations (get_ticket_price, set_ticket_price) and the same SQLite
schema, but exposed as @tool-decorated functions so langgraph's ToolNode (or
our custom call_tools node in graph.py) can bind and dispatch them directly,
instead of the hand-written price_function JSON schema + function_map +
handle_tool_call() dispatcher.

Fix vs. the original: the DB path was hardcoded to a Windows user path
(C:\\Users\\PGCP-AI\\projects\\...). It's now relative to this file, so the
project runs on any machine.
"""

import os
import sqlite3 as sql
from typing import Literal

from langchain_core.tools import tool

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tables.db")

_STATIC_PRICES = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


@tool
def get_ticket_price(destination_city: str) -> str:
    """Look up the known flight ticket price for a destination city from the
    static price list."""
    price = _STATIC_PRICES.get(destination_city.lower(), "Price is not known")
    return f"The ticket price for {destination_city} is {price}"


@tool
def set_ticket_price(
    destination_city: str = "",
    price: int = 0,
    action: Literal[
        "set", "get_by_name", "get_by_price", "filter_by_price_min", "filter_by_price_max"
    ] = "set",
) -> str:
    """Manage a database of flight ticket prices. 'set' saves/updates a price
    for a city; 'get_by_name' finds one city's price; 'get_by_price' finds
    cities at an exact price; 'filter_by_price_min'/'filter_by_price_max'
    finds cities at or above/below a price."""
    conn = sql.connect(_DB_PATH)
    conn.row_factory = sql.Row
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS cityPrices (city TEXT PRIMARY KEY, price INTEGER)"
    )

    destination_city = destination_city.lower()

    try:
        match action:
            case "set":
                cursor.execute(
                    "INSERT OR REPLACE INTO cityPrices VALUES(?,?)", (destination_city, price)
                )
                result = f"adding {destination_city} price as {price}"
            case "get_by_name":
                cursor.execute("SELECT * FROM cityPrices WHERE city=?", (destination_city,))
                row = cursor.fetchone()
                result = dict(row) if row else "not found"
            case "get_by_price":
                cursor.execute("SELECT * FROM cityPrices WHERE price=?", (price,))
                result = [dict(r) for r in cursor.fetchall()]
            case "filter_by_price_min":
                cursor.execute("SELECT * FROM cityPrices WHERE price>=?", (price,))
                result = [dict(r) for r in cursor.fetchall()]
            case "filter_by_price_max":
                cursor.execute("SELECT * FROM cityPrices WHERE price<=?", (price,))
                result = [dict(r) for r in cursor.fetchall()]
            case _:
                result = "Invalid Query"
        conn.commit()
    finally:
        conn.close()

    return f"The result of your query is {result}"


ticket_tools = [get_ticket_price, set_ticket_price]
