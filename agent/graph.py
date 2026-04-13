from langgraph.graph import END, StateGraph

from .nodes import (
    astream_respond,
    respond,
    retrieve,
    route,
    should_regenerate_sql,
    should_skip_sql,
    sql_gen,
    validate_sql,
)
from .state import AgentState


def build_graph():
    """Graph for synchronous invoke."""
    workflow = StateGraph(AgentState)

    workflow.add_node("route", route)
    workflow.add_node("sql_gen", sql_gen)
    workflow.add_node("validate_sql", validate_sql)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("respond", respond)

    workflow.set_entry_point("route")

    workflow.add_conditional_edges(
        "route",
        should_skip_sql,
        {
            "sql_gen": "sql_gen",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("sql_gen", "validate_sql")

    workflow.add_conditional_edges(
        "validate_sql",
        should_regenerate_sql,
        {
            "sql_gen": "sql_gen",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


def build_streaming_graph():
    """Graph for streaming invoke - uses async respond node."""
    workflow = StateGraph(AgentState)

    workflow.add_node("route", route)
    workflow.add_node("sql_gen", sql_gen)
    workflow.add_node("validate_sql", validate_sql)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("respond", astream_respond)

    workflow.set_entry_point("route")

    workflow.add_conditional_edges(
        "route",
        should_skip_sql,
        {
            "sql_gen": "sql_gen",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("sql_gen", "validate_sql")

    workflow.add_conditional_edges(
        "validate_sql",
        should_regenerate_sql,
        {
            "sql_gen": "sql_gen",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


graph = build_graph()
streaming_graph = build_streaming_graph()
