from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    user_input: str
    needs_sql: bool
    sql_filter: Optional[str]
    sql_thoughts: str
    sql_validation: bool
    sql_validation_msg: str
    sql_retry_count: int
    courses: Optional[List]
    retrieval_thoughts: str
    final_response: str
    messages: List
