import os

from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities import SQLDatabase


def get_sql_tools(connection_string: str | None = None) -> list:
    """
    Return a set of SQL tools for a given connection string.
    Falls back to SQL_CONNECTION_STRING env var if not provided.

    Tools returned:
      - QuerySQLDataBaseTool  : execute a SQL SELECT and return results
      - InfoSQLDatabaseTool   : describe table schema
      - ListSQLDatabaseTool   : list all table names
    """
    conn_str = connection_string or os.getenv("SQL_CONNECTION_STRING", "")
    if not conn_str:
        return []

    db = SQLDatabase.from_uri(conn_str)
    return [
        QuerySQLDataBaseTool(db=db),
        InfoSQLDatabaseTool(db=db),
        ListSQLDatabaseTool(db=db),
    ]
