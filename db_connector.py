import os
import re
from contextlib import contextmanager
from decimal import Decimal

import pandas as pd
import pyodbc
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

MAX_ROWS = 10000
QUERY_TIMEOUT_SECONDS = 30
CONNECT_TIMEOUT_SECONDS = 30


# Connect to the database using environment variables
def get_connection():
    server = os.environ.get("DB_SERVER")
    database = os.environ.get("DB_NAME")
    username = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    driver = os.environ.get("DB_DRIVER", "ODBC Driver 17 for SQL Server")

    if not server or not database:
        raise ValueError("DB_SERVER and DB_NAME must be set in the .env file.")

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        "Encrypt=yes",
        "TrustServerCertificate=yes",
    ]
    if username and password:
        parts += [f"UID={username}", f"PWD={{{password}}}"]
    else:
        parts.append("Trusted_Connection=yes")

    # Never log the connection string — it contains the password
    return pyodbc.connect(";".join(parts) + ";", timeout=CONNECT_TIMEOUT_SECONDS)


@contextmanager
def connection():
    """Open a connection and always close it, even if the query fails."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def test_connection() -> bool:
    try:
        with connection():
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


def _query_df(query: str, params=None) -> pd.DataFrame:
    with connection() as conn:
        return pd.read_sql(query, conn, params=params)


def quote_identifier(name: str) -> str:
    """Bracket-quote a SQL Server identifier, escaping any closing bracket."""
    return "[" + name.replace("]", "]]") + "]"


# get all views in the database
def get_all_views() -> list:
    df = _query_df("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS ORDER BY TABLE_NAME")
    return df["TABLE_NAME"].tolist()


# get all tables in the database
def get_all_tables() -> list:
    df = _query_df("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
    """)
    return df["TABLE_NAME"].tolist()


# Load view or table into a pandas DataFrame
def load_table(name: str, limit: int = MAX_ROWS) -> pd.DataFrame:
    """Load up to `limit` rows (default MAX_ROWS). Pass limit=None to load everything."""
    top = f"TOP {int(limit)} " if limit is not None else ""
    return _query_df(f"SELECT {top}* FROM {quote_identifier(name)}")


# run custom sql query and return results as a pandas DataFrame
# NOTE: only for trusted, hard-coded queries. AI-generated SQL must use run_safe_query.
def run_query(query: str) -> pd.DataFrame:
    return _query_df(query)


_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|merge|exec|execute|"
    r"grant|revoke|deny|into|waitfor|shutdown|backup|restore|openrowset|"
    r"opendatasource|openquery|bulk|dbcc|kill|reconfigure)\b|\bxp_|\bsp_",
    re.IGNORECASE,
)


def validate_select_only(query: str) -> str:
    """Return a cleaned single SELECT statement, or raise ValueError if it is unsafe."""
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("Empty query.")
    if "--" in cleaned or "/*" in cleaned:
        raise ValueError("SQL comments are not allowed.")
    # Ignore string literals when checking for keywords
    without_literals = re.sub(r"'(?:[^']|'')*'", "''", cleaned)
    if ";" in without_literals:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"(?i)^\s*(select|with)\b", without_literals):
        raise ValueError("Only SELECT queries are allowed.")
    match = _FORBIDDEN_KEYWORDS.search(without_literals)
    if match:
        raise ValueError(f"Query contains a forbidden keyword: '{match.group(0)}'.")
    return cleaned


# run an AI-generated query safely: SELECT only, row-capped, with a timeout
def run_safe_query(query: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    safe_query = validate_select_only(query)
    with connection() as conn:
        conn.timeout = QUERY_TIMEOUT_SECONDS
        cursor = conn.cursor()
        cursor.execute(safe_query)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(max_rows)
        # pyodbc returns DECIMAL/NUMERIC/MONEY columns as Decimal; DataFrame.from_records
        # then keeps them as dtype=object, which select_dtypes(include="number") misses
        # (e.g. in charts.build_chart). Convert to float so they're treated as numeric.
        records = [
            tuple(float(v) if isinstance(v, Decimal) else v for v in r)
            for r in rows
        ]
        return pd.DataFrame.from_records(records, columns=columns)


def get_schema(name: str) -> pd.DataFrame:
    return _query_df("""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
    """, params=[name])
