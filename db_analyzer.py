from db_connector import get_schema, quote_identifier, run_safe_query
from llm import CodeRunError, explain_result, generate_and_run

MAX_RESULT_ROWS_FOR_EXPLANATION = 50

SQL_SYSTEM_PROMPT = """You are a SQL expert.
Given a table/view schema and a question, write only a valid sql query.
Rules:
- Use correct sql server syntax
- Table name must wrap in square brackets []
- No explanation, no markdown, no backticks, no comments
- Just raw sql query
- use Top insted of Limit
- use GETDATE() for current date
- "Revenue" or "sales" means the [Sales] column (net sales after discounts)
- Use [Gross_Sales] only if the question says "gross"
- If no column matches the question, reply exactly: NO_MATCHING_COLUMN: <closest column names>

Examples:
Q: show me Top 10 records
A: SELECT TOP 10 * FROM [table_name]
"""

NO_MATCH_PREFIX = "NO_MATCHING_COLUMN:"


def _run_sql(code: str):
    """Run generated SQL; the model's 'no matching column' reply passes through untouched."""
    if code.startswith(NO_MATCH_PREFIX):
        return code
    return run_safe_query(code)


# Ask AI to write SQL, run it safely, then explain the result
def ask_db(table_name: str, question: str) -> tuple:
    """Return (sql_query, result_df, explanation) for a given table and question."""
    schema_str = get_schema(table_name).to_string(index=False)

    # Steps 1+2 — AI writes the SQL; we run it (SELECT only, row-capped, with timeout).
    # If it fails, the AI sees the error and gets one chance to fix its query.
    try:
        sql_query, result_df = generate_and_run(
            SQL_SYSTEM_PROMPT,
            f"""
Table/view name: {quote_identifier(table_name)}
Schema:
{schema_str}
Question: {question}
Write a valid SQL query:""",
            run=_run_sql,
            # "low" reasoning: this is a short, mechanical query-writing task, and keeping the hidden
            # "thinking" tokens small leaves more of max_tokens free for the actual SQL (see analyzer2.ask_ai
            # for the failure mode this avoids: a big schema can make the model think through the whole
            # budget and return empty content).
            reasoning_effort="low",
        )
    except CodeRunError as e:
        if isinstance(e.original, ValueError):
            return e.code, None, f"Query blocked for safety: {e.original}"
        return e.code, None, f"Query execution failed: {e.original}"

    # The model said no column matches — tell the user instead of showing a table
    if isinstance(result_df, str):
        available = result_df.split(NO_MATCH_PREFIX, 1)[1].strip()
        message = (
            f"'{question}' doesn't match any column in [{table_name}]. "
            f"Closest available columns: {available}. "
            f"Try rephrasing using one of these."
        )
        return sql_query, None, message

    # Step 3 — AI explains the result (only the first rows, to keep the prompt small)
    shown = result_df.head(MAX_RESULT_ROWS_FOR_EXPLANATION)
    note = (
        f"\n(Showing first {len(shown)} of {len(result_df)} rows)"
        if len(result_df) > len(shown) else ""
    )
    result_text = shown.to_string(index=False) if not shown.empty else "NO data available"
    return sql_query, result_df, explain_result(question, sql_query, result_text, note)


# Console interface for testing
def main() -> None:
    import sys
    from db_connector import test_connection, get_all_tables, get_all_views

    print("InsightIQ — AI Database Analyzer | By Genpact")
    print("=" * 100)

    if not test_connection():
        sys.exit("Database connection failed. Please check your .env settings.")
    print("Database connection successful!")

    views = get_all_views()
    tables = get_all_tables()
    print(f"\nFound {len(views)} views and {len(tables)} tables")
    print("\nViews:")
    for i, v in enumerate(views, 1):
        print(f"  {i}. {v}")
    print("\nTables:")
    for i, t in enumerate(tables, 1):
        print(f"  {i}. {t}")

    name = input("\nEnter the name of the table/view to analyze: ").strip()
    print(f"\nLoaded: {name}")

    while True:
        question = input("\nYour question (blank to exit): ").strip()
        if not question:
            print("Goodbye!")
            break
        try:
            print("\nThinking...\n")
            sql_query, result_df, explanation = ask_db(name, question)
            print(f"SQL Query:\n{sql_query}")
            if result_df is not None:
                print(f"\nResult:\n{result_df.to_string(index=False)}")
            print(f"\nExplanation:\n{explanation}")
            print("-" * 100)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
