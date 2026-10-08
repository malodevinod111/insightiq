import ast
import sys

import pandas as pd

from llm import CodeRunError, chat, explain_result, generate_and_run

# ─────────────────────────────────────────────
#  Safety layer for AI-generated pandas code
#  NOTE: this is defence in depth, not a true sandbox. For untrusted
#  users, run the code in a separate locked-down process/container.
# ─────────────────────────────────────────────
_ALLOWED_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "len", "sum", "min", "max", "abs", "round", "sorted", "range", "int", "float",
        "str", "bool", "list", "dict", "set", "tuple", "enumerate", "zip", "any", "all",
        "reversed", "isinstance",
    )
}
_BANNED_CALLS = {
    "eval", "exec", "open", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "dir", "type",
}
_BANNED_ATTRS = {
    "eval", "to_csv", "to_excel", "to_pickle", "to_sql", "to_parquet", "to_json",
    "to_feather", "to_hdf", "to_clipboard", "to_xml", "to_html", "to_latex", "to_markdown",
}


def _validate_pandas_code(code: str) -> None:
    """Raise ValueError if the code does anything beyond analysing `df`."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Generated code is not valid Python: {e}")

    assigns_result = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise ValueError("Imports are not allowed.")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr.startswith("read_") or node.attr in _BANNED_ATTRS
        ):
            raise ValueError(f"Attribute '{node.attr}' is not allowed.")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ValueError(f"Name '{node.id}' is not allowed.")
            if isinstance(node.ctx, ast.Store) and node.id == "result":
                assigns_result = True
            if node.id in _BANNED_CALLS:
                raise ValueError(f"'{node.id}' is not allowed.")
    if not assigns_result:
        raise ValueError("Generated code does not assign a 'result' variable.")


def _run_pandas_code(df: pd.DataFrame, code: str):
    _validate_pandas_code(code)
    scope = {"__builtins__": _ALLOWED_BUILTINS, "df": df.copy(), "pd": pd}
    exec(code, scope)
    return scope.get("result", "No result found")


# ─────────────────────────────────────────────
#  Shared data-quality checks
# ─────────────────────────────────────────────
def find_issues(df: pd.DataFrame) -> list:
    """Missing values, duplicates and negative numbers, as readable lines."""
    issues = []
    total = len(df)
    if total == 0:
        return ["Dataset has no rows"]

    for col, count in df.isnull().sum().items():
        if count > 0:
            issues.append(f"Column '{col}' has {count} missing values ({round(count / total * 100, 1)}%)")

    dupes = int(df.duplicated().sum())
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate rows")

    for col in df.select_dtypes(include="number").columns:
        neg = int((df[col] < 0).sum())
        if neg > 0:
            issues.append(f"Column '{col}' has {neg} negative values")
    return issues


# ─────────────────────────────────────────────
#  1. ASK AI — Pandas code execution approach
# ─────────────────────────────────────────────
ASK_SYSTEM_PROMPT = """You are a Python Pandas expert.
Given a dataframe schema and a question, write ONLY executable Python/Pandas code.
Rules:
- Dataframe variable is always called 'df'
- Last line MUST store result in variable called 'result'
- No imports, no file or network access
- No explanation, no markdown, no backticks, no comments
- Just raw executable Python code
Examples:
Q: show me sales id 3257
A: result = df[df['sales_id'] == 3257]

Q: how many rows have missing values
A: result = df.isnull().sum()

Q: total net amount
A: result = df['net_amount'].sum()

Q: average unit price
A: result = df['unit_price'].mean()

Q: how many unique customers
A: result = df['customer_sk'].nunique()"""

MAX_RESULT_CHARS = 8000


def _to_table(result) -> pd.DataFrame | None:
    """Turn a DataFrame/Series result into a display table; scalars have no table."""
    if isinstance(result, pd.Series):
        result = result.rename(result.name if result.name is not None else "value").reset_index()
    if isinstance(result, pd.DataFrame):
        table = result.copy()
        table.columns = [str(c) for c in table.columns]
        return table
    return None


def ask_ai(df: pd.DataFrame, question: str) -> tuple:
    """Return (pandas_code, result_table_or_None, explanation) for a question about `df`."""
    schema_context = f"""
Columns : {', '.join(map(str, df.columns))}
Dtypes  : {df.dtypes.astype(str).to_dict()}
Rows    : {len(df)}
Sample  :
{df.head(3).to_string(index=False)}
"""

    # Steps 1+2 — AI writes pandas code; validate and run it on a copy of the REAL dataframe.
    # If it fails, the AI sees the error and gets one chance to fix its code.
    try:
        pandas_code, result = generate_and_run(
            ASK_SYSTEM_PROMPT,
            f"Schema:\n{schema_context}\n\nQuestion: {question}\n\nPandas code:",
            run=lambda code: _run_pandas_code(df, code),
            # reasoning_effort="low": the model (gpt-oss) spends hidden "thinking" tokens from the same
            # max_tokens budget before writing the actual code. At the old max_tokens=300 with default
            # reasoning, a non-trivial schema could make it think through the whole budget and return
            # empty content, which failed validation with a misleading "doesn't assign 'result'" error.
            max_tokens=600,
            reasoning_effort="low",
        )
    except CodeRunError as e:
        return e.code, None, f"❌ Could not answer this question: {e.original}"

    result_text = str(result)
    if len(result_text) > MAX_RESULT_CHARS:
        result_text = result_text[:MAX_RESULT_CHARS] + "\n... (result truncated)"

    # Step 3 — AI explains the REAL result
    return pandas_code, _to_table(result), explain_result(question, pandas_code, result_text)


# ─────────────────────────────────────────────
#  2. VALIDATE — Data quality check
# ─────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> str:
    issues_text = "\n".join(find_issues(df)) or "No issues found"

    stats = f"""
Total Rows     : {len(df)}
Total Columns  : {len(df.columns)}
Duplicate Rows : {df.duplicated().sum()}
Missing Values : {df.isnull().sum().sum()} total
"""

    return chat(
        "You are a data quality expert. Use ONLY the exact stats and issues provided. Never make up numbers.",
        f"""
Dataset Stats:
{stats}

Issues Found:
{issues_text}

Provide:
1. Summary of issues
2. Severity (High/Medium/Low) for each
3. Recommended fix for each
4. Overall data quality score out of 10
Use ONLY the numbers provided above.
""",
        max_tokens=2000,  # same truncation risk as generate_summary
        reasoning_effort="low",
    )


# ─────────────────────────────────────────────
#  3. SUMMARY — Executive summary
# ─────────────────────────────────────────────
MAX_CATEGORY_COLUMNS = 3
MAX_CATEGORY_VALUES = 8


def _numeric_stats(df: pd.DataFrame) -> list:
    return [
        f"{col}: min={df[col].min()}, max={df[col].max()}, "
        f"mean={round(df[col].mean(), 2)}, sum={round(df[col].sum(), 2)}"
        for col in df.select_dtypes(include="number").columns
    ]


def _category_stats(df: pd.DataFrame) -> list:
    """Top values for a few low-cardinality text columns (replaces the hard-coded payment_method)."""
    lines = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if 1 < df[col].nunique() <= 20:
            top = df[col].value_counts().head(MAX_CATEGORY_VALUES).to_dict()
            lines.append(f"{col}: {top}")
        if len(lines) >= MAX_CATEGORY_COLUMNS:
            break
    return lines


def generate_summary(df: pd.DataFrame) -> str:
    real_stats = f"""
Total Rows     : {len(df)}
Total Columns  : {len(df.columns)}
Columns        : {', '.join(map(str, df.columns))}
Missing Values : {df.isnull().sum().to_dict()}
Duplicates     : {df.duplicated().sum()}

Numeric Column Stats:
{chr(10).join(_numeric_stats(df)) or 'N/A'}

Top Category Values:
{chr(10).join(_category_stats(df)) or 'N/A'}
"""

    return chat(
        """You are a senior data analyst.
Generate executive summaries using ONLY the exact numbers provided.
Never assume or make up any values.""",
        f"""
Generate an executive summary using ONLY these exact stats:
{real_stats}

Include:
1. Dataset overview
2. Key metrics with exact numbers
3. Data quality observations
4. Important highlights
Keep to 6-8 bullet points. Use only numbers from the stats above.
""",
        # Low reasoning + a larger budget: gpt-oss draws its hidden "thinking" from max_tokens, and at
        # 800 with default reasoning the summary was regularly cut off mid-sentence.
        max_tokens=2000,
        reasoning_effort="low",
    )


# ─────────────────────────────────────────────
#  CONSOLE MENU
# ─────────────────────────────────────────────
def _load_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    if file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    raise ValueError("Only .xlsx, .xls or .csv supported")


def main() -> None:
    print("📊 InsightIQ | By Genpact")
    print("=" * 55)

    file_path = sys.argv[1] if len(sys.argv) >= 2 else input("Enter Excel/CSV file path: ").strip()

    try:
        df = _load_file(file_path)
    except FileNotFoundError:
        sys.exit(f"❌ File not found: {file_path}")
    except Exception as e:
        sys.exit(f"❌ Error: {e}")

    print(f"\n✅ File loaded: {file_path}")
    print(f"   Rows    : {len(df)}")
    print(f"   Columns : {len(df.columns)}")
    print(f"   Columns : {', '.join(map(str, df.columns))}")
    print("=" * 55)

    actions = {
        "2": ("Running validation...", validate_data),
        "3": ("Generating summary...", generate_summary),
    }

    while True:
        print("\nWhat do you want to do?")
        print("  1 → Ask a question")
        print("  2 → Validate data quality")
        print("  3 → Generate executive summary")
        print("  0 → Exit")

        choice = input("\nEnter choice (0/1/2/3): ").strip()

        try:
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                question = input("Your question: ").strip()
                if question:
                    print("\nThinking...\n")
                    _, table, answer = ask_ai(df, question)
                    if table is not None:
                        print(table.to_string(index=False))
                    print(answer)
                    print("-" * 55)
            elif choice in actions:
                message, func = actions[choice]
                print(f"\n{message}\n")
                print(func(df))
                print("-" * 55)
            else:
                print("❌ Invalid choice. Enter 0, 1, 2 or 3")
        except Exception as e:
            print(f"❌ {e}")


if __name__ == "__main__":
    main()
