"""Chat layer: decides what each message is (small talk, data question, validate, summary) and answers it."""
import pandas as pd

from analyzer2 import ask_ai, generate_summary, validate_data
from charts import build_chart
from db_analyzer import ask_db
from llm import chat, parse_json_object

MAX_HISTORY_MESSAGES = 8
MAX_COLUMNS_IN_ROUTER = 60

INTENT_CHITCHAT = "chitchat"
INTENT_QUESTION = "data_question"
INTENT_VALIDATE = "validate"
INTENT_SUMMARY = "summary"
_INTENTS = {INTENT_CHITCHAT, INTENT_QUESTION, INTENT_VALIDATE, INTENT_SUMMARY}

ROUTER_SYSTEM_PROMPT = """You are the assistant inside "InsightIQ", a chat tool that answers
questions about a dataset the user has loaded. Read the user's latest message (and the conversation so far) and
reply with ONE JSON object and nothing else:

{"intent": "...", "standalone_question": "...", "reply": "..."}

intent must be exactly one of:
- "chitchat": greetings, thanks, small talk, or questions about what you can do. Put a short, friendly reply in
  "reply". You can mention that you answer questions about the loaded data, run a data-quality check, and write an
  executive summary. Reply in the user's language.
- "data_question": any question that needs the dataset (totals, filters, top N, trends, comparisons, ...).
- "validate": the user wants a data-quality / validation report.
- "summary": the user wants an executive summary of the dataset.

For "data_question", set "standalone_question" to the user's question rewritten so it makes sense WITHOUT the
conversation (resolve words like "it", "that", "those", "same for last year" using earlier messages). Keep the
user's meaning; do not invent columns or numbers. For other intents use "" for "standalone_question".
For non-chitchat intents use "" for "reply"."""


def _recent(history: list) -> list:
    return [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY_MESSAGES:]]


def route_message(question: str, history: list, columns: list) -> dict:
    """Classify the message. Falls back to a plain data question if the model's reply can't be parsed."""
    if columns:
        shown = ", ".join(map(str, columns[:MAX_COLUMNS_IN_ROUTER]))
        context = f"A dataset is loaded. Its columns: {shown}"
    else:
        context = "No dataset is loaded yet."

    parsed = parse_json_object(chat(
        ROUTER_SYSTEM_PROMPT,
        f"{context}\n\nLatest user message: {question}",
        max_tokens=400,
        history=_recent(history),
    ))
    if not parsed or parsed.get("intent") not in _INTENTS:
        return {"intent": INTENT_QUESTION, "standalone_question": question, "reply": ""}
    if parsed["intent"] == INTENT_QUESTION and not str(parsed.get("standalone_question", "")).strip():
        parsed["standalone_question"] = question
    return parsed


def _message(content: str, code: str | None = None, code_lang: str | None = None,
             table: pd.DataFrame | None = None) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "code": code,
        "code_lang": code_lang,
        "table": table,
        "chart": build_chart(table),
    }


def respond(df: pd.DataFrame | None, source_name: str | None, is_sql: bool, question: str, history: list) -> dict:
    """Answer one chat message. `history` holds earlier {"role", "content"} turns (not including `question`)."""
    columns = list(df.columns) if df is not None else []
    route = route_message(question, history, columns)
    intent = route["intent"]

    if intent == INTENT_CHITCHAT:
        return _message(route.get("reply") or "Hi! How can I help you with your data today?")

    if df is None:
        return _message("Please upload a file or load a table from SQL Server first, then I can help with that.")

    if intent == INTENT_VALIDATE:
        return _message(validate_data(df))
    if intent == INTENT_SUMMARY:
        return _message(generate_summary(df))

    standalone = str(route["standalone_question"]).strip()
    if is_sql:
        sql_query, table, answer = ask_db(source_name, standalone)
        return _message(answer, sql_query, "sql", table)
    code, table, answer = ask_ai(df, standalone)
    return _message(answer, code, "python", table)
