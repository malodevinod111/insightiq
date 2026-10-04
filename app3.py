import html
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer2 import validate_data, generate_summary
from chat_engine import respond, route_message, INTENT_CHITCHAT
from db_connector import test_connection, get_all_views, get_all_tables, load_table
from llm import transcribe_audio, text_to_speech

MAX_SPEECH_CHARS = 800  # keep TTS calls short/fast; replies are already meant to be concise

BASE_DIR = Path(__file__).parent

# ── Constants ─────────────────────────────────────────────────────
APP_TITLE = "InsightIQ"
FOOTER_TEXT = "Genpact © 2026 | Confidential"
HEADER_SUBTITLE = "Powered by Groq AI &nbsp;|&nbsp; Developed by Genpact &nbsp;|&nbsp; POC v1.0"
SOURCE_FILE = "📂 Excel / CSV"
SOURCE_SQL = "🗄️ SQL Server"

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Helpers ───────────────────────────────────────────────────────
def load_css(path: Path) -> None:
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def metric_card(value, label: str, trusted_label: bool = False) -> None:
    """Render a card. Dynamic text is HTML-escaped unless the label is our own static markup."""
    label_html = label if trusted_label else html.escape(label)
    st.markdown(
        f'<div class="metric-card"><h3>{html.escape(str(value))}</h3><p>{label_html}</p></div>',
        unsafe_allow_html=True,
    )


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray whitespace from column headers (e.g. source files with ' Sales ' instead of
    'Sales'). Left as-is, these cause silent KeyErrors: the AI writes df['Sales'] — the name anyone
    would expect — and the exact-whitespace mismatch fails without an obvious reason."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def read_upload(name: str, data: bytes) -> pd.DataFrame:
    """Parse an uploaded file once per unique content, not on every rerun."""
    buffer = BytesIO(data)
    if name.lower().endswith(".csv"):
        df = pd.read_csv(buffer)
    else:
        df = pd.read_excel(buffer)
    return _clean_columns(df)


@st.cache_data(show_spinner=False)
def read_sql_table(name: str) -> pd.DataFrame:
    return _clean_columns(load_table(name))


def set_dataset(df: pd.DataFrame, name: str, is_sql: bool) -> None:
    """Make df the active dataset and reset everything derived from the previous one."""
    st.session_state["df"] = df
    st.session_state["source_name"] = name
    st.session_state["is_sql_source"] = is_sql
    st.session_state["results"] = {}
    st.session_state["chat"] = []
    st.session_state["stats"] = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "dupes": int(df.duplicated().sum()),
    }


def clear_dataset() -> None:
    for key in ("df", "source_name", "is_sql_source", "results", "chat", "stats", "file_key"):
        st.session_state.pop(key, None)


def store_result(key: str, value) -> None:
    st.session_state.setdefault("results", {})[key] = value


def get_result(key: str):
    return st.session_state.get("results", {}).get(key)


def show_error(prefix: str, e: Exception) -> None:
    st.error(f"❌ {prefix}: {e}")


def show_chart(chart: dict) -> None:
    if chart["kind"] == "line":
        st.line_chart(chart["data"])
    else:
        st.bar_chart(chart["data"], sort=False)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart"):
            show_chart(msg["chart"])
        if msg.get("table") is not None:
            with st.expander(f"Result table ({len(msg['table']):,} rows)", expanded=not msg.get("chart")):
                st.dataframe(msg["table"], use_container_width=True)
        if msg.get("code"):
            with st.expander("Generated SQL" if msg["code_lang"] == "sql" else "Generated code", expanded=False):
                st.code(msg["code"], language=msg["code_lang"])


def speak(text: str) -> tuple[bytes, str] | None:
    """Text-to-speech for a reply. Returns None if there's nothing to say; surfaces (not hides)
    any failure, since silently swallowing it just looks like "voice does nothing" to the user."""
    if not text.strip():
        return None
    try:
        return text_to_speech(text[:MAX_SPEECH_CHARS])
    except Exception as e:
        st.toast(f"🔊 Voice reply failed: {e}", icon="⚠️")
        return None


def chat_transcript(messages: list) -> str:
    lines = []
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
        if m.get("code"):
            lines.append(f"[{m['code_lang'].upper()}]\n{m['code']}")
        lines.append("")
    return "\n".join(lines)


def render_chat(df, source_name, is_sql: bool) -> None:
    """Chat with memory: every turn is kept in session_state and earlier turns are sent to the AI."""
    messages = st.session_state.setdefault("chat", [])

    # Audio for the most recent reply, queued by the previous run; played once, then dropped.
    pending_audio = st.session_state.pop("voice_audio", None)
    for i, msg in enumerate(messages):
        render_message(msg)
        if pending_audio and pending_audio[0] == i:
            audio_bytes, audio_mime = pending_audio[1]
            st.audio(audio_bytes, format=audio_mime, autoplay=True)

    # Native mic icon inside the chat box itself (Streamlit >= 1.41's built-in audio recorder).
    submission = st.chat_input(
        "Ask a question, e.g. \"top 5 products by sales\" — or tap 🎤 to speak",
        accept_audio=True,
    )

    question = None
    spoken_input = False  # True when this turn's question came from the mic, not typing
    if submission:
        if isinstance(submission, str):
            question = submission.strip()
        else:
            typed = (submission.text or "").strip()
            if typed:
                question = typed
            elif submission.audio is not None:
                with st.spinner("🎧 Transcribing..."):
                    try:
                        question = transcribe_audio(submission.audio.read(), filename="audio.wav")
                        spoken_input = bool(question)
                    except Exception as e:
                        st.toast(f"🎤 Transcription failed: {e}", icon="⚠️")
                if not question:
                    st.toast("🎤 Didn't catch any speech — try again.")

    if df is None:
        if question and question.strip():
            question = question.strip()
            messages.append({"role": "user", "content": question})
            route = route_message(question, messages[:-1], [])
            if route["intent"] == INTENT_CHITCHAT:
                # Small talk doesn't need data loaded — answer right away.
                messages.append({"role": "assistant",
                                  "content": route.get("reply") or "Hi! How can I help you today?"})
            else:
                # A real data question: park it, ask the user to pick a source, answer it once loaded
                st.session_state["pending_question"] = question
                messages.append({"role": "assistant", "content": "I don't have any data loaded yet. "
                                 "Upload a file or select a table from the sidebar, and I'll answer your "
                                 "question right away."})
            st.rerun()
        return

    pending = st.session_state.pop("pending_question", None)
    if pending:
        question = pending
    if question and question.strip():
        question = question.strip()
        history = list(messages)
        messages.append({"role": "user", "content": question})
        render_message(messages[-1])
        try:
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    reply = respond(df, source_name, is_sql, question, history)
            messages.append(reply)
            if spoken_input or st.session_state.get("voice_on"):
                with st.spinner("🔊 Preparing voice..."):
                    spoken = speak(reply["content"])
                if spoken:
                    st.session_state["voice_audio"] = (len(messages) - 1, spoken)
            st.rerun()
        except Exception as e:
            messages.pop()  # drop the unanswered question so it doesn't pollute the history
            show_error("AI request failed", e)

    if messages:
        col_dl, col_clear = st.columns(2)
        with col_dl:
            st.download_button("📥 Download Chat", chat_transcript(messages), file_name="kestra_chat.txt",
                               mime="text/plain", key="chat_dl")
        with col_clear:
            if st.button("🗑️ Clear Chat", key="chat_clear"):
                st.session_state["chat"] = []
                st.rerun()


# ── Styling & Header ──────────────────────────────────────────────
load_css(BASE_DIR / "style.css")

st.markdown(f"""
<div class="header-box">
    <h1>📊 InsightIQ</h1>
    <p>{HEADER_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Source")

    source = st.radio("Choose source:", [SOURCE_FILE, SOURCE_SQL])

    if source == SOURCE_FILE:
        uploaded_file = st.file_uploader(
            "Upload Excel or CSV file",
            type=["xlsx", "xls", "csv"]
        )
        if uploaded_file is not None:
            file_key = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("file_key") != file_key:
                try:
                    df_loaded = read_upload(uploaded_file.name, uploaded_file.getvalue())
                except Exception as e:
                    show_error("Error loading file", e)
                    st.stop()
                set_dataset(df_loaded, uploaded_file.name, is_sql=False)
                st.session_state["file_key"] = file_key
        elif "file_key" in st.session_state:
            # The uploaded file was removed — drop its stale data
            clear_dataset()

    else:
        st.markdown("#### SQL Server Connection")
        if st.button("Test Connection"):
            try:
                if test_connection():
                    st.success("Connected")
                else:
                    st.error("Failed - Check .env")
            except Exception as e:
                show_error("Connection error", e)

        data_type = st.radio("Load:", ["Views", "Tables", "Both"])
        if st.button("📋 Load List"):
            try:
                items = []
                if data_type in ("Views", "Both"):
                    items += get_all_views()
                if data_type in ("Tables", "Both"):
                    items += get_all_tables()
                st.session_state["db_items"] = items
                st.success(f"Found {len(items)} items")
            except Exception as e:
                show_error("Error", e)

        if "db_items" in st.session_state:
            selected_name = st.selectbox("Select:", st.session_state["db_items"])
            if st.button("📥 Load Data"):
                with st.spinner(f"Loading {selected_name}..."):
                    try:
                        loaded_df = read_sql_table(selected_name)
                    except Exception as e:
                        show_error(f"Could not load {selected_name}", e)
                    else:
                        set_dataset(loaded_df, selected_name, is_sql=True)
                        st.session_state.pop("file_key", None)
                        st.success(f"✅ {len(loaded_df):,} rows loaded")

    st.markdown("---")
    st.markdown("### 🎙️ Voice")
    st.caption("Tap 🎤 in the chat box to ask by voice — replies to voice questions are always spoken back.")
    st.checkbox("Also speak replies to typed questions", key="voice_on")

    st.markdown("---")
    st.caption(FOOTER_TEXT)

# ── Main Content ──────────────────────────────────────────────────
df = st.session_state.get("df")

if df is None:
    st.markdown("### 👈 Upload a file or load data from SQL Server to get started")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("💬", "<b>Ask AI</b> — Type any question about your data in plain English", trusted_label=True)
    with col2:
        metric_card("🔍", "<b>Validate</b> — Auto-detect missing values, duplicates, anomalies", trusted_label=True)
    with col3:
        metric_card("📋", "<b>Summary</b> — Generate executive summary with real numbers", trusted_label=True)

    st.markdown("---")
    render_chat(None, None, False)

else:
    source_name = st.session_state.get("source_name", "Uploaded file")
    is_sql = st.session_state.get("is_sql_source", False)
    stats = st.session_state["stats"]
    st.caption(f"📌 Currently analyzing: **{source_name}**")

    # ── Dataset Metrics ───────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        metric_card(f"{stats['rows']:,}", "Total Rows")
    with cols[1]:
        metric_card(stats["cols"], "Total Columns")
    with cols[2]:
        metric_card(f"{stats['missing']:,}", "Missing Values")
    with cols[3]:
        metric_card(f"{stats['dupes']:,}", "Duplicate Rows")

    # ── Data Preview ──────────────────────────────────────────────
    with st.expander("📊 Preview Data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(df):,} rows")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "💬 Chat",
        "🔍 Validate Data",
        "📋 Executive Summary"
    ])

    # ── TAB 1 — Chat ──────────────────────────────────────────────
    with tab1:
        render_chat(df, source_name, is_sql)

    # ── TAB 2 — Validate ──────────────────────────────────────────
    with tab2:
        st.markdown("#### Automated Data Quality Report")
        st.markdown("Detects missing values, duplicates, negative values and generates AI recommendations.")

        col1, col2 = st.columns(2)
        with col1:
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["Missing %"] = (missing_df["Missing Count"] / len(df) * 100).round(1)
            missing_df = missing_df[missing_df["Missing Count"] > 0]

            if len(missing_df) > 0:
                st.markdown("**Missing Values:**")
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found")

        with col2:
            neg_data = [
                {"Column": col, "Negative Count": int((df[col] < 0).sum())}
                for col in df.select_dtypes(include="number").columns
                if (df[col] < 0).any()
            ]
            if neg_data:
                st.markdown("**Negative Values:**")
                st.dataframe(pd.DataFrame(neg_data), use_container_width=True)
            else:
                st.success("✅ No negative values found")

        if st.button("🔍 Run Full AI Validation Report", key="validate_btn"):
            try:
                with st.spinner("Running validation..."):
                    store_result("validate", validate_data(df))
            except Exception as e:
                show_error("Validation failed", e)

        report = get_result("validate")
        if report:
            with st.container(border=True):
                st.markdown("**🔍 AI Validation Report:**")
                st.markdown(report)

            st.download_button(
                label="📥 Download Validation Report",
                data=report,
                file_name="kestra_validation_report.txt",
                mime="text/plain"
            )

    # ── TAB 3 — Summary ───────────────────────────────────────────
    with tab3:
        st.markdown("#### Executive Summary")
        st.markdown("Generates a management-ready summary with real numbers from your data.")

        st.markdown("**Dataset Statistics:**")
        st.dataframe(df.describe(include="all").T.astype(str), use_container_width=True)

        if st.button("📋 Generate Executive Summary", key="summary_btn"):
            try:
                with st.spinner("Generating summary..."):
                    store_result("summary", generate_summary(df))
            except Exception as e:
                show_error("Summary failed", e)

        summary = get_result("summary")
        if summary:
            with st.container(border=True):
                st.markdown("**📋 Executive Summary:**")
                st.markdown(summary)

            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name="kestra_executive_summary.txt",
                mime="text/plain"
            )
