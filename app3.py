import base64
import html
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath

import pandas as pd
import streamlit as st

from analyzer2 import validate_data, generate_summary
from chat_engine import respond, route_message, INTENT_CHITCHAT
from db_connector import get_all_views, load_table

BASE_DIR = Path(__file__).parent

# ── Constants ─────────────────────────────────────────────────────
APP_TITLE = "InsightIQ"
# Fallback heights only: style.css stretches the panels to the window and gives the chat the leftover space.
PANEL_HEIGHT = 680
CHAT_HEIGHT = 400
USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/auto_awesome:"

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Helpers ───────────────────────────────────────────────────────
def load_css(path: Path) -> None:
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


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
def read_view(name: str) -> pd.DataFrame:
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
    for key in ("df", "source_name", "is_sql_source", "results", "chat", "stats", "loaded_key", "selected"):
        st.session_state.pop(key, None)


def store_result(key: str, value) -> None:
    st.session_state.setdefault("results", {})[key] = value


def get_result(key: str):
    return st.session_state.get("results", {}).get(key)


def show_error(prefix: str, e: Exception) -> None:
    st.error(f"{prefix}: {e}")


def show_chart(chart: dict) -> None:
    if chart["kind"] == "line":
        st.line_chart(chart["data"], height=240)
    else:
        st.bar_chart(chart["data"], sort=False, height=240)


def render_message(msg: dict) -> None:
    is_user = msg["role"] == "user"
    with st.chat_message(msg["role"], avatar=USER_AVATAR if is_user else AI_AVATAR):
        if not is_user:
            st.markdown('<div class="chat-label">AI response</div>', unsafe_allow_html=True)
        ai_markdown(msg["content"])
        if msg.get("chart"):
            show_chart(msg["chart"])
        if msg.get("table") is not None:
            with st.expander(f"Result table ({len(msg['table']):,} rows)", expanded=not msg.get("chart")):
                st.dataframe(msg["table"], width="stretch")


def chat_transcript(messages: list) -> str:
    lines = []
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
        lines.append("")
    return "\n".join(lines)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def ai_markdown(text: str) -> None:
    """Render AI text. Dollar signs are escaped: Streamlit treats "$2.36 to $474.57" as a LaTeX formula."""
    st.markdown(text.replace("$", r"\$"))


# ── Left panel: data sources ──────────────────────────────────────
def select_source(key: tuple) -> None:
    st.session_state["selected"] = key


def clear_folder() -> None:
    st.session_state["folder_gen"] = st.session_state.get("folder_gen", 0) + 1
    if (st.session_state.get("loaded_key") or ("",))[0] == "file":
        clear_dataset()


def file_labels(paths) -> dict:
    """Display name per uploaded path: just the file name, no folder or extension
    ("archive (1)/rentals.csv" → "rentals"). A name found more than once gets its folder
    (or, within the same folder, its extension) appended so the entries stay distinct."""
    paths = list(paths)
    counts = Counter(PurePosixPath(p).stem for p in paths)
    labels = {}
    for p in paths:
        path = PurePosixPath(p)
        labels[p] = path.stem if counts[path.stem] == 1 else f"{path.stem} ({path.parent.name or 'root'})"
    clashes = Counter(labels.values())
    for p in paths:
        if clashes[labels[p]] > 1:
            labels[p] = f"{labels[p][:-1]}, {PurePosixPath(p).suffix.lstrip('.')})" if labels[p].endswith(")") \
                else f"{labels[p]} ({PurePosixPath(p).suffix.lstrip('.')})"
    return labels


def source_item(label: str, key: tuple, widget_key: str) -> None:
    """One clickable row in the source list; the active dataset gets the highlighted style."""
    # "selected" is set by the click callback before this run, so the highlight moves on the same click
    active = (st.session_state.get("selected") or st.session_state.get("loaded_key")) == key
    st.button(label, key=f"{'navactive' if active else 'nav'}_{widget_key}", type="tertiary",
              width="stretch", on_click=select_source, args=(key,))


def load_views() -> None:
    """Fetch the view list once per session (not on every rerun). A failure is kept too, so a down
    server doesn't cost a connection timeout on every click — the refresh button retries."""
    if "views" in st.session_state:
        return
    with st.spinner("Loading views..."):
        try:
            st.session_state["views"] = get_all_views()
            st.session_state.pop("views_error", None)
        except Exception as e:
            st.session_state["views"] = []
            st.session_state["views_error"] = str(e)


def render_sources() -> dict:
    """Left panel. Returns the files from the selected folder, keyed by name."""
    html_block('<p class="panel-title">Data Sources</p>'
               '<p class="panel-sub">Select a view or file to analyze.</p>')
    query = st.text_input("Search", placeholder="Search views and files",
                          label_visibility="collapsed", key="source_search").strip().lower()

    # SQL Server views
    col_label, col_refresh = st.columns([5, 1], vertical_alignment="center")
    with col_label:
        html_block('<div class="section-label flush">SQL Server views</div>')
    with col_refresh:
        if st.button(":material/refresh:", key="refresh_views", type="tertiary", help="Refresh view list"):
            st.session_state.pop("views", None)
    load_views()
    views = st.session_state["views"]
    if st.session_state.get("views_error"):
        st.error(f"Could not list views. Check the .env connection settings. {st.session_state['views_error']}")
    elif not views:
        html_block('<div class="muted">No views found.</div>')
    else:
        shown = [v for v in views if query in v.lower()]
        for i, view in enumerate(shown):
            source_item(view, ("view", view), f"v{i}")
        if not shown:
            html_block('<div class="muted">No views match your search.</div>')

    # Files from a folder (the browser uploads every Excel/CSV file in it and its subfolders).
    # Once a folder is loaded, style.css hides the picker and its file chips; the list below replaces them.
    html_block('<div class="section-label">Files</div>')
    files = st.file_uploader(
        "Folder", type=["xlsx", "xls", "csv"], accept_multiple_files="directory",
        label_visibility="collapsed", key=f"folder_{st.session_state.get('folder_gen', 0)}",
    ) or []
    files_by_name = {f.name: f for f in files}
    if files_by_name:
        top_folders = {PurePosixPath(n).parts[0] for n in files_by_name if len(PurePosixPath(n).parts) > 1}
        folder = f"{top_folders.pop()} · " if len(top_folders) == 1 else ""
        col_info, col_change = st.columns([3, 2], vertical_alignment="center")
        with col_info:
            html_block(f'<div class="folder-info">{html.escape(folder)}{len(files_by_name)} files</div>')
        with col_change:
            st.button("Change folder", key="change_folder", type="tertiary", on_click=clear_folder)

    labels = file_labels(files_by_name)
    shown = sorted((n for n in files_by_name if query in labels[n].lower()), key=lambda n: labels[n].lower())
    for i, name in enumerate(shown):
        f = files_by_name[name]
        source_item(labels[name], ("file", f.name, f.size), f"f{i}")
    if files_by_name and not shown:
        html_block('<div class="muted">No files match your search.</div>')
    return files_by_name


# ── Middle panel: dataset summary ─────────────────────────────────
def sync_selection(files_by_name: dict) -> None:
    """Load whatever was clicked in the left panel, unless it's already the active dataset."""
    loaded = st.session_state.get("loaded_key")
    if loaded and loaded[0] == "file" and loaded[1] not in files_by_name:
        clear_dataset()  # its folder was cleared or replaced — drop the stale data
        return

    selected = st.session_state.get("selected")
    if not selected or selected == loaded:
        return
    kind, name = selected[0], selected[1]
    label = name if kind == "view" else file_labels(files_by_name).get(name, name)
    try:
        with st.spinner(f"Loading {label}..."):
            if kind == "view":
                df = read_view(name)
            else:
                f = files_by_name[name]
                df = read_upload(f.name, f.getvalue())
    except Exception as e:
        st.session_state["selected"] = loaded
        show_error(f"Could not load {label}", e)
        return
    set_dataset(df, label, is_sql=kind == "view")
    st.session_state["loaded_key"] = selected


def render_summary_card(df: pd.DataFrame) -> None:
    with st.container(border=True, key="summary_card"):
        html_block('<div class="card-head"><span class="card-title">Executive Summary</span>'
                   '<span class="tag">AI-generated · verify with source</span></div>')

        summary = get_result("summary")
        if summary is None and not get_result("summary_error"):
            with st.spinner("Generating executive summary..."):
                try:
                    summary = generate_summary(df)
                    store_result("summary", summary)
                except Exception as e:
                    store_result("summary_error", str(e))

        if summary:
            ai_markdown(summary)
            st.download_button("Download summary", summary, file_name="kestra_executive_summary.txt",
                               mime="text/plain", type="tertiary", icon=":material/download:")
        else:
            st.error(f"Could not generate the summary: {get_result('summary_error')}")
            if st.button("Try again", key="summary_retry", type="tertiary"):
                st.session_state["results"].pop("summary_error", None)
                st.rerun()


def render_quality(df: pd.DataFrame) -> None:
    with st.expander("Data Quality", expanded=False):
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / max(len(df), 1) * 100).round(1)
        missing = missing[missing["Missing Count"] > 0]
        negatives = pd.DataFrame([
            {"Column": col, "Negative Count": int((df[col] < 0).sum())}
            for col in df.select_dtypes(include="number").columns
            if (df[col] < 0).any()
        ])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing values**")
            if len(missing):
                st.dataframe(missing, width="stretch", hide_index=True)
            else:
                st.caption("No missing values.")
        with col2:
            st.markdown("**Negative values**")
            if len(negatives):
                st.dataframe(negatives, width="stretch", hide_index=True)
            else:
                st.caption("No negative values.")

        if st.button("Run AI quality review", key="validate_btn", type="secondary"):
            try:
                with st.spinner("Reviewing data quality..."):
                    store_result("validate", validate_data(df))
            except Exception as e:
                show_error("Validation failed", e)

        report = get_result("validate")
        if report:
            ai_markdown(report)
            st.download_button("Download report", report, file_name="kestra_validation_report.txt",
                               mime="text/plain", type="tertiary", icon=":material/download:")


def render_overview(files_by_name: dict) -> None:
    sync_selection(files_by_name)
    df = st.session_state.get("df")
    if df is None:
        html_block('<div class="empty"><div class="empty-title">Select a data source</div>'
                   '<div class="empty-sub">Choose a SQL Server view or a file from the left panel to see '
                   'its executive summary, data quality and a preview.</div></div>')
        return

    name = st.session_state.get("source_name", "Dataset")
    stats = st.session_state["stats"]
    kind = "SQL view" if st.session_state.get("is_sql_source") else "File"
    html_block(
        f'<div class="ds-title">{html.escape(name)}<span class="tag">{kind}</span></div>'
        f'<div class="ds-meta"><span>Rows <b>{stats["rows"]:,}</b></span>'
        f'<span>Columns <b>{stats["cols"]:,}</b></span>'
        f'<span>Missing values <b>{stats["missing"]:,}</b></span>'
        f'<span>Duplicate rows <b>{stats["dupes"]:,}</b></span></div>'
    )

    render_summary_card(df)
    render_quality(df)
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(20), width="stretch")
        st.caption(f"Showing first 20 of {len(df):,} rows")
    with st.expander("Column Statistics", expanded=False):
        st.dataframe(df.describe(include="all").T.astype(str), width="stretch")


# ── Right panel: chat ─────────────────────────────────────────────
def clear_chat() -> None:
    st.session_state["chat"] = []


def render_chat() -> None:
    """Chat with memory: every turn is kept in session_state and earlier turns are sent to the AI."""
    df = st.session_state.get("df")
    source_name = st.session_state.get("source_name")
    is_sql = st.session_state.get("is_sql_source", False)
    messages = st.session_state.setdefault("chat", [])

    col_title, col_dl, col_clear = st.columns([6, 1, 1], vertical_alignment="center")
    with col_title:
        html_block('<p class="panel-title">Ask About This Data</p>'
                   '<p class="panel-sub">Ask questions in plain English.</p>')
    if messages:
        with col_dl:
            st.download_button(":material/download:", chat_transcript(messages), file_name="kestra_chat.txt",
                               mime="text/plain", key="chat_dl", type="tertiary", help="Download chat")
        with col_clear:
            st.button(":material/delete:", key="chat_clear", type="tertiary", help="Clear chat",
                      on_click=clear_chat)

    box = st.container(height=CHAT_HEIGHT, border=False, key="chat_box")
    with box:
        if not messages:
            hint = ("Ask anything about the selected data — totals, trends, top records, comparisons."
                    if df is not None else "Select a data source to start asking questions.")
            html_block(f'<div class="chat-empty">{hint}</div>')
        for msg in messages:
            render_message(msg)

    question = st.chat_input("Ask a question about this data...", key="chat_input")
    if df is not None and not question:
        question = st.session_state.pop("pending_question", None)
    if not question or not question.strip():
        return
    question = question.strip()

    if df is None:
        messages.append({"role": "user", "content": question})
        route = route_message(question, messages[:-1], [])
        if route["intent"] == INTENT_CHITCHAT:
            # Small talk doesn't need data loaded — answer right away.
            messages.append({"role": "assistant", "content": route.get("reply") or "Hi! How can I help you today?"})
        else:
            # A real data question: park it, ask the user to pick a source, answer it once loaded
            st.session_state["pending_question"] = question
            messages.append({"role": "assistant", "content": "I don't have any data loaded yet. Pick a view or a "
                             "file from the left panel, and I'll answer your question right away."})
        st.rerun()

    history = list(messages)
    messages.append({"role": "user", "content": question})
    with box:
        render_message(messages[-1])
        try:
            with st.chat_message("assistant", avatar=AI_AVATAR):
                with st.spinner("Thinking..."):
                    reply = respond(df, source_name, is_sql, question, history)
            messages.append(reply)
            st.rerun()
        except Exception as e:
            messages.pop()  # drop the unanswered question so it doesn't pollute the history
            show_error("AI request failed", e)


# ── Layout ────────────────────────────────────────────────────────
load_css(BASE_DIR / "style.css")

logo_b64 = base64.b64encode((BASE_DIR / "assets" / "kestra_logo.png").read_bytes()).decode()
html_block(
    '<div class="topbar"><div class="brand-row">'
    f'<img class="brand-logo" src="data:image/png;base64,{logo_b64}" alt="Kestra Medical Technologies">'
    '<span class="brand">InsightIQ</span>'
    '<span class="brand-sub">Data Insights Assistant</span>'
    '</div></div>'
)

left, middle, right = st.columns([1.1, 2.3, 1.7], gap="medium")
with left:
    with st.container(height=PANEL_HEIGHT, border=True, key="panel_sources"):
        folder_files = render_sources()
with middle:
    with st.container(height=PANEL_HEIGHT, border=True, key="panel_summary"):
        render_overview(folder_files)
with right:
    with st.container(height=PANEL_HEIGHT, border=True, key="panel_chat"):
        render_chat()
