import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_client = None


def get_client() -> Groq:
    """Create the Groq client lazily so a missing key gives a clear error instead of failing at import."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file.")
        _client = Groq(api_key=api_key)
    return _client


def chat(system: str, user: str, max_tokens: int = 1000, history: list | None = None,
         reasoning_effort: str | None = None) -> str:
    """Single deterministic chat completion; returns the reply text.

    `history` is an optional list of {"role", "content"} turns placed between the system prompt and `user`.
    `reasoning_effort` ("low"/"medium"/"high") matters for reasoning models like gpt-oss: their hidden
    "thinking" tokens are drawn from the same `max_tokens` budget as the visible reply, so on a non-trivial
    prompt the model can burn the whole budget thinking and return empty content. Pass "low" for short,
    mechanical replies (e.g. generated code) where deep reasoning isn't needed, to leave room for the answer.
    """
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history or []]
    messages.append({"role": "user", "content": user})
    kwargs = {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Speech-to-text via Groq Whisper. Returns the transcribed text (may be empty)."""
    result = get_client().audio.transcriptions.create(
        model="whisper-large-v3",
        file=(filename, audio_bytes),
    )
    return (result.text or "").strip()


def text_to_speech(text: str, voice: str = "troy") -> tuple[bytes, str]:
    """Text-to-speech. Returns (audio_bytes, mime_type).

    Tries Groq's Orpheus TTS first. Falls back to gTTS (free, no API key, needs internet) if the Groq
    model isn't usable on this account yet — e.g. its terms haven't been accepted at
    https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english (org admin only).
    """
    try:
        response = get_client().audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice,
            input=text,
            response_format="wav",
        )
        return response.read(), "audio/wav"
    except Exception:
        from io import BytesIO

        from gtts import gTTS
        buffer = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buffer)
        return buffer.getvalue(), "audio/mp3"


def strip_code_fences(text: str) -> str:
    """Remove markdown fences the model sometimes adds despite instructions."""
    for fence in ("```python", "```sql", "```json", "```"):
        text = text.replace(fence, "")
    return text.strip()


def parse_json_object(text: str) -> dict | None:
    """Extract the first JSON object from a model reply, or None if there isn't a valid one."""
    match = re.search(r"\{.*\}", strip_code_fences(text), re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ── Shared building blocks for "AI writes a query → we run it → AI explains" ──
class CodeRunError(Exception):
    """AI-generated code failed even after retrying. Keeps the last code and the underlying error."""

    def __init__(self, code: str, original: Exception):
        super().__init__(str(original))
        self.code = code
        self.original = original


def generate_and_run(system: str, user: str, run, max_retries: int = 1, max_tokens: int = 1000,
                      reasoning_effort: str | None = None):
    """Ask the AI for code, run it, and on failure show the AI the error so it can fix its own code.

    `run(code)` must return the result or raise. Returns (code, result); raises CodeRunError when
    every attempt failed.
    """
    code = strip_code_fences(chat(system, user, max_tokens=max_tokens, reasoning_effort=reasoning_effort))
    for attempt in range(max_retries + 1):
        try:
            return code, run(code)
        except Exception as e:
            if attempt == max_retries:
                raise CodeRunError(code, e) from e
            code = strip_code_fences(chat(
                system,
                f"{user}\n\nYour previous attempt:\n{code}\n\nIt failed with this error: {e}\n"
                "Write a corrected version. Return only the corrected code.",
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            ))


EXPLAIN_SYSTEM_PROMPT = """You are a senior data analyst.
Explain query results clearly and concisely in plain English.
STRICT RULES:
- Use ONLY the exact values from the result provided
- Never change, round or assume any numbers
- The full result table is shown to the user separately, so do not re-print it; summarise the key findings"""


def explain_result(question: str, query: str, result_text: str, note: str = "") -> str:
    """One shared 'explain this result' step for both the pandas and SQL paths."""
    return chat(
        EXPLAIN_SYSTEM_PROMPT,
        f"""
Question: {question}
Query used:
{query}
Actual result:
{result_text}{note}
Explain this result in plain English using exact values.
""",
    )
