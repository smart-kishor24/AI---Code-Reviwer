import os
import json
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Code Reviewer", layout="wide")

st.title("AI Code Reviewer")
st.markdown("Paste Python code; Gemini will explain it and suggest fixes (JSON output).")

# sidebar controls
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input(
    "GEMINI API KEY (leave empty to use env var GEMINI_API_KEY)", type="password"
)
model = st.sidebar.selectbox(
    "Model",
    options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
max_output_tokens = st.sidebar.number_input("Max output tokens", min_value=128, max_value=8192, value=800)

# create client (if api key given, use it; else rely on GEMINI_API_KEY env var)
api_key = api_key_input.strip() or os.environ.get("GEMINI_API_KEY")
try:
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()  # will pick up GEMINI_API_KEY from env if present
except Exception as e:
    st.error(f"Failed to initialize GenAI client: {e}")
    st.stop()

# sample starter code
DEFAULT_CODE = """def factorial(n):
    \"\"\"Return factorial of n (int).\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""

code = st.text_area("Paste your Python code here", value=DEFAULT_CODE, height=360)

action = st.radio("Action", ["Explain code", "Suggest improvements", "Return patched file (apply fixes)"])

run_button = st.button("Run review")

if run_button:
    if not code.strip():
        st.warning("Paste some Python code first.")
    else:
        # system instruction: strict JSON output
        system_instruction = (
            "You are an expert senior Python developer and code reviewer. "
            "When given Python source code, return a **single valid JSON object** with keys:\n"
            "  - summary: short (1-3 lines) description of what the code does\n"
            "  - issues: a list of objects {line_start, line_end, severity, title, explanation, fix}\n"
            "  - suggestions: short bullet-list of higher-level suggestions\n"
            "  - fixed_code: the full file content after applying the fixes (if possible)\n\n"
            "Requirements: respond **ONLY** with valid JSON (no extra commentary). "
            "Include line numbers where possible and short code snippets under 'fix'."
        )

        user_prompt = f"Action: {action}\n\nCode:\n```python\n{code}\n```"

        contents = [system_instruction, user_prompt]

        with st.spinner("Calling Gemini..."):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=float(temperature),
                        max_output_tokens=int(max_output_tokens),
                        response_mime_type="application/json",
                    ),
                )
                text = resp.text  # SDK exposes .text
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        # Try to parse JSON (models occasionally return invalid JSON; handle gracefully)
        try:
            parsed = json.loads(text)
        except Exception:
            st.error("Model did not return valid JSON. Raw output shown below.")
            st.code(text)
            st.stop()

        # Display parsed output
        st.subheader("Summary")
        st.write(parsed.get("summary", ""))

        st.subheader("Issues")
        issues = parsed.get("issues", [])
        if not issues:
            st.write("No issues found (or model returned none).")
        else:
            for i, iss in enumerate(issues, 1):
                st.markdown(f"**{i}. {iss.get('title','(no title)')}** — lines {iss.get('line_start')}–{iss.get('line_end')} ({iss.get('severity')})")
                st.write(iss.get("explanation", ""))
                if iss.get("fix"):
                    st.code(iss.get("fix"), language="python")

        st.subheader("High-level suggestions")
        for s in parsed.get("suggestions", []):
            st.write(f"- {s}")

        st.subheader("Patched file (fixed_code)")
        fixed = parsed.get("fixed_code", "")
        if fixed:
            st.code(fixed, language="python")
        else:
            st.write("No patched file returned by the model.")
