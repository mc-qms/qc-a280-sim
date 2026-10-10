```python
import re
from openai import OpenAI
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QC Bench 4 | UV-Vis Release Testing",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CLIENT & SECRETS INITIALIZATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SYSTEM_SPEC = st.secrets["QC_SIM_SPEC"]
except KeyError as e:
    st.error(
        f"Missing configuration secret: {e}. Define GROQ_API_KEY and QC_SIM_SPEC in Streamlit Secrets."
    )
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

MODEL_ID = "openai/gpt-oss-20b"

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "state_badge" not in st.session_state:
    st.session_state.state_badge = {
        "Power": "OFF (0 min)",
        "Blank": "NONE",
        "Prep": "NONE",
        "Measured": "NO",
        "In-Range": "UNTESTED",
    }


def parse_and_strip_state(text: str) -> str:
    """Extracts STATE token to update UI badges and strips it from dialogue display."""
    match = re.search(r"^STATE:\s*\[(.*?)\]", text, re.MULTILINE)
    if match:
        raw_state = match.group(1)
        for item in raw_state.split("|"):
            parts = item.strip().split(":")
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
                if key in st.session_state.state_badge:
                    st.session_state.state_badge[key] = val

        cleaned = re.sub(r"^STATE:\s*\[.*?\]\n*", "", text, flags=re.MULTILINE)
        return cleaned.strip()
    return text


def execute_turn(user_input: str):
    """Appends user message, calls Groq API, updates state badges, and renders output."""
    st.session_state.messages.append({"role": "user", "content": user_input})

    payload = [{"role": "system", "content": SYSTEM_SPEC}] + st.session_state.messages

    with st.chat_message("assistant"):
        response_box = st.empty()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=payload,
            temperature=0.0,
            stream=False,
        )
        raw_text = response.choices[0].message.content
        cleaned_text = parse_and_strip_state(raw_text)
        response_box.markdown(cleaned_text)
        st.session_state.messages.append({"role": "assistant", "content": cleaned_text})


# --- SIDEBAR: TELEMETRY & DOCUMENT DECK ---
with st.sidebar:
    st.header("🔬 Workstation Telemetry")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Power / Lamp", st.session_state.state_badge["Power"])
        st.metric("Sample Prep", st.session_state.state_badge["Prep"])
    with col2:
        st.metric("Blank Baseline", st.session_state.state_badge["Blank"])
        st.metric("Measured", st.session_state.state_badge["Measured"])

    st.divider()

    st.subheader("📋 Document Deck: TM-0428 (Rev. 04)")
    tab1, tab2, tab3 = st.tabs(["Page 1", "Page 2", "Page 3"])

    with tab1:
        st.caption("Section 3: Equipment & System Suitability")
        st.markdown(
            """
            * **3.1 Instrument:** Dual-beam UV-Vis with quartz micro-cuvette (10 mm pathlength).
            * **3.2 Lamp Warm-Up:** Lamps must stabilize **>= 20 minutes** before blanking or sample analysis.
            * **3.3 Optical Inspection:** Ensure quartz optical faces are clean, unchipped, and properly oriented.
            """
        )
    with tab2:
        st.caption("Section 4: Solution Preparation & Blanking")
        st.markdown(
            """
            * **4.1 Baseline Zero (Blank):** Zero against **Formulation Buffer**. Do NOT use water; matrix excipients require baseline subtraction.
            * **4.2 Sample Prep:** Prepare duplicate independent 1:100 dilutions (Prep A and B) using Formulation Buffer as diluent.
            * **4.3 Linear Range & Re-Prep:** Validated linear range is **0.200 to 1.200 AU** at 280 nm. If reading exceeds 1.200 AU, re-preparation with adjusted dilution is permitted and does NOT constitute a deviation.
            """
        )
    with tab3:
        st.caption("Section 5: Testing & Acceptance Criteria")
        st.markdown(
            """
            * **5.1 Wavelengths:** Record 280 nm (protein peak) and 320 nm (scatter/turbidity).
            * **5.2 Correction:** A_corr = A280 - A320
            * **5.3 Quantitation:** Extinction Coefficient = 1.45 mL / (mg · cm)
            * **5.4 Precision:** % Difference between duplicates must be **<= 2.0%**.
            """
        )

    st.divider()
    if st.button("🔄 Reset Workstation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.state_badge = {
            "Power": "OFF (0 min)",
            "Blank": "NONE",
            "Prep": "NONE",
            "Measured": "NO",
            "In-Range": "UNTESTED",
        }
        st.rerun()

# --- MAIN TERMINAL WORKSPACE ---
st.title("Biopharma QC Analytical Terminal")
st.caption("Simulated Release Testing Platform | UV-Vis Spectrophotometer Workstation 4")

# Load Initial Opener if History is Empty
if len(st.session_state.messages) == 0:
    opener = (
        "**[QC SUITE 2 — UV-VIS WORKSTATION]**\n\n"
        "You are logged in at Bench 4 to perform release testing for **mAb Drug Substance (Lot BDS-552)**.\n\n"
        "Target concentration is approximately **50–60 mg/mL**.\n\n"
        "**Current Bench Status:**\n"
        "* **Spectrophotometer:** POWER OFF (Cold, 0 min).\n"
        "* **Sample:** 1x 2.0 mL vial of mAb Drug Substance (Lot BDS-552).\n"
        "* **Reagents:** 1x 500 mL bottle of Formulation Buffer; 1x 1000 mL bottle of HPLC-grade Water.\n"
        "* **Consumables:** Calibrated pipettes (P20, P200, P1000), barrier tips, quartz micro-cuvette rack.\n"
        "* **Documentation:** Test Method binder **TM-0428** available in the sidebar document deck."
    )
    st.session_state.messages.append({"role": "assistant", "content": opener})

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick-Action Button Grid
st.markdown("##### Standard Analytical Operations")
bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)

button_action = None
with bcol1:
    if st.button("⚡ Power ON Spectro", use_container_width=True):
        button_action = "Power ON the Spectrophotometer"
with bcol2:
    if st.button("⏳ Wait for Warm-up", use_container_width=True):
        button_action = "Wait 20 minutes for lamp stabilization"
with bcol3:
    if st.button("🧪 Blank Instrument", use_container_width=True):
        button_action = "Blank with Formulation Buffer"
with bcol4:
    if st.button("📊 Measure Samples", use_container_width=True):
        button_action = "Measure sample preparations"
with bcol5:
    if st.button("📞 Call Supervisor", use_container_width=True):
        button_action = "Call Supervisor"

# Process Quick-Action Button Input
if button_action:
    execute_turn(button_action)
    st.rerun()

# Free-Text Input for Custom Actions & Calculations
user_prompt = st.chat_input(
    "Enter command or calculation data (e.g., 'Prepare sample', 'Calculate', or custom actions)..."
)
if user_prompt:
    execute_turn(user_prompt)
    st.rerun()

```
