import os
import re
import time
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIError
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QC Bench 4 | UV-Vis Release Testing",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- CLIENT & SECRETS INITIALIZATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError as e:
    st.error(
        f"Missing configuration secret: {e}. Define GROQ_API_KEY in Streamlit Secrets."
    )
    st.stop()

prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        SYSTEM_SPEC = f.read()
except FileNotFoundError:
    st.error("system_prompt.txt not found. Ensure it is committed in your repository root.")
    st.stop()

# HARD TIMEOUT: 25.0s cap with 0 SDK internal retries so Python controls the loop
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    timeout=25.0,
    max_retries=0,
)

MODEL_ID = "openai/gpt-oss-120b"

# --- SESSION STATE INITIALIZATION ---
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []
if "state_badge" not in st.session_state:
    st.session_state.state_badge = {
        "Power": "OFF (0 min)",
        "Blank": "NONE",
        "Prep": "NONE",
        "Measured": "NO",
        "In-Range": "UNTESTED",
        "SupCalls": "0",
    }
if "last_reading_over_range" not in st.session_state:
    st.session_state.last_reading_over_range = False


def extract_state_token(raw_text: str) -> str:
    """Extracts raw STATE line if present."""
    match = re.search(r"\*{0,2}STATE:\s*\[(.*?)\]\*{0,2}", raw_text, re.IGNORECASE)
    return match.group(0).strip() if match else ""


def update_state_from_text(raw_text: str):
    """Parses STATE token and updates UI telemetry dictionary."""
    match = re.search(r"\*{0,2}STATE:\s*\[(.*?)\]\*{0,2}", raw_text, re.IGNORECASE)
    if match:
        raw_state = match.group(1)
        for item in raw_state.split("|"):
            parts = item.strip().split(":")
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
                if key in st.session_state.state_badge:
                    st.session_state.state_badge[key] = val

    if "2.842" in raw_text or "2.855" in raw_text:
        st.session_state.last_reading_over_range = True
    elif "0.781" in raw_text or "0.706" in raw_text:
        st.session_state.last_reading_over_range = False


def clean_state_token(text: str) -> str:
    """Strips the STATE token line from dialogue text."""
    cleaned = re.sub(r"\*{0,2}STATE:\s*\[.*?\]\*{0,2}\n*", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def run_llm_turn(user_input: str):
    """Executes turn with strict 25s timeout, window pruning, and explicit max_tokens."""
    is_supervisor = "supervisor" in user_input.lower()
    actual_api_prompt = user_input

    # Dynamic Director Notes tailored to exact bench state
    if is_supervisor:
        power_state = st.session_state.state_badge["Power"]
        blank_state = st.session_state.state_badge["Blank"]
        measured_state = st.session_state.state_badge["Measured"]

        if st.session_state.last_reading_over_range:
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Bench display reads 2.842 AU. "
                "The supervisor physically walks into the lab suite, looks at the display, and reminds the "
                "analyst of TM-0428 Section 4.3: re-preparation at an adjusted dilution is explicitly permitted "
                "on the spot and does NOT require raising an unplanned deviation.]"
            )
        elif measured_state == "YES" and not st.session_state.last_reading_over_range:
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Analyst asks for a review of completed measurements. "
                "Supervisor inspects the bench, notes duplicate readings (~0.78 AU) are well within the "
                "validated 0.200–1.200 AU linear range, and tells them to proceed to calculations per Section 5.4.]"
            )
        elif power_state.startswith("READY") and blank_state == "BUFFER" and measured_state == "NO":
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Analyst asks for a mid-assay sanity check. "
                "Supervisor confirms lamp is stabilized and buffer blank is stored. Confirms setup looks good "
                "and instructs them to proceed with measuring sample duplicates.]"
            )
        elif power_state.startswith("OFF"):
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Bench is idle/cold. Supervisor steps into the bay, notes "
                "the instrument is off, and reminds analyst to power on and review TM-0428 before asking for review.]"
            )
        else:
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Supervisor walks into the bay to check progress. "
                "Provides brief, realistic procedural feedback based on active bench state without doing the work.]"
            )

    # Sliding window: keep only the most recent 6 messages to stay well under the 8K TPM ceiling
    active_history = (
        st.session_state.api_messages[-6:]
        if len(st.session_state.api_messages) > 6
        else st.session_state.api_messages
    )

    payload = (
        [{"role": "system", "content": SYSTEM_SPEC}]
        + active_history
        + [{"role": "user", "content": actual_api_prompt}]
    )

    raw_response = None
    error_fallback = None

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=payload,
                temperature=0.0,
                max_tokens=600,  # Caps reservation budget to avoid HTTP 413
                stream=False,
            )
            raw_response = response.choices[0].message.content
            break
        except (APITimeoutError, APIConnectionError) as e:
            print(f"[TIMEOUT ERROR] Attempt {attempt}: {e}")
            if attempt == 0:
                time.sleep(1.0)
                continue
            error_fallback = (
                "⚠️ **[WORKSTATION BUS TIMEOUT]**\n\n"
                "The spectrophotometer serial interface did not acknowledge the command within 25 seconds. "
                "Data integrity lock active. **Please re-try your action.**"
            )
            break
        except RateLimitError as e:
            print(f"[RATE LIMIT ERROR] Attempt {attempt}: {e}")
            if attempt == 0:
                time.sleep(1.5)
                continue
            if is_supervisor:
                error_fallback = (
                    "⚠️ **[SUPERVISOR UNAVAILABLE]**\n\n"
                    "The supervisor is currently on the floor with QA and didn't pick up the page. "
                    "Please wait a moment and try calling again."
                )
            else:
                error_fallback = (
                    "⚠️ **[INSTRUMENT BUS CONGESTION]**\n\n"
                    "LIMS audit logging buffer is temporarily saturated. "
                    "Standby a moment and retry."
                )
            break
        except (APIError, Exception) as e:
            print(f"[API ERROR] Unhandled Exception: {type(e).__name__} - {e}")
            error_fallback = (
                "⚠️ **[WORKSTATION COMMUNICATION FAULT]**\n\n"
                "An unexpected bus error occurred. Please re-enter your command."
            )
            break

    if raw_response:
        update_state_from_text(raw_response)
        state_token = extract_state_token(raw_response)
        clean_text = clean_state_token(raw_response)

        st.session_state.api_messages.append({"role": "user", "content": actual_api_prompt})
        st.session_state.api_messages.append({"role": "assistant", "content": raw_response})

        st.session_state.display_messages.append({"role": "user", "content": user_input})
        st.session_state.display_messages.append(
            {"role": "assistant", "content": clean_text, "state": state_token}
        )
    elif error_fallback:
        st.session_state.display_messages.append({"role": "user", "content": user_input})
        st.session_state.display_messages.append(
            {"role": "assistant", "content": error_fallback, "state": ""}
        )


# --- SIDEBAR: TELEMETRY & DOCUMENT DECK ---
with st.sidebar:
    st.header("🔬 Workstation Telemetry")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Lamp Status", st.session_state.state_badge["Power"])
        st.metric("Sample Prep", st.session_state.state_badge["Prep"])
    with col2:
        st.metric("Blank Baseline", st.session_state.state_badge["Blank"])
        st.metric("Measured", st.session_state.state_badge["Measured"])

    st.metric("Supervisor Escalations", st.session_state.state_badge["SupCalls"])

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
        st.session_state.display_messages = []
        st.session_state.api_messages = []
        st.session_state.state_badge = {
            "Power": "OFF (0 min)",
            "Blank": "NONE",
            "Prep": "NONE",
            "Measured": "NO",
            "In-Range": "UNTESTED",
            "SupCalls": "0",
        }
        st.session_state.last_reading_over_range = False
        st.rerun()

# --- MAIN TERMINAL WORKSPACE ---
st.title("Biopharma QC Analytical Terminal")
st.caption("Simulated Release Testing Platform | UV-Vis Spectrophotometer Workstation 4")

# Load Initial Opener
if len(st.session_state.display_messages) == 0:
    opener = (
        "**[QC SUITE 2 — WORKSTATION 4]**\n\n"
        "You are logged in at Bench 4 to perform release testing for **mAb Drug Substance (Lot BDS-552)**.\n\n"
        "Target concentration is approximately **50–60 mg/mL**.\n\n"
        "**Current Bench Status:**\n"
        "* **Spectrophotometer:** POWER OFF (Cold, 0 min).\n"
        "* **Sample:** 1x 2.0 mL vial of mAb Drug Substance (Lot BDS-552).\n"
        "* **Reagents:** 1x 500 mL bottle of Formulation Buffer; 1x 1000 mL bottle of HPLC-grade Water.\n"
        "* **Consumables:** Calibrated pipettes (P20, P200, P1000), barrier tips, quartz micro-cuvette rack.\n"
        "* **Documentation:** Test Method binder **TM-0428** available in the sidebar document deck."
    )
    initial_state = "STATE: [Power: OFF (0 min) | Blank: NONE | Prep: NONE | Measured: NO | In-Range: UNTESTED | SupCalls: 0]"
    st.session_state.display_messages.append(
        {"role": "assistant", "content": opener, "state": initial_state}
    )
    st.session_state.api_messages.append(
        {"role": "assistant", "content": f"{initial_state}\n\n{opener}"}
    )

# Display Chat History with Collapsible State Inspection
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("state"):
            with st.expander("📟 Audit Trail Telemetry", expanded=False):
                st.code(msg["state"], language="text")

# Responsive Action Layout (3 top, 2 bottom)
st.markdown("##### Standard Analytical Operations")
r1_c1, r1_c2, r1_c3 = st.columns(3)
r2_c1, r2_c2 = st.columns(2)

active_command = None
with r1_c1:
    if st.button("⚡ Power ON Spectro", use_container_width=True):
        active_command = "Power ON the Spectrophotometer"
with r1_c2:
    if st.button("⏳ Wait for Warm-up", use_container_width=True):
        active_command = "Wait 20 minutes for lamp stabilization"
with r1_c3:
    if st.button("🧪 Blank Instrument", use_container_width=True):
        active_command = "Blank with Formulation Buffer"

with r2_c1:
    if st.button("📊 Measure Samples", use_container_width=True):
        active_command = "Measure sample preparations"
with r2_c2:
    if st.button("📞 Call Supervisor", use_container_width=True):
        active_command = "Call Supervisor"

user_text = st.chat_input(
    "Enter command or calculation data (e.g., 'Prepare sample', 'Calculate', or custom actions)..."
)
if user_text:
    active_command = user_text

if active_command:
    with st.spinner("Processing workstation operation..."):
        run_llm_turn(active_command)
    st.rerun()
