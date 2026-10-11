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

# --- CLIENT & SECRETS INITIALIZATION (OPENROUTER) ---
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except KeyError as e:
    st.error(
        f"Missing configuration secret: {e}. Define OPENROUTER_API_KEY in Streamlit Secrets."
    )
    st.stop()

prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        SYSTEM_SPEC = f.read()
except FileNotFoundError:
    st.error("system_prompt.txt not found. Ensure it is committed in your repository root.")
    st.stop()

# Hard 25.0s cap with 0 SDK retries so Python controls the loop
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=25.0,
    max_retries=0,
)

# High-speed production model (~$0.00025 per interaction)
MODEL_ID = "meta-llama/llama-3.3-70b-instruct"

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
if "last_readings" not in st.session_state:
    st.session_state.last_readings = None


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
    elif "0.781" in raw_text or "0.706" in raw_text or "0.966" in raw_text or "0.856" in raw_text:
        st.session_state.last_reading_over_range = False


def clean_state_token(text: str) -> str:
    """Strips the STATE token line from dialogue text."""
    cleaned = re.sub(r"\*{0,2}STATE:\s*\[.*?\]\*{0,2}\n*", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def run_llm_turn(user_input: str):
    """Executes turn with deterministic physics, strict math evaluation, and template enforcement."""
    actual_api_prompt = user_input
    u_lower = user_input.lower().strip()

    power_state = st.session_state.state_badge["Power"].upper()
    blank_state = st.session_state.state_badge["Blank"].upper()
    prep_state = st.session_state.state_badge["Prep"].upper()
    measured_state = st.session_state.state_badge["Measured"].upper()

    # =========================================================================
    # 1. SUPERVISOR INTERACTION ROUTING
    # =========================================================================
    if any(k in u_lower for k in ["supervisor", "ask lead", "elevate"]):
        if power_state.startswith("OFF"):
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Bench is idle/cold. Supervisor steps into the bay, notes "
                "the instrument is off, and reminds analyst to power on and review TM-0428 Section 4 before asking for review.]"
            )
        elif st.session_state.last_reading_over_range and measured_state == "YES":
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Bench display reads 2.842 AU. Supervisor walks into the bay, "
                "looks at the display, and reminds analyst of TM-0428 Section 5.3: re-preparation at an adjusted dilution "
                "is explicitly permitted on the spot and does NOT require raising an unplanned deviation.]"
            )
        elif measured_state == "YES" and blank_state == "NONE":
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Analyst measured sample without blanking (Blank: NONE). "
                "Supervisor reviews raw scan data, spots complete absence of blank entry, and points out the "
                "elevated absorbances (~0.96 AU) that will calculate to ~66 mg/mL (exceeding 50-60 target). "
                "Commends analyst for stopping before calculating, logs in-process non-conformance DEV-QC-UNBLANKED "
                "for the aborted run, and tells them to scrap prep and re-initialize bench clean.]"
            )
        elif measured_state == "NO" and (blank_state == "WATER" or "WARMING" in power_state):
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Analyst asks about an early setup error (cold lamp or water blank) "
                "before measuring the mAb sample. Supervisor notes no sample was scanned, so no lot release data exists. "
                "Confirms no deviation is needed: document contemporaneous note in raw log, wait for 20-min stabilization, "
                "re-zero with Formulation Buffer, and proceed.]"
            )
        elif measured_state == "YES" and not st.session_state.last_reading_over_range:
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Analyst asks for review of completed measurements. "
                "Supervisor inspects bench, confirms duplicate readings (~0.78 AU) are well within the "
                "0.200–1.200 AU linear range, and tells them to proceed to calculations per Section 6.4.]"
            )
        elif power_state.startswith("READY") and blank_state == "BUFFER" and measured_state == "NO":
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Analyst asks for a mid-assay sanity check. Supervisor confirms "
                "lamp is stabilized and buffer blank is stored. Setup is verified; instructs them to measure samples.]"
            )
        else:
            actual_api_prompt = (
                "Call Supervisor [DIRECTOR NOTE: Supervisor steps into bay to check progress. Provides brief, "
                "realistic procedural feedback based on active bench state without doing the work.]"
            )

    # =========================================================================
    # 2. INSTRUMENT MEASUREMENT ROUTING (DETERMINISTIC OPTICAL ENGINE)
    # =========================================================================
    elif any(k in u_lower for k in ["measure", "scan"]) and "blank" not in u_lower:
        if "OFF" in power_state:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: HARD GATE REFUSAL: Instrument is POWER OFF. "
                f"State that the optical bench is unenergized and refuse measurement.]"
            )
        elif prep_state == "NONE":
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: HARD GATE REFUSAL: No sample prepared. "
                f"State that no sample has been transferred into the cuvette and refuse measurement.]"
            )
        else:
            is_cold = "WARMING" in power_state
            jitter = 0.045 if is_cold else 0.000

            if prep_state == "NEAT":
                rep1_280, rep1_320 = 2.842, 0.084
                rep2_280, rep2_320 = round(2.855 + jitter, 3), 0.081
                st.session_state.last_reading_over_range = True
            elif prep_state == "DILUTED_WATER":
                base_offset = -0.075 if blank_state == "BUFFER" else 0.000
                rep1_280, rep1_320 = round(0.781 + base_offset, 3), 0.006
                rep2_280, rep2_320 = round(0.778 + base_offset + jitter, 3), 0.005
                st.session_state.last_reading_over_range = False
            else:  # DILUTED_BUFFER or standard diluted
                base_offset = 0.185 if blank_state == "NONE" else (0.075 if blank_state == "WATER" else 0.000)
                rep1_280, rep1_320 = round(0.781 + base_offset, 3), 0.006
                rep2_280, rep2_320 = round(0.778 + base_offset + jitter, 3), 0.005
                st.session_state.last_reading_over_range = False

            st.session_state.last_readings = {
                "rep1_280": rep1_280,
                "rep1_320": rep1_320,
                "rep2_280": rep2_280,
                "rep2_320": rep2_320,
            }

            blank_header = (
                "[Baseline Zero Stored: Formulation Buffer | A280 = 0.000 AU | A320 = 0.000 AU]\n"
                if blank_state == "BUFFER"
                else ""
            )
            trace_block = (
                f"{blank_header}"
                f"Prep A (Rep 1): A280 = {rep1_280:.3f} AU | A320 = {rep1_320:.3f} AU\n"
                f"Prep B (Rep 2): A280 = {rep2_280:.3f} AU | A320 = {rep2_320:.3f} AU"
            )

            actual_api_prompt = (
                f"Measure sample preparations [DIRECTOR NOTE: Transcribe the following raw spectrophotometer "
                f"data trace verbatim into your response, update STATE to Measured: YES, and do not provide "
                f"unsolicited advice or recommended next steps:\n"
                f"```\n{trace_block}\n```]"
            )

    # =========================================================================
    # 3. CALCULATION & DISPOSITION ROUTING (DETERMINISTIC MATH ENGINE)
    # =========================================================================
    elif any(k in u_lower for k in ["calc", "disposition"]):
        if measured_state != "YES":
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Pre-Calculation Gate: No analytical data collected (Measured: NO). "
                f"Refuse calculation and state that samples must be measured first.]"
            )
        else:
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", user_input)
            if len(nums) < 4:
                actual_api_prompt = (
                    f"{user_input} [DIRECTOR NOTE: The analyst is initiating calculations. "
                    f"Prompt the analyst to submit the four required parameters: "
                    f"1) Replicate 1 Corrected Absorbance (A280 - A320), "
                    f"2) Replicate 2 Corrected Absorbance (A280 - A320), "
                    f"3) Dilution Factor, and 4) Extinction Coefficient cited in TM-0428. "
                    f"Use dummy illustrative values like 0.520 and 0.518 in your format prompt "
                    f"so the assay values are not revealed. Do not perform the calculation yet.]"
                )
            else:
                r = st.session_state.last_readings or {
                    "rep1_280": 0.781,
                    "rep1_320": 0.006,
                    "rep2_280": 0.778,
                    "rep2_320": 0.005,
                }
                exp_a1 = round(r["rep1_280"] - r["rep1_320"], 3)
                exp_a2 = round(r["rep2_280"] - r["rep2_320"], 3)

                val_a1 = float(nums[0])
                val_a2 = float(nums[1])
                val_df = float(nums[2])
                val_coeff = float(nums[3])

                # Check if entries reconcile with raw bench trace and method parameters
                diff_a1 = abs(val_a1 - exp_a1)
                diff_a2 = abs(val_a2 - exp_a2)
                diff_coeff = abs(val_coeff - 1.45)

                if diff_a1 > 0.002 or diff_a2 > 0.002 or diff_coeff > 0.01:
                    actual_api_prompt = (
                        f"{user_input} [DIRECTOR NOTE: Technical Review Check: Submitted values do not reconcile with "
                        f"raw data trace or TM-0428 §6.3. Expected A_corr1 = {exp_a1:.3f}, Expected A_corr2 = {exp_a2:.3f}, "
                        f"Extinction Coefficient = 1.45. Reject submission for arithmetic/entry correction. "
                        f"This is an arithmetic kickback, not a method deviation.]"
                    )
                else:
                    # Deterministic mathematical calculation in Python floats
                    mean_a = round((val_a1 + val_a2) / 2.0, 3)
                    conc1 = (val_a1 * val_df) / val_coeff
                    conc2 = (val_a2 * val_df) / val_coeff
                    mean_conc = (mean_a * val_df) / val_coeff
                    pct_diff = (abs(conc1 - conc2) / mean_conc) * 100.0

                    # Evaluate procedural compliance
                    is_ready = power_state.startswith("READY")
                    is_buffer_blank = blank_state == "BUFFER"
                    is_buffer_prep = prep_state == "DILUTED_BUFFER"
                    is_in_range = 0.200 <= mean_a <= 1.200
                    is_precise = pct_diff <= 2.0

                    if is_ready and is_buffer_blank and is_buffer_prep and is_in_range and is_precise:
                        summary_block = (
                            "================================================================================\n"
                            "ANALYTICAL TESTING SUMMARY: PASS\n"
                            "Status: ASSAY COMPLIANT & RELEASED\n"
                            "--------------------------------------------------------------------------------\n"
                            f"- Replicate 1 Corrected Absorbance (A_corr1): {val_a1:.3f} AU\n"
                            f"- Replicate 2 Corrected Absorbance (A_corr2): {val_a2:.3f} AU\n"
                            f"- Mean Corrected Absorbance (Mean A_corr): {mean_a:.3f} AU (Calculated)\n"
                            f"- Dilution Factor Applied: {int(val_df)}\n"
                            "- Extinction Coefficient Used: 1.45 mL / (mg · cm)\n"
                            f"- Reported Protein Concentration (Mean_Conc): {mean_conc:.1f} mg/mL (Target: 50–60 mg/mL)\n"
                            f"- Duplicate % Difference: {pct_diff:.2f}% (Acceptance Criteria: <= 2.0%)\n\n"
                            "QC Review: Method TM-0428 executed in full compliance. Lamp thermal stabilization confirmed, "
                            "matrix-matched Formulation Buffer blank verified, duplicate preparations within validated "
                            "linear range (0.200–1.200 AU), replicate precision met.\n"
                            "================================================================================"
                        )
                    else:
                        findings = []
                        if not is_ready:
                            findings.append(
                                "Section 4.2 Non-Compliance: Spectrophotometer operated prior to 20-minute thermal "
                                "stabilization; baseline thermal drift exceeded tolerance."
                            )
                        if blank_state == "NONE":
                            findings.append(
                                "Section 5.1 Non-Compliance: Sample scanned without valid baseline zero. TM-0428 requires "
                                "baseline zero against Formulation Buffer; raw data record lacks blank entry, resulting in "
                                f"uncompensated optical background and gross positive matrix bias (~{mean_conc:.1f} mg/mL vs 50–60 mg/mL target)."
                            )
                        elif blank_state == "WATER":
                            findings.append(
                                "Section 5.1 Non-Compliance: Baseline zero performed against Milli-Q Water instead of "
                                "Formulation Buffer, introducing matrix excipient baseline bias."
                            )
                        if prep_state == "DILUTED_WATER":
                            findings.append(
                                "Section 5.2 Non-Compliance: Sample prepared using unapproved diluent (Milli-Q Water "
                                "instead of Formulation Buffer)."
                            )
                        if mean_a > 1.200 or prep_state == "NEAT":
                            findings.append(
                                f"Section 5.3 Non-Compliance: Sample absorbance reported at {mean_a:.3f} AU, exceeding "
                                "validated 1.200 AU linear boundary without utilizing approved re-preparation provision."
                            )
                        elif mean_a < 0.200:
                            findings.append(
                                f"Section 5.3 Non-Compliance: Sample absorbance reported at {mean_a:.3f} AU, falling "
                                "below validated 0.200 AU linear range."
                            )
                        if not is_precise:
                            findings.append(
                                f"Section 6.4 Non-Compliance: Duplicate % Difference ({pct_diff:.2f}%) exceeded "
                                "maximum allowable threshold of <= 2.0%."
                            )

                        findings_str = "\n- ".join(findings)
                        summary_block = (
                            "================================================================================\n"
                            "ANALYTICAL TESTING SUMMARY: DEVIATION\n"
                            "Status: PACKET REJECTED AT TECHNICAL REVIEW (DEV-QC-8412)\n"
                            "Assay Result: INVALIDATED PRIOR TO RELEASE\n"
                            "--------------------------------------------------------------------------------\n"
                            f"- Replicate 1 Corrected Absorbance: {val_a1:.3f} AU\n"
                            f"- Replicate 2 Corrected Absorbance: {val_a2:.3f} AU\n"
                            f"- Calculated Mean Concentration (Mean_Conc): {mean_conc:.1f} mg/mL (Analytical Artifact)\n"
                            f"- Duplicate % Difference: {pct_diff:.2f}%\n\n"
                            "QC Review Findings:\n"
                            f"- {findings_str}\n\n"
                            "Technical Review Disposition: Testing packet rejected during second-person review. "
                            "Data invalidated; lot cannot be dispositioned on this packet. Corrective Action: Open lab "
                            "deviation for incomplete method execution, void run, and retest.\n"
                            "================================================================================"
                        )

                    actual_api_prompt = (
                        f"{user_input} [DIRECTOR NOTE: Technical review complete. Output the following "
                        f"analytical summary block verbatim in your response with no alterations:\n"
                        f"{summary_block}]"
                    )

    # Sliding window: keep only the most recent 6 messages to protect token budget
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
                max_tokens=650,
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
            error_fallback = (
                "⚠️ **[INSTRUMENT BUS CONGESTION]**\n\n"
                "LIMS audit logging buffer is temporarily saturated. Standby a moment and retry."
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

        # Strip accidental prompt echoing
        if clean_text.lower().startswith(user_input.lower()):
            clean_text = clean_text[len(user_input):].lstrip(":\n -")

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

    b = st.session_state.state_badge
    telemetry_table = f"""
| Instrument Parameter | Status |
| :--- | :--- |
| **Lamp Status** | `{b['Power']}` |
| **Blank Baseline** | `{b['Blank']}` |
| **Sample Prep** | `{b['Prep']}` |
| **Measurement Run** | `{b['Measured']}` |
"""
    st.markdown(telemetry_table)

    st.divider()

    st.subheader("📋 Protein Conc. by UV: TM-0428 (Rev. 04)")
    tab1, tab2, tab3, tab4 = st.tabs(["Page 1", "Page 2", "Page 3", "Page 4"])

    with tab1:
        st.caption("Sections 1–3: Purpose, Scope & Responsibility")
        st.markdown(
            """
            * **1.0 Objective:** Define procedural requirements for determining mAb protein concentration at 280 nm.
            * **2.0 Scope:** Applies to release and in-process testing of mAb Drug Substance (Lot BDS-552).
            * **3.0 Responsibility:** QC analysts execute testing per method; QC management reviews and disposes data.
            """
        )
    with tab2:
        st.caption("Section 4: Equipment & System Setup")
        st.markdown(
            """
            * **4.1 Instrument:** Dual-beam UV-Vis spectrophotometer with 10 mm quartz micro-cuvette.
            * **4.2 Lamp Warm-Up:** Switch on D2 and W lamps; stabilize **>= 20 minutes** before blanking or sample analysis.
            * **4.3 Optical Inspection:** Inspect quartz optical faces for cleanliness, scratches, or chips.
            """
        )
    with tab3:
        st.caption("Section 5: Solution Preparation & Blanking")
        st.markdown(
            """
            * **5.1 Baseline Zero (Blank):** Zero the spectrophotometer with **Formulation Buffer**.
            * **5.2 Sample Prep:** Prepare duplicate independent dilutions using a **1:100 dilution factor (DF)** with Formulation Buffer as diluent.
            * **5.3 Linear Range & Re-Prep:** Validated linear range is **0.200 to 1.200 AU** at 280 nm. If reading exceeds 1.200 AU, re-preparation with adjusted dilution is permitted and does NOT constitute a deviation.
            """
        )
    with tab4:
        st.caption("Section 6: Testing, Calculations & Acceptance Criteria")
        st.markdown(
            """
            * **6.1 Testing:** Scan samples in duplicate at 280 nm and 320 nm.
            * **6.2 Correction:** Calculate corrected absorbance: `A_corr = A280 - A320`.
            * **6.3 Quantitation:** Calculate concentration: `(A_corr / 1.45) * DF`.
            * **6.4 Precision:** % Difference between duplicates must be **<= 2.0%**.
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
        st.session_state.last_readings = None
        st.rerun()

# --- MAIN TERMINAL WORKSPACE ---
st.title("Biopharma QC Analytical Terminal")
st.caption("Simulated Release Testing Platform | UV-Vis Spectrophotometer Workstation 4")
st.caption("💡 *Analytical Tip: Allow 10–15 seconds between bench actions to observe instrument stabilization and maintain audit trail synchronization.*")

# Load Initial Opener
if len(st.session_state.display_messages) == 0:
    opener = (
        "**[QC SUITE 2 — WORKSTATION 4]**\n\n"
        "You are logged in at Bench 4 to perform release testing for **mAb Drug Substance (Lot BDS-552)**.\n\n"
        "Target concentration is approximately **50–60 mg/mL**.\n\n"
        "**Current Bench Status:**\n"
        "* **Spectrophotometer:** POWER OFF (Cold, 0 min).\n"
        "* **Equipment:** Calibrated pipettors (P20, P200, P1000).\n"
        "* **Optics & Consumables:** 10 mm quartz micro-cuvette, barrier tips, microcentrifuge tubes, lint-free wipes.\n"
        "* **Sample:** 1x 2.0 mL vial of mAb Drug Substance (Lot BDS-552).\n"
        "* **Reagents:** 1x 500 mL bottle of Formulation Buffer; 1x Milli-Q Water wash station.\n"
        "* **Documentation:** Test Method binder **TM-0428** available in the sidebar document deck."
    )
    initial_state = "STATE: [Power: OFF (0 min) | Blank: NONE | Prep: NONE | Measured: NO | In-Range: UNTESTED | SupCalls: 0]"
    st.session_state.display_messages.append(
        {"role": "assistant", "content": opener, "state": initial_state}
    )
    st.session_state.api_messages.append(
        {"role": "assistant", "content": f"{initial_state}\n\n{opener}"}
    )

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("state"):
            with st.expander("📟 Audit Trail Telemetry", expanded=False):
                st.code(msg["state"], language="text")

# Determine if the analytical session has reached a terminal conclusion
last_assistant_content = ""
for m in reversed(st.session_state.display_messages):
    if m["role"] == "assistant":
        last_assistant_content = m["content"]
        break

is_terminal = (
    "ANALYTICAL TESTING SUMMARY:" in last_assistant_content
    or "DEV-QC-UNBLANKED" in last_assistant_content
    or "DATA INTEGRITY FAILURE" in last_assistant_content
)

# --- CONTEXT-AWARE ANALYTICAL OPERATIONS ---
st.markdown("##### Standard Analytical Operations")

active_command = None

if is_terminal:
    if st.button("🔄 Reset Workstation for Next Analytical Session", use_container_width=True):
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
        st.session_state.last_readings = None
        st.rerun()
else:
    power_state = st.session_state.state_badge["Power"].upper()
    measured_state = st.session_state.state_badge["Measured"].upper()
    is_cold = "OFF" in power_state
    is_warming = "WARMING" in power_state

    # Warming Layout (3 top, 2 bottom)
    if is_warming:
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        r2_c1, r2_c2 = st.columns(2)

        with r1_c1:
            if st.button("🧪 Prepare Sample", use_container_width=True):
                active_command = "Prepare Sample"
        with r1_c2:
            if st.button("⏳ Wait", use_container_width=True):
                active_command = "Wait 20 minutes for lamp stabilization"
        with r1_c3:
            if st.button("🧪 Blank Instrument", use_container_width=True):
                active_command = "Blank with Formulation Buffer"

        with r2_c1:
            if measured_state == "YES":
                if st.button("🧮 Calculate Results", use_container_width=True):
                    active_command = "Calculate concentration per TM-0428"
            else:
                if st.button("📊 Measure Samples", use_container_width=True):
                    active_command = "Measure sample preparations"
        with r2_c2:
            if st.button("📞 Call Supervisor", use_container_width=True):
                active_command = "Call Supervisor"

    # Stabilized (>= 20 min) or Cold Layout (Clean 2x2 grid)
    else:
        r1_c1, r1_c2 = st.columns(2)
        r2_c1, r2_c2 = st.columns(2)

        with r1_c1:
            if is_cold:
                if st.button("⚡ Power ON Instrument", use_container_width=True):
                    active_command = "Power ON the Spectrophotometer"
            else:
                if st.button("🧪 Prepare Sample", use_container_width=True):
                    active_command = "Prepare Sample"

        with r1_c2:
            if st.button("🧪 Blank Instrument", use_container_width=True):
                active_command = "Blank with Formulation Buffer"

        with r2_c1:
            if measured_state == "YES":
                if st.button("🧮 Calculate Results", use_container_width=True):
                    active_command = "Calculate concentration per TM-0428"
            else:
                if st.button("📊 Measure Samples", use_container_width=True):
                    active_command = "Measure sample preparations"
        with r2_c2:
            if st.button("📞 Call Supervisor", use_container_width=True):
                active_command = "Call Supervisor"

user_text = st.chat_input(
    "Enter command or calculation data (e.g., 'Prepare sample', 'Calculate', or custom actions)...",
    disabled=is_terminal,
)
if user_text:
    active_command = user_text

if active_command:
    with st.spinner("Processing workstation operation..."):
        run_llm_turn(active_command)
    st.rerun()
