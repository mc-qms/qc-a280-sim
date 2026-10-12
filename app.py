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

prompt_dir = os.path.dirname(__file__)
try:
    with open(os.path.join(prompt_dir, "narrator_prompt.txt"), "r", encoding="utf-8") as f:
        NARRATOR_SPEC = f.read()
    with open(os.path.join(prompt_dir, "supervisor_prompt.txt"), "r", encoding="utf-8") as f:
        SUPERVISOR_SPEC = f.read()
except FileNotFoundError as e:
    st.error(f"Missing prompt specification file: {e}. Ensure both narrator_prompt.txt and supervisor_prompt.txt exist.")
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
if "selecting_blank" not in st.session_state:
    st.session_state.selecting_blank = False
if "cuvette_contents" not in st.session_state:
    st.session_state.cuvette_contents = "EMPTY"
if "awaiting_prep" not in st.session_state:
    st.session_state.awaiting_prep = False


def extract_state_token(raw_text: str) -> str:
    """Extracts raw STATE line if present."""
    match = re.search(r"\*{0,2}STATE:\s*\[(.*?)\]\*{0,2}", raw_text, re.IGNORECASE)
    return match.group(0).strip() if match else ""


def update_state_from_text(raw_text: str):
    """Monitors raw text for optical flags without clobbering Python's state machine."""
    if "2.842" in raw_text or "2.855" in raw_text:
        st.session_state.last_reading_over_range = True
    elif "0.781" in raw_text or "0.706" in raw_text or "0.966" in raw_text or "0.856" in raw_text:
        st.session_state.last_reading_over_range = False


def clean_state_token(text: str) -> str:
    """Strips the STATE token line from dialogue text."""
    cleaned = re.sub(r"\*{0,2}STATE:\s*\[.*?\]\*{0,2}\n*", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def run_llm_turn(user_input: str, is_supervisor_action: bool = False):
    """Executes turn with dynamic persona routing, deterministic physics, and direct Python disposition."""
    actual_api_prompt = user_input
    u_lower = user_input.lower().strip()

    # Persona Selection: Flag-first with bounded free-text fallback
    supervisor_keywords = ["supervisor", "ask lead", "page lead", "call lead", "elevate"]
    is_supervisor = is_supervisor_action or any(k in u_lower for k in supervisor_keywords)
    active_system_spec = SUPERVISOR_SPEC if is_supervisor else NARRATOR_SPEC

    # Track supervisor calls before freezing state string
    if is_supervisor:
        try:
            curr_calls = int(st.session_state.state_badge.get("SupCalls", 0))
            st.session_state.state_badge["SupCalls"] = str(curr_calls + 1)
        except ValueError:
            st.session_state.state_badge["SupCalls"] = "1"

    power_state = st.session_state.state_badge["Power"].upper()
    blank_state = st.session_state.state_badge["Blank"].upper()
    prep_state = st.session_state.state_badge["Prep"].upper()
    measured_state = st.session_state.state_badge["Measured"].upper()

    is_measurement_turn = False
    active_trace_block = None

    current_frozen_state = (
        f"STATE: [Power: {st.session_state.state_badge['Power']} | "
        f"Blank: {st.session_state.state_badge['Blank']} | "
        f"Prep: {st.session_state.state_badge['Prep']} | "
        f"Measured: {st.session_state.state_badge['Measured']} | "
        f"In-Range: {st.session_state.state_badge['In-Range']} | "
        f"SupCalls: {st.session_state.state_badge['SupCalls']}]"
    )

    # =========================================================================
    # PENDING-INTENT HANDLER: SAMPLE PREPARATION SLOT FILLING
    # =========================================================================
    is_global_escape = any(k in u_lower for k in [
        "supervisor", "lead", "power", "blank", "wait", "method", "page", "reset", "calc"
    ])
    if st.session_state.awaiting_prep:
        if is_global_escape:
            st.session_state.awaiting_prep = False
            is_prep_submission = False
        else:
            st.session_state.awaiting_prep = False
            is_prep_submission = True
    else:
        is_prep_submission = False

    # =========================================================================
    # 1. SUPERVISOR ROUTING (SUPERVISOR SPEC ACTIVE)
    # =========================================================================
    if is_supervisor:
        # Hierarchy 1: Bench unpowered/cold
        if power_state.startswith("OFF"):
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Bench is idle and unpowered. Supervisor steps into the bay, glances at the dark power switch, "
                "and speaks candidly: 'The spectrophotometer isn't even powered on. Switch it on, let the lamps "
                "stabilize per TM-0428 Section 4, and review the method before calling me.' "
                "The supervisor turns and heads back to the office.]"
            )
        # Hierarchy 2: COMPOUND FAILURE (Fatal First: Invalid Blank + Neat Sample)
        elif measured_state == "YES" and blank_state != "BUFFER" and (st.session_state.last_reading_over_range or prep_state == "NEAT"):
            blank_desc = (
                "complete omission of a baseline zero (§5.1)"
                if blank_state == "NONE"
                else "using Milli-Q Water instead of Formulation Buffer (§5.1)"
            )
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "COMPOUND FAILURE ROUTE — EXACTLY 4 TO 5 SENTENCES: "
                "Sentence 1 (Observation): Note that raw data on the PC screen shows duplicate readings saturated at 2.84 AU with an invalid baseline. "
                f"Sentence 2 (Fatal Error): Call out FATAL-FIRST that the {blank_desc} completely invalidates the baseline before any sample can be acquired. "
                "Sentence 3 (Secondary Error): Call out the undiluted neat sample preparation (§5.2) and firmly clarify that Section 5.3 re-preparation only protects compliant dilutions that overshoot, NOT runs where foundational baseline steps were skipped. "
                "Sentence 4 (Directive): Since raw GMP data was acquired under invalid conditions, the run is voided; log in-process non-conformance DEV-QC-SETUP-FAIL, instruct them to scrap all tubes and clean the cuvette, and restart from scratch. "
                "Sentence 5 (Departure): Exit the bay immediately back to the office.]"
            )
        # Hierarchy 3: Over-Range ONLY (Properly Blanked with Formulation Buffer, §5.3 Concession Valid)
        elif st.session_state.last_reading_over_range and measured_state == "YES" and blank_state == "BUFFER":
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Bench display reads 2.842 AU, but baseline zero with Formulation Buffer was stored properly. "
                "Supervisor walks into the bay, looks at the PC monitor, and reminds analyst of TM-0428 Section 5.3: "
                "re-preparation at an adjusted dilution is explicitly permitted on the spot and does NOT require raising an unplanned deviation. "
                "Advises them to re-prep and re-run, then exits the bay.]"
            )
        # Hierarchy 4: Invalid Blank ONLY (Diluted sample, but Blank is NONE or WATER)
        elif measured_state == "YES" and blank_state in ["NONE", "WATER"]:
            blank_desc = "without blanking (Blank: NONE)" if blank_state == "NONE" else "with Milli-Q Water instead of Formulation Buffer"
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                f"Analyst measured diluted sample {blank_desc}. Supervisor reviews raw scan data on the PC monitor, spots "
                "the non-compliant baseline, and points out the matrix artifact. Commends analyst for stopping before calculating "
                "and signing off, logs in-process non-conformance DEV-QC-SETUP-FAIL for the aborted run, tells them to scrap prep "
                "and re-initialize bench clean, and exits the bay.]"
            )
        # Hierarchy 5: Pre-measurement Water Blank caught BEFORE scanning sample
        elif measured_state == "NO" and blank_state == "WATER":
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Supervisor steps into the bay, glances at the PC monitor, and notices the stored baseline says Milli-Q Water "
                "instead of Formulation Buffer. Notes that since no mAb sample was scanned yet, no lot release data exists and "
                "no deviation is required. Instructs analyst to document a contemporaneous note in the raw log, re-zero with "
                "Formulation Buffer once the lamp is ready, and proceed. The supervisor exits the bay.]"
            )
        # Hierarchy 6: Routine Warmup
        elif "WARMING" in power_state and measured_state == "NO":
            if prep_state == "DILUTED_BUFFER" and blank_state == "BUFFER":
                actual_api_prompt = (
                    f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                    "ROUTINE WARMUP ROUTE — STRICT 2 TO 3 SENTENCES TOTAL: "
                    "Sentence 1 (Observation): Point out the PC monitor shows the lamps are still warming up even though preps and buffer blank are ready. "
                    "Sentence 2 (Directive): Tell the analyst directly with mild exasperation that the setup is fine, but the optical bench physically needs 20 minutes to stabilize—stop paging the lead desk just to watch a timer tick. "
                    "Sentence 3 (Departure): Turn and exit the bay back to the office.]"
                )
            elif prep_state != "NONE" or blank_state != "NONE":
                actual_api_prompt = (
                    f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                    "ROUTINE WARMUP ROUTE — STRICT 2 TO 3 SENTENCES TOTAL: "
                    "Sentence 1 (Observation): Note the lamps are still warming on the monitor while preparations are underway. "
                    "Sentence 2 (Directive): Remind them the instrument cannot scan until the 20-minute lamp stabilization cycle finishes; tell them to finish bench prep and wait without paging the desk. "
                    "Sentence 3 (Departure): Turn and exit the bay back to the office.]"
                )
            else:
                actual_api_prompt = (
                    f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                    "IDLE WARMUP ROUTE — STRICT 2 TO 3 SENTENCES TOTAL: "
                    "Sentence 1 (Observation): Note the monitor shows LAMP WARMING and the bench is completely idle. "
                    "Sentence 2 (Directive): Tell them candidly not to page the lead desk just to watch a timer—use this time to read TM-0428 or set up pipettes and tubes. "
                    "Sentence 3 (Departure): Turn and exit the bay back to the office.]"
                )
        # Hierarchy 7: Completed in-range measurements (~0.78 AU)
        elif measured_state == "YES" and not st.session_state.last_reading_over_range:
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Analyst asks for review of completed measurements. Supervisor inspects PC monitor, confirms duplicate readings "
                "(~0.78 AU) are well within the 0.200–1.200 AU linear range, tells them to proceed to calculations per Section 6.4, "
                "and exits the bay.]"
            )
        # Hierarchy 8: Mid-assay sanity check (ready + buffer blank + measured NO)
        elif power_state.startswith("READY") and blank_state == "BUFFER" and measured_state == "NO":
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Analyst asks for a mid-assay sanity check. Supervisor confirms lamp is stabilized and buffer blank is stored. "
                "Setup is verified; instructs them to measure samples, and exits the bay.]"
            )
        else:
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Supervisor steps into bay to check progress. Provides brief, realistic procedural feedback based on active "
                "bench state without doing the work, then exits the bay.]"
            )

    # =========================================================================
    # 2. INSTRUMENT POWER ROUTING (NARRATOR SPEC ACTIVE)
    # =========================================================================
    elif any(k in u_lower for k in ["power on", "switch on", "turn on"]):
        st.session_state.state_badge["Power"] = "WARMING (2 min)"
        actual_api_prompt = (
            f"{user_input} [DIRECTOR NOTE: Analyst flips the rocker switch on the spectrophotometer chassis. "
            "Describe the physical hardware response: cooling fan engages with a quiet whir, internal stepper motors "
            "click as monochromator and filter wheels complete homing alignment, and front panel status LED illuminates solid green. "
            "(The optical bench is sealed and light-tight; no visible glow escapes the chassis). "
            "On the PC workstation monitor, the UV-Vis software establishes connection and updates its status bar: "
            "'Hardware: Online | Lamps: Ignited | Status: LAMP WARMING (2 min / 20 min required)'. "
            "CRITICAL: DO NOT do subtraction math. Stop immediately after reporting software status.]"
        )

    # =========================================================================
    # 2B. INSTRUMENT WARMUP / WAIT ROUTING (PYTHON STATE OWNERSHIP)
    # =========================================================================
    elif any(k in u_lower for k in ["wait", "warm up", "warmup", "stabilize"]):
        if "OFF" in power_state:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: The spectrophotometer is powered off. "
                "The lamps cannot warm up while the instrument is unenergized. "
                "State that the main power switch remains OFF. Do not advise next steps.]"
            )
        else:
            st.session_state.state_badge["Power"] = "READY (20 min)"
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: The analyst allows 20 minutes for lamp stabilization. "
                "On the PC workstation monitor, the UV-Vis software status updates to: "
                "'Hardware: Online | Lamps: Stabilized | Status: LAMP READY (>=20 min)'. "
                "Stop immediately after reporting software status without providing unsolicited advice.]"
            )

    # =========================================================================
    # 3. BLANKING ROUTING (NARRATOR SPEC ACTIVE)
    # =========================================================================
    elif "blank" in u_lower and not any(k in u_lower for k in ["calc", "supervisor"]):
        reagent = "Formulation Buffer" if "buffer" in u_lower else ("Milli-Q Water" if "water" in u_lower else "unknown")
        
        rinse_narrative = ""
        if st.session_state.cuvette_contents != "EMPTY" and st.session_state.cuvette_contents != reagent:
            rinse_narrative = (
                f"Noticing the micro-cuvette previously contained {st.session_state.cuvette_contents}, the analyst takes it "
                "to the wash station, aspirates the old liquid using the vacuum waste tip, flushes the cell with Milli-Q water "
                "followed by reagent-grade ethanol, and dries it thoroughly. "
            )

        if "OFF" in power_state:
            st.session_state.cuvette_contents = reagent if reagent != "unknown" else st.session_state.cuvette_contents
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: HARD GATE REFUSAL: Instrument is POWER OFF. "
                f"{rinse_narrative}"
                f"The analyst pipettes {reagent if reagent != 'unknown' else 'solution'} into the quartz micro-cuvette using a P1000 pipettor, "
                "wipes optical faces with a lint-free wipe, and seats it in the cell holder. "
                "However, the spectrophotometer power switch is OFF and the PC software shows 'Hardware: Offline'. "
                "State that the optical bench is unenergized; no baseline can be acquired or stored. "
                "CRITICAL: DO NOT cite TM-0428 warm-up requirements. DO NOT advise next steps. Stop immediately.]"
            )
        elif reagent == "unknown":
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Unqualified blank command. Prompt the analyst neutrally: "
                "'Which solution do you pipette into the cuvette to zero the instrument (Formulation Buffer or Milli-Q Water)?' "
                "Do NOT reveal which reagent is approved in TM-0428.]"
            )
        else:
            st.session_state.cuvette_contents = reagent
            chosen_blank_token = "BUFFER" if "buffer" in u_lower else "WATER"
            st.session_state.state_badge["Blank"] = chosen_blank_token
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: "
                f"{rinse_narrative}"
                f"The analyst uses a calibrated P1000 pipettor to transfer {reagent} "
                "into the 10 mm quartz micro-cuvette, inspects optical windows, wipes them with a lint-free wipe, "
                "and closes the sample compartment lid. In the workstation software, they click 'Zero / Baseline'. "
                "The optical bench takes a baseline read and stores it. "
                f"Update STATE to Blank: {chosen_blank_token}. "
                "CRITICAL: Report ONLY physical pipetting/zeroing action and PC software confirmation. Stop immediately.]"
            )

    # =========================================================================
    # 4. INSTRUMENT MEASUREMENT ROUTING (NOW ABOVE PREP)
    # =========================================================================
    elif any(k in u_lower for k in ["measure", "scan"]) and "blank" not in u_lower:
        if "OFF" in power_state:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: HARD GATE REFUSAL: Instrument is POWER OFF. "
                "State that the power switch is OFF and the PC software shows 'Hardware: Offline'. Refuse measurement.]"
            )
        elif prep_state == "NONE":
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: HARD GATE REFUSAL: Empty sample cell. "
                "Describe inspecting the cell compartment: no sample preparation has been pipetted into the cuvette. "
                "The software remains idle. Refuse scan. Stop immediately after reporting cuvette state.]"
            )
        else:
            is_measurement_turn = True
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
            else:
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

            blank_header = ""
            if blank_state == "BUFFER":
                blank_header = "[Baseline Zero Stored: Formulation Buffer | A280 = 0.000 AU | A320 = 0.000 AU]\n"
            elif blank_state == "WATER":
                blank_header = "[Baseline Zero Stored: Milli-Q Water | A280 = 0.000 AU | A320 = 0.000 AU]\n"

            trace_block = (
                f"{blank_header}"
                f"Prep A (Rep 1): A280 = {rep1_280:.3f} AU | A320 = {rep1_320:.3f} AU\n"
                f"Prep B (Rep 2): A280 = {rep2_280:.3f} AU | A320 = {rep2_320:.3f} AU"
            )
            active_trace_block = trace_block

            cuvette_wash_narrative = ""
            if st.session_state.cuvette_contents not in ["EMPTY", "mAb Sample Preparation"]:
                cuvette_wash_narrative = (
                    f"The analyst takes the cuvette to the wash station to aspirate the remaining {st.session_state.cuvette_contents}, "
                    "flushes the cell with Milli-Q water followed by ethanol, dries it with vacuum, and pipettes the duplicate preparation "
                    "into the 10 mm quartz cuvette. "
                )

            st.session_state.cuvette_contents = "mAb Sample Preparation"
            st.session_state.state_badge["Measured"] = "YES"

            actual_api_prompt = (
                f"Measure sample preparations [DIRECTOR NOTE: {cuvette_wash_narrative}"
                "Describe the mechanical sample scan acquisition at the bench and data station. "
                "Update STATE to Measured: YES. DO NOT generate absorbance numbers or code blocks in your text—"
                "the raw data station trace is logged automatically by the system bus.]"
            )

    # =========================================================================
    # 5. SAMPLE PREPARATION ROUTING (NOW BELOW MEASUREMENT)
    # =========================================================================
    elif (is_prep_submission or any(k in u_lower for k in ["prepare", "prep", "dilute"])) and not any(k in u_lower for k in ["measure", "scan", "blank"]):
        is_neat = any(k in u_lower for k in ["neat", "undiluted", "direct"])
        has_buffer = any(k in u_lower for k in ["buffer", "formulation"])
        has_water = any(k in u_lower for k in ["water", "milli-q"])

        is_serial_100 = "serial" in u_lower and "100" in u_lower and "900" in u_lower
        is_direct_100 = any(k in u_lower for k in [
            "1:100", "1/100", "1 to 100", "100-fold", "100x",
            "100 df", "df 100", "df of 100", "dilution factor of 100", "dilution factor 100"
        ])

        if is_neat:
            st.session_state.state_badge["Prep"] = "NEAT"
            st.session_state.state_badge["Measured"] = "NO"
            st.session_state.last_reading_over_range = True
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: The analyst pipettes neat, undiluted mAb Drug Substance directly into "
                "duplicate microcentrifuge tubes. Confirm duplicate neat preparations (Prep A and Prep B). "
                "Update STATE to Prep: NEAT and Measured: NO. Stop immediately without advice.]"
            )
        elif (is_serial_100 or is_direct_100) and (has_buffer or not has_water):
            st.session_state.state_badge["Prep"] = "DILUTED_BUFFER"
            st.session_state.state_badge["Measured"] = "NO"
            st.session_state.last_reading_over_range = False
            math_explanation = (
                "a two-step 1:10 serial dilution (100 µL into 900 µL Formulation Buffer, followed by 100 µL into 900 µL Formulation Buffer), "
                "yielding a validated final dilution factor of 1:100 (DF = 100)"
                if is_serial_100
                else "a 1:100 dilution using Formulation Buffer as the diluent (DF = 100)"
            )
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: The analyst executes duplicate sample preparations using {math_explanation}. "
                "CRITICAL ARITHMETIC ENFORCEMENT: Confirm duplicate preparations (Prep A and Prep B) at 1:100 dilution factor. "
                "DO NOT calculate or state 1:1000. Describe pipetting with calibrated pipettors and vortexing tubes. "
                "Update STATE to Prep: DILUTED_BUFFER and Measured: NO. Stop immediately without advice.]"
            )
        elif (is_serial_100 or is_direct_100) and has_water:
            st.session_state.state_badge["Prep"] = "DILUTED_WATER"
            st.session_state.state_badge["Measured"] = "NO"
            st.session_state.last_reading_over_range = False
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: The analyst prepares duplicate samples using a 1:100 dilution factor in Milli-Q Water. "
                "Confirm duplicate preparations (Prep A and Prep B) at 1:100 in water. "
                "Update STATE to Prep: DILUTED_WATER and Measured: NO. Stop immediately without advice.]"
            )
        else:
            st.session_state.awaiting_prep = True
            actual_api_prompt = user_input

    # =========================================================================
    # 6. CALCULATION & DISPOSITION ROUTING (DIRECT PYTHON EMISSION)
    # =========================================================================
    elif any(k in u_lower for k in ["calc", "disposition"]):
        if measured_state != "YES":
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Pre-Calculation Gate: No analytical data collected (Measured: NO). "
                f"Refuse calculation and state that no scan data exists in the active run buffer.]"
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
                    f"so the assay values are not revealed. Do not perform calculation yet.]"
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

                diff_a1 = abs(val_a1 - exp_a1)
                diff_a2 = abs(val_a2 - exp_a2)
                diff_coeff = abs(val_coeff - 1.45)

                if diff_a1 > 0.002 or diff_a2 > 0.002 or diff_coeff > 0.01:
                    actual_api_prompt = (
                        f"{user_input} [DIRECTOR NOTE: Technical Review Check: Submitted values do not reconcile with "
                        f"raw data trace or TM-0428 §6.3. Expected A_corr1 = {exp_a1:.3f}, Expected A_corr2 = {exp_a2:.3f}, "
                        f"Extinction Coefficient = 1.45. Reject submission for arithmetic correction. This is not a deviation.]"
                    )
                else:
                    mean_a = round((val_a1 + val_a2) / 2.0, 3)
                    conc1 = (val_a1 * val_df) / val_coeff
                    conc2 = (val_a2 * val_df) / val_coeff
                    mean_conc = (mean_a * val_df) / val_coeff
                    pct_diff = (abs(conc1 - conc2) / mean_conc) * 100.0

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
                            f"- Replicate 1 Protein Concentration (Conc_1): {conc1:.2f} mg/mL\n"
                            f"- Replicate 2 Protein Concentration (Conc_2): {conc2:.2f} mg/mL\n"
                            f"- Reported Protein Concentration (Mean_Conc): {mean_conc:.2f} mg/mL (Target: 50–60 mg/mL)\n"
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
                                f"uncompensated optical background and gross positive matrix bias (~{mean_conc:.2f} mg/mL vs 50–60 mg/mL target)."
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
                            f"- Replicate 1 Protein Concentration (Conc_1): {conc1:.2f} mg/mL\n"
                            f"- Replicate 2 Protein Concentration (Conc_2): {conc2:.2f} mg/mL\n"
                            f"- Calculated Mean Concentration (Mean_Conc): {mean_conc:.2f} mg/mL (Analytical Artifact)\n"
                            f"- Duplicate % Difference: {pct_diff:.2f}%\n\n"
                            "QC Review Findings:\n"
                            f"- {findings_str}\n\n"
                            "Technical Review Disposition: Testing packet rejected during second-person review. "
                            "Data invalidated; lot cannot be dispositioned on this packet. Corrective Action: Open lab "
                            "deviation for incomplete method execution, void run, and retest.\n"
                            "================================================================================"
                        )

                    state_token = f"STATE: [Power: {power_state} | Blank: {blank_state} | Prep: {prep_state} | Measured: YES | In-Range: {'YES' if is_in_range else 'NO'} | SupCalls: {st.session_state.state_badge['SupCalls']}]"
                    st.session_state.api_messages.append({"role": "user", "content": user_input})
                    st.session_state.api_messages.append(
                        {"role": "assistant", "content": f"{state_token}\n\n{summary_block}"}
                    )
                    st.session_state.display_messages.append({"role": "user", "content": user_input})
                    st.session_state.display_messages.append(
                        {"role": "assistant", "content": summary_block, "state": state_token}
                    )
                    return

    # =========================================================================
    # 7. LLM INFERENCE DISPATCH
    # =========================================================================
    active_history = (
        st.session_state.api_messages[-6:]
        if len(st.session_state.api_messages) > 6
        else st.session_state.api_messages
    )

    payload = (
        [{"role": "system", "content": active_system_spec}]
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
        if not is_supervisor:
            update_state_from_text(raw_response)

        clean_text = clean_state_token(raw_response)

        if clean_text.lower().startswith(user_input.lower()):
            clean_text = clean_text[len(user_input):].lstrip(":\n -")

        # Deterministic Trace Guarantee: Strip any hallucinated code block and append exact Python physics
        if is_measurement_turn and active_trace_block:
            clean_text = re.sub(r"```[\s\S]*?```", "", clean_text).strip()
            clean_text = f"{clean_text}\n\n```text\n{active_trace_block}\n```"

        # SINGLE SOURCE OF TRUTH: Audit trail badge matches Python state exactly
        authoritative_state_token = (
            f"STATE: [Power: {st.session_state.state_badge['Power']} | "
            f"Blank: {st.session_state.state_badge['Blank']} | "
            f"Prep: {st.session_state.state_badge['Prep']} | "
            f"Measured: {st.session_state.state_badge['Measured']} | "
            f"In-Range: {st.session_state.state_badge['In-Range']} | "
            f"SupCalls: {st.session_state.state_badge['SupCalls']}]"
        )

        st.session_state.api_messages.append({"role": "user", "content": actual_api_prompt})
        st.session_state.api_messages.append(
            {"role": "assistant", "content": f"{authoritative_state_token}\n\n{clean_text}"}
        )

        st.session_state.display_messages.append({"role": "user", "content": user_input})
        st.session_state.display_messages.append(
            {"role": "assistant", "content": clean_text, "state": authoritative_state_token}
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
        st.session_state.selecting_blank = False
        st.session_state.cuvette_contents = "EMPTY"
        st.session_state.awaiting_prep = False
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
        "* **Spectrophotometer:** Benchtop dual-beam optical chassis is POWER OFF (Cold, 0 min). Optical bench is sealed and light-tight; front LED is dark.\n"
        "* **Workstation PC:** Data Station online; spectrophotometer control software shows `Hardware: Offline`.\n"
        "* **Equipment:** Calibrated pipettors (P20, P200, P1000) and barrier tips.\n"
        "* **Optics & Consumables:** 10 mm quartz micro-cuvette, microcentrifuge tubes, lint-free wipes.\n"
        "* **Sample:** 1x 2.0 mL vial of mAb Drug Substance (Lot BDS-552).\n"
        "* **Reagents & Wash:** 1x 500 mL bottle of Formulation Buffer; 1x wash station equipped with Milli-Q Water and reagent-grade Ethanol squeeze wash bottles, and a vacuum waste aspirator.\n"
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

last_assistant_content = ""
for m in reversed(st.session_state.display_messages):
    if m["role"] == "assistant":
        last_assistant_content = m["content"]
        break

is_terminal = (
    "ANALYTICAL TESTING SUMMARY:" in last_assistant_content
    or "DEV-QC-" in last_assistant_content
    or "DATA INTEGRITY FAILURE" in last_assistant_content
)

# --- CONTEXT-AWARE ANALYTICAL OPERATIONS ---
st.markdown("##### Standard Analytical Operations")

active_command = None
is_supervisor_click = False

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
        st.session_state.selecting_blank = False
        st.session_state.cuvette_contents = "EMPTY"
        st.session_state.awaiting_prep = False
        st.rerun()

elif st.session_state.selecting_blank:
    st.info("Select reagent to pipette into the 10 mm quartz micro-cuvette to zero the baseline:")
    c1, c2, c3 = st.columns([1, 1, 0.8])
    with c1:
        if st.button("🧪 Formulation Buffer", use_container_width=True):
            st.session_state.selecting_blank = False
            active_command = "Blank with Formulation Buffer"
    with c2:
        if st.button("💧 Milli-Q Water", use_container_width=True):
            st.session_state.selecting_blank = False
            active_command = "Blank with Milli-Q Water"
    with c3:
        if st.button("↩️ Cancel", use_container_width=True):
            st.session_state.selecting_blank = False
            st.rerun()

else:
    power_state = st.session_state.state_badge["Power"].upper()
    measured_state = st.session_state.state_badge["Measured"].upper()
    is_cold = "OFF" in power_state
    is_warming = "WARMING" in power_state

    if is_warming:
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        r2_c1, r2_c2 = st.columns(2)

        with r1_c1:
            if st.button("🧪 Prepare Sample", use_container_width=True):
                st.session_state.awaiting_prep = True
                active_command = "Prepare Sample"
        with r1_c2:
            if st.button("⏳ Wait", use_container_width=True):
                active_command = "Wait 20 minutes for lamp stabilization"
        with r1_c3:
            if st.button("🧪 Blank Instrument", use_container_width=True):
                st.session_state.selecting_blank = True
                st.rerun()

        with r2_c1:
            if measured_state == "YES":
                if st.button("🧮 Calculate Results", use_container_width=True):
                    active_command = "Calculate concentration per TM-0428"
            else:
                if st.button("📊 Measure Samples", use_container_width=True):
                    active_command = "Measure samples"
        with r2_c2:
            if st.button("📞 Call Supervisor", use_container_width=True):
                active_command = "Call Supervisor"
                is_supervisor_click = True

    else:
        r1_c1, r1_c2 = st.columns(2)
        r2_c1, r2_c2 = st.columns(2)

        with r1_c1:
            if is_cold:
                if st.button("⚡ Power ON Instrument", use_container_width=True):
                    active_command = "Power ON the Spectrophotometer"
            else:
                if st.button("🧪 Prepare Sample", use_container_width=True):
                    st.session_state.awaiting_prep = True
                    active_command = "Prepare Sample"

        with r1_c2:
            if st.button("🧪 Blank Instrument", use_container_width=True):
                st.session_state.selecting_blank = True
                st.rerun()

        with r2_c1:
            if measured_state == "YES":
                if st.button("🧮 Calculate Results", use_container_width=True):
                    active_command = "Calculate concentration per TM-0428"
            else:
                if st.button("📊 Measure Samples", use_container_width=True):
                    active_command = "Measure samples"
        with r2_c2:
            if st.button("📞 Call Supervisor", use_container_width=True):
                active_command = "Call Supervisor"
                is_supervisor_click = True

user_text = st.chat_input(
    "Enter command or calculation data (e.g., 'Prepare sample', 'Calculate', or custom actions)...",
    disabled=is_terminal,
)
if user_text:
    active_command = user_text
    st.session_state.selecting_blank = False

if active_command:
    with st.spinner("Processing workstation operation..."):
        run_llm_turn(active_command, is_supervisor_action=is_supervisor_click)
    st.rerun()
