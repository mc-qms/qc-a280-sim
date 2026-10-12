import os
import re
import streamlit as st
from openai import OpenAI

# =============================================================================
# PAGE CONFIGURATION & SETUP
# =============================================================================
st.set_page_config(
    page_title="QC Bench 4: UV-Vis Assay Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# OpenRouter / OpenAI API Setup
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OpenRouter API Key. Please configure `OPENROUTER_API_KEY` in Streamlit secrets or environment variables.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

SYSTEM_PROMPT = """You are the simulation engine and bench narrator for QC Workstation 4 running Test Method TM-0428 (Protein Concentration by UV-Vis Spectrophotometry).

OPERATING INSTRUCTIONS:
1. Adhere strictly to the injected [DIRECTOR NOTE] enclosed in the user turn. The Director Note contains deterministic state truths, exact instrument physics, and compliance boundaries calculated by the Python runtime.
2. If the Director Note includes a [SYSTEM AUDIT TELEMETRY: ...] block, transcribe it verbatim at the very top of your response.
3. If the Director Note provides a code block (such as an ANALYTICAL TESTING SUMMARY certificate), output it verbatim without modification.
4. When narrating the supervisor persona, maintain an authentic, professional, audit-weary tone. Keep dialogue concise (3 to 4 sentences maximum), cite specific SOP sections when pointing out deviations, and always end supervisor turns by having them physically exit the bench bay back to the office.
5. Never hallucinate physical assay data, readings, or calculations that contradict the Director Note. Python owns what must be true; you own the bench atmosphere and dialogue."""

WELCOME_TEXT = (
    "You are logged into **QC Workstation 4** for release testing of mAb Drug Substance lot **BDS-552** "
    "using **Test Method TM-0428**.\n\n"
    "The spectrophotometer is currently powered off (0 min warm-up). Calibrated pipettors, "
    "10 mm quartz cuvettes, and Formulation Buffer are staged at your bench.\n\n"
    "Refer to the **TM-0428** method binder in the sidebar to review the analytical procedure and begin testing."
)

# =============================================================================
# SESSION STATE INITIALIZATION & RESET
# =============================================================================
def reset_entire_workstation():
    """Restores all state badges, intent flags, and audit flags to pristine boot condition."""
    st.session_state.display_messages = [
        {"role": "assistant", "content": WELCOME_TEXT}
    ]
    st.session_state.api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": WELCOME_TEXT}
    ]
    st.session_state.state_badge = {
        "Power": "OFF (0 min)",
        "Blank": "NONE",
        "Prep": "NONE",
        "Measured": "NO",
        "In-Range": "UNTESTED",
    }
    st.session_state.last_reading_over_range = False
    st.session_state.last_readings = None
    st.session_state.selecting_blank = False
    st.session_state.cuvette_contents = "EMPTY"
    st.session_state.awaiting_prep = False
    st.session_state.awaiting_calc = False
    st.session_state.measured_while_warming = False

if "state_badge" not in st.session_state:
    reset_entire_workstation()

# Ensure keys exist if reloading mid-session
if "awaiting_prep" not in st.session_state:
    st.session_state.awaiting_prep = False
if "awaiting_calc" not in st.session_state:
    st.session_state.awaiting_calc = False
if "measured_while_warming" not in st.session_state:
    st.session_state.measured_while_warming = False
if "selecting_blank" not in st.session_state:
    st.session_state.selecting_blank = False
if "last_reading_over_range" not in st.session_state:
    st.session_state.last_reading_over_range = False
if "cuvette_contents" not in st.session_state:
    st.session_state.cuvette_contents = "EMPTY"
if "last_readings" not in st.session_state:
    st.session_state.last_readings = None

# =============================================================================
# DETERMINISTIC STATE & ROUTING ENGINE
# =============================================================================
def run_llm_turn(user_input: str):
    """
    Evaluates inputs against analytical rules, updates hardware state flags,
    and returns an injected Director Note and terminal execution flag.
    """
    u_lower = user_input.lower().strip()
    power_state = st.session_state.state_badge.get("Power", "OFF (0 min)")
    blank_state = st.session_state.state_badge.get("Blank", "NONE")
    prep_state = st.session_state.state_badge.get("Prep", "NONE")
    measured_state = st.session_state.state_badge.get("Measured", "NO")
    in_range_state = st.session_state.state_badge.get("In-Range", "UNTESTED")

    current_frozen_state = (
        f"[SYSTEM AUDIT TELEMETRY: Power: {power_state} | Blank: {blank_state} | "
        f"Prep: {prep_state} | Measured: {measured_state} | Range: {in_range_state}]"
    )

    is_terminal = False
    actual_api_prompt = ""

    # Global Escape detection (prevents intent traps from capturing navigation)
    is_global_escape = any(k in u_lower for k in [
        "supervisor", "lead", "power", "turn on", "ignite", "blank", "wait", 
        "method", "page", "reset", "wash", "rinse"
    ])

    # Slot-Filling Trap: Sample Prep
    is_prep_submission = False
    if st.session_state.awaiting_prep:
        if is_global_escape:
            st.session_state.awaiting_prep = False
        else:
            st.session_state.awaiting_prep = False
            is_prep_submission = True

    # Slot-Filling Trap: Calculations (captures bare number entries like '0.775, 0.773, 100, 1.45')
    is_calc_submission = False
    nums_in_input = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", user_input)
    if st.session_state.awaiting_calc:
        if is_global_escape:
            st.session_state.awaiting_calc = False
        else:
            st.session_state.awaiting_calc = False
            is_calc_submission = True
    elif measured_state == "YES" and len(nums_in_input) >= 4 and not is_global_escape:
        is_calc_submission = True

    # -------------------------------------------------------------------------
    # ROUTE 1: SUPERVISOR ESCALATION
    # -------------------------------------------------------------------------
    if any(k in u_lower for k in ["supervisor", "lead"]):
        if st.session_state.cuvette_contents == "EMPTY" and measured_state == "NO":
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Supervisor steps into the bay, glances at the empty cuvette holder, and reminds the analyst "
                "to rinse and load either the blanking buffer or test preparation per Section 5 before attempting an acquisition. Exits bay.]"
            )
        elif "OFF" in power_state:
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Supervisor looks at the unpowered spectrophotometer, points at the chassis rocker switch, "
                "and notes that both lamps require ignition and a 20-minute thermal stabilization period (§4.2). Exits bay.]"
            )
        elif st.session_state.last_reading_over_range:
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "ABSORBANCE SATURATION ROUTE: Supervisor notes A280 > 2.5 AU on the workstation monitor. "
                "Explains that reading neat bulk drug substance saturates the optical detector. Instructs the analyst "
                "to prepare a 1:100 volumetric dilution with Formulation Buffer per Section 5.3, re-blank, and re-scan. Exits bay.]"
            )
        elif measured_state == "YES" and st.session_state.measured_while_warming:
            lamp_status_str = "is stabilized now" if "READY" in power_state else "is STILL warming"
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "UN-STABILIZED OPTICAL ACQUISITION AUDIT ROUTE — 4 SENTENCES: "
                "Sentence 1 (Audit Check): Supervisor reviews the workstation trace log and points out that these scans were collected while the deuterium lamp was warming (T < 20 min). "
                "Sentence 2 (Thermal Drift): Explains that thermal baseline drift split the duplicate scans (0.781 vs 0.823 AU), causing the duplicate % difference to fail the <=2.0% precision limit (§6.4). "
                f"Sentence 3 (Directive): Clarifies that while the lamp {lamp_status_str}, this acquired dataset is invalid. Instructs them to re-zero against Formulation Buffer and re-scan under stabilized optics. "
                "Sentence 4 (Departure): Exits the bay back to the office.]"
            )
        elif measured_state == "YES" and not st.session_state.last_reading_over_range:
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Supervisor reviews the dual scan trace (~0.77-0.78 AU), confirms duplicate readings sit cleanly "
                "inside the linear range (0.200 to 1.200 AU), advises the analyst to run calculations per Section 6, and exits.]"
            )
        else:
            actual_api_prompt = (
                f"Call Supervisor [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
                "Supervisor steps in, reminds the analyst to review the method tabs in the sidebar binder for the next "
                "procedural step, and heads back to the office.]"
            )

    # -------------------------------------------------------------------------
    # ROUTE 2: HARDWARE POWER & THERMAL STABILIZATION
    # -------------------------------------------------------------------------
    elif any(k in u_lower for k in ["power", "turn on", "switch on", "ignite"]):
        if "OFF" in power_state:
            st.session_state.state_badge["Power"] = "WARMING (2 min)"
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Analyst flips the rocker switch on the spectrophotometer chassis. "
                "Describe the physical hardware response: cooling fan engages with a quiet whir, internal stepper motors "
                "click as monochromator and filter wheels complete homing alignment, and front panel status LED illuminates solid green. "
                "(The optical bench is light-tight; no visible glow escapes the chassis). "
                "On the PC workstation monitor, quote the UV-Vis software status verbatim: "
                "'Hardware: Online | Lamps: Ignited | Status: LAMP WARMING (2 of 20 minutes elapsed)'. "
                "CRITICAL: Stop immediately after reporting software status without providing unsolicited advice.]"
            )
        else:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Hardware is already powered on. State current status on PC: '{power_state}'.]"
            )

    elif any(k in u_lower for k in ["wait", "stabilize", "20 min", "warm"]):
        if "WARMING" in power_state:
            st.session_state.state_badge["Power"] = "READY (20 min)"
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Analyst waits 20 minutes for thermal equilibrium. "
                "Workstation monitor updates: 'Hardware: Online | Deuterium & Tungsten Lamps: Stabilized | Status: READY (20 min)'.]"
            )
        elif "READY" in power_state:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Lamps are already at steady-state thermal equilibrium (READY). "
                "No additional stabilization wait required.]"
            )
        else:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Spectrophotometer is powered OFF. Cannot stabilize unignited optics.]"
            )

    # -------------------------------------------------------------------------
    # ROUTE 3: BASELINE ZERO / BLANKING
    # -------------------------------------------------------------------------
    elif any(k in u_lower for k in ["blank", "zero"]):
        if "OFF" in power_state:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Hardware is offline. Blank rejected. State that instrument is powered off.]"
            )
        else:
            if any(w in u_lower for w in ["water", "di water", "diwater", "h2o"]):
                st.session_state.state_badge["Blank"] = "WATER (INVALID)"
                st.session_state.cuvette_contents = "WATER"
                actual_api_prompt = (
                    f"{user_input} [DIRECTOR NOTE: Analyst zeroes against purified DI water. Instrument registers 0.000 AU baseline. "
                    "(Note: Section 5.1 strictly mandates Formulation Buffer; DI water baseline constitutes a procedural deviation).]"
                )
            elif any(b in u_lower for b in ["buffer", "formulation"]):
                st.session_state.state_badge["Blank"] = "BUFFER"
                st.session_state.cuvette_contents = "BUFFER"
                actual_api_prompt = (
                    f"{user_input} [DIRECTOR NOTE: Analyst wipes quartz optical surfaces with a lint-free wipe, inserts cell into bay, "
                    "and performs zero against Formulation Buffer. Software executes auto-zero sweep at 280 nm and 320 nm: 0.000 AU.]"
                )
            else:
                st.session_state.selecting_blank = True
                actual_api_prompt = (
                    f"{user_input} [DIRECTOR NOTE: Analyst requests blank calibration. Prompt them to state the specific baseline solution "
                    "being loaded into the optical cell.]"
                )

    # -------------------------------------------------------------------------
    # ROUTE 4: SAMPLE PREPARATION
    # -------------------------------------------------------------------------
    elif is_prep_submission or any(k in u_lower for k in ["prep", "dilut", "pipette", "sample"]):
        if any(k in u_lower for k in ["neat", "undiluted", "direct"]):
            st.session_state.state_badge["Prep"] = "NEAT (UNDILUTED)"
            st.session_state.cuvette_contents = "NEAT_SAMPLE"
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Analyst pipettes concentrated BDS-552 bulk drug substance directly into the cuvette "
                "without dilution. Cuvette placed in optical bay.]"
            )
        elif any(k in u_lower for k in ["1:100", "1/100", "100", "dilute"]):
            st.session_state.state_badge["Prep"] = "1:100 (BUFFER)"
            st.session_state.cuvette_contents = "DILUTED_SAMPLE"
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Analyst executes 1:100 volumetric dilution using positive-displacement pipette: "
                "100 uL BDS-552 into 9.90 mL Formulation Buffer. Cuvette rinsed and loaded into bench.]"
            )
        else:
            st.session_state.awaiting_prep = True
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Analyst initiates sample preparation. Prompt them to state their sample recipe: "
                "dilution ratio (e.g., 1:100) and diluent choice.]"
            )

    # -------------------------------------------------------------------------
    # ROUTE 5: SPECTRAL SCANNING & MEASUREMENT
    # -------------------------------------------------------------------------
    elif any(k in u_lower for k in ["measure", "scan", "read", "acquire"]):
        if "OFF" in power_state:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Optical shutter cannot trigger while instrument power is OFF.]"
            )
        elif blank_state == "NONE":
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Acquisition aborted by software: Instrument has not been zeroed/blanked (§5.1).]"
            )
        elif st.session_state.cuvette_contents in ["EMPTY", "BUFFER", "WATER"]:
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Cuvette currently contains {st.session_state.cuvette_contents}. "
                "No analytical test preparation present in the optical cell.]"
            )
        else:
            st.session_state.state_badge["Measured"] = "YES"
            # Freeze optical acquisition state metadata
            st.session_state.measured_while_warming = ("WARMING" in power_state)

            if "NEAT" in prep_state:
                st.session_state.last_reading_over_range = True
                st.session_state.state_badge["In-Range"] = "NO (> 2.5 AU)"
                st.session_state.last_readings = (3.125, 0.412, 3.118, 0.409)
                actual_api_prompt = (
                    f"{user_input} [DIRECTOR NOTE: Optical detector saturates! "
                    "Display acquisition trace: Rep 1: A280 = 3.125 AU, A320 = 0.412 AU | Rep 2: A280 = 3.118 AU, A320 = 0.409 AU. "
                    "WORKSTATION WARNING: Absorbance exceeds linear dynamic range (0.200 - 1.200 AU). Re-dilution required.]"
                )
            else:
                st.session_state.last_reading_over_range = False
                st.session_state.state_badge["In-Range"] = "YES (0.2-1.2)"

                if st.session_state.measured_while_warming:
                    # Injects thermal noise into Replicate 2 to force precision failure
                    st.session_state.last_readings = (0.785, 0.010, 0.835, 0.012)
                    actual_api_prompt = (
                        f"{user_input} [DIRECTOR NOTE: Acquisition trace logged under UN-STABILIZED optics: "
                        "Replicate 1: A280 = 0.785 AU, A320 = 0.010 AU (Corrected A280 = 0.775 AU) | "
                        "Replicate 2: A280 = 0.835 AU, A320 = 0.012 AU (Corrected A280 = 0.823 AU). "
                        "Workstation note: High drift observed across duplicate scans.]"
                    )
                else:
                    st.session_state.last_readings = (0.783, 0.008, 0.781, 0.008)
                    actual_api_prompt = (
                        f"{user_input} [DIRECTOR NOTE: Acquisition trace logged under stabilized optics: "
                        "Replicate 1: A280 = 0.783 AU, A320 = 0.008 AU (Corrected A280 = 0.775 AU) | "
                        "Replicate 2: A280 = 0.781 AU, A320 = 0.008 AU (Corrected A280 = 0.773 AU). "
                        "Duplicate scans cleanly recorded within linear range.]"
                    )

    # -------------------------------------------------------------------------
    # ROUTE 6: CALCULATIONS & DISPOSITION
    # -------------------------------------------------------------------------
    elif is_calc_submission or any(k in u_lower for k in ["calc", "disposition"]):
        if measured_state != "YES":
            actual_api_prompt = (
                f"{user_input} [DIRECTOR NOTE: Pre-Calculation Gate: No analytical data collected (Measured: NO). "
                "Refuse calculation and state that no scan data exists in the active run buffer.]"
            )
        else:
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", user_input)
            if len(nums) < 4:
                st.session_state.awaiting_calc = True
                actual_api_prompt = (
                    f"{user_input} [DIRECTOR NOTE: Analyst is initiating Section 6 calculations. "
                    "Prompt them to submit the four required parameters: "
                    "1) Replicate 1 Corrected Absorbance (A280 - A320), "
                    "2) Replicate 2 Corrected Absorbance (A280 - A320), "
                    "3) Dilution Factor, and "
                    "4) Extinction Coefficient cited in TM-0428 Section 6.3 (Page 4 in the sidebar, where E = 1.45). "
                    "Provide the format example: '0.775, 0.773, 100, 1.45'. Do not calculate yet.]"
                )
            else:
                is_terminal = True
                st.session_state.awaiting_calc = False

                a1, a2, df, ext = float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])
                mean_a = (a1 + a2) / 2.0
                pct_diff = (abs(a1 - a2) / mean_a) * 100.0 if mean_a > 0 else 999.0
                conc = (mean_a / ext) * df if ext > 0 else 0.0

                failure_reasons = []
                if "WATER" in blank_state:
                    failure_reasons.append("Invalid Blank: Baseline zeroed with Water instead of Formulation Buffer (§5.1).")
                if "NEAT" in prep_state or st.session_state.last_reading_over_range:
                    failure_reasons.append("Saturation: Sample tested neat, exceeding detector linear dynamic range (§5.3).")
                if st.session_state.measured_while_warming:
                    failure_reasons.append("Thermal Drift: Scans acquired prior to 20-minute optical stabilization (§4.2).")
                if pct_diff > 2.0:
                    failure_reasons.append(f"Precision Failure: Duplicate % difference was {pct_diff:.2f}% (Limit: <= 2.0%, §6.4).")
                if not (50.0 <= conc <= 60.0):
                    failure_reasons.append(f"Out of Specification: Calculated concentration {conc:.2f} mg/mL outside 50.0-60.0 mg/mL range.")
                if abs(ext - 1.45) > 0.01:
                    failure_reasons.append(f"Incorrect Extinction Coefficient: Entered {ext} instead of 1.45 mL/(mg·cm) (§6.3).")

                if failure_reasons:
                    reasons_str = "\n".join([f"  • {r}" for r in failure_reasons])
                    summary_block = (
                        f"```text\n"
                        f"================================================================================\n"
                        f"ANALYTICAL TESTING SUMMARY: INVALID / REJECTED\n"
                        f"================================================================================\n"
                        f"Lot Number:           BDS-552\n"
                        f"Reported Value:       {conc:.2f} mg/mL\n"
                        f"Duplicate Precision:  {pct_diff:.2f}%\n"
                        f"Disposition:          OOS / DEVIATION RECORD OPENED\n\n"
                        f"Identified Deficiencies:\n"
                        f"{reasons_str}\n"
                        f"================================================================================\n"
                        f"```"
                    )
                    actual_api_prompt = (
                        f"{user_input} [DIRECTOR NOTE: Emit the following GxP audit summary verbatim without modification:\n\n"
                        f"{summary_block}\n\n"
                        f"Follow with 2 sentences from the supervisor explaining why the run was invalidated.]"
                    )
                else:
                    summary_block = (
                        f"```text\n"
                        f"================================================================================\n"
                        f"ANALYTICAL TESTING SUMMARY: PASS\n"
                        f"================================================================================\n"
                        f"Test Method:          TM-0428 (Protein Concentration by UV-Vis)\n"
                        f"Lot Number:           BDS-552\n"
                        f"Replicate 1 (Corr):   {a1:.3f} AU\n"
                        f"Replicate 2 (Corr):   {a2:.3f} AU\n"
                        f"Duplicate % Diff:     {pct_diff:.2f}% (Criteria: <= 2.0%)\n"
                        f"Dilution Factor:      {df:.0f}\n"
                        f"Extinction Coeff:     {ext:.2f} mL/(mg·cm)\n"
                        f"Mean Concentration:   {conc:.2f} mg/mL (Release Range: 50.0 - 60.0 mg/mL)\n"
                        f"Disposition:          CONFORMS TO SPECIFICATION - RELEASE APPROVED\n"
                        f"================================================================================\n"
                        f"```"
                    )
                    actual_api_prompt = (
                        f"{user_input} [DIRECTOR NOTE: Emit the following GxP audit certificate verbatim without modification:\n\n"
                        f"{summary_block}\n\n"
                        f"Follow with 2 sentences of supervisor praise approving second-person sign-off.]"
                    )

    # -------------------------------------------------------------------------
    # ROUTE 7: AMBIENT / UNROUTED INPUT
    # -------------------------------------------------------------------------
    else:
        actual_api_prompt = (
            f"{user_input} [DIRECTOR NOTE: Transcribe `{current_frozen_state}` on line 1. "
            "Narrate physical bench surroundings or subtle instrument fan hum in 1 to 2 concise sentences. "
            "Do not progress assay state or provide unsolicited instructions.]"
        )

    return actual_api_prompt, is_terminal

# =============================================================================
# SIDEBAR: AUDIT STATE & METHOD BINDER
# =============================================================================
with st.sidebar:
    st.subheader("🖥️ Hardware Audit State")
    st.json(st.session_state.state_badge)

    if st.button("🔄 Reset Workstation Bay", use_container_width=True):
        reset_entire_workstation()
        st.rerun()

    st.divider()

    st.subheader("📋 QC Method Binder")
    st.caption("Standard Operating Procedure TM-0428 (Rev 04)")

    tab1, tab2, tab3, tab4 = st.tabs(["Scope", "Hardware", "Prep", "Calculations"])

    with tab1:
        st.markdown(
            """
            **Section 1: Scope & Principle**
            Quantitation of protein concentration in bulk drug substance lot **BDS-552** by ultraviolet absorbance at 280 nm. Corrected for light scattering at 320 nm.
            """
        )

    with tab2:
        st.markdown(
            """
            **Section 4: Equipment & Parameters**
            * Dual-beam UV-Vis Spectrophotometer.
            * Matched 10 mm quartz cuvettes.
            * **§4.2 Lamp Warm-up:** Tungsten and Deuterium lamps must be ignited for **>= 20 minutes** prior to zeroing or acquisition to establish thermal equilibrium.
            """
        )

    with tab3:
        st.markdown(
            """
            **Section 5: Assay Procedure**
            * **§5.1 Baseline:** Zero instrument using **Formulation Buffer**. Do NOT use DI water.
            * **§5.2 Cuvette Wash:** Triple-rinse cell with sample prior to acquisition.
            * **§5.3 Sample Preparation:** Dilute BDS-552 **1:100** using Formulation Buffer (100 µL into 9.90 mL). Direct/neat injection causes detector saturation.
            """
        )

    with tab4:
        st.markdown(
            """
            **Section 6: Testing, Calculations & Release Specs**
            * **6.1 Testing:** Scan duplicate preparations at 280 nm and 320 nm.
            * **6.2 Correction:** `A_corr = A280 - A320`
            * **6.3 Extinction Coefficient & Quantitation:**
              * Extinction Coefficient ($\epsilon$) = **1.45 mL / (mg · cm)** at 280 nm.
              * Concentration Formula: `(Mean A_corr / 1.45) * DF`
            * **6.4 Precision Criterion:** Duplicate % difference must be **<= 2.0%**.
            * **6.5 Release Limits:** BDS-552 release range is **50.0 – 60.0 mg/mL**.
            """
        )

# =============================================================================
# MAIN INTERFACE: WORKBENCH CONSOLE
# =============================================================================
st.title("🔬 Analytical QC Bench 4: UV-Vis Testing Simulator")
st.caption("Release Testing for Bulk Drug Substance BDS-552 | Inverted-Control Architecture")

# Quick Action Buttons
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("⚡ Power On", use_container_width=True):
        st.session_state["queued_action"] = "Power on instrument"
with col2:
    if st.button("⏳ Wait 20 Min", use_container_width=True):
        st.session_state["queued_action"] = "Wait 20 minutes"
with col3:
    if st.button("0️⃣ Blank Buffer", use_container_width=True):
        st.session_state["queued_action"] = "Blank with Formulation Buffer"
with col4:
    if st.button("🧪 Prep 1:100", use_container_width=True):
        st.session_state["queued_action"] = "Prep 1:100 with buffer"
with col5:
    if st.button("👔 Call Lead", use_container_width=True):
        st.session_state["queued_action"] = "Call Supervisor"

# Display Existing Chat Thread
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Terminal Session State Detection
is_terminal_session = False
if st.session_state.display_messages:
    last_assistant_msg = st.session_state.display_messages[-1]["content"]
    if "ANALYTICAL TESTING SUMMARY:" in last_assistant_msg:
        is_terminal_session = True

# Completion UI Banner
if is_terminal_session:
    last_content = st.session_state.display_messages[-1]["content"]
    if "SUMMARY: PASS" in last_content:
        st.success("✅ **Assay Complete & Lot BDS-552 Released.** All analytical acceptance criteria met and verified.")
    else:
        st.error("❌ **Assay Invalidated & Deviation Logged.** Technical review packet rejected during QC disposition.")

    st.info("Analytical session closed. Review the telemetry audit trail above or reset the bay for another analyst.")

    if st.button("🏁 Clear Session & Reset Workstation", use_container_width=True):
        reset_entire_workstation()
        st.rerun()

# Input Processing (Action Buttons or Chat Input)
chat_input_val = st.chat_input("Enter workbench command or calculation parameters...")
active_user_prompt = st.session_state.pop("queued_action", None) or chat_input_val

if active_user_prompt and not is_terminal_session:
    # 1. Render User Message
    st.session_state.display_messages.append({"role": "user", "content": active_user_prompt})
    with st.chat_message("user"):
        st.markdown(active_user_prompt)

    # 2. Evaluate Deterministic State Engine
    director_prompt, is_turn_terminal = run_llm_turn(active_user_prompt)

    # 3. Call OpenRouter LLM
    st.session_state.api_messages.append({"role": "user", "content": director_prompt})

    with st.chat_message("assistant"):
        with st.spinner("Bench hardware responding..."):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct",
                    messages=st.session_state.api_messages,
                    temperature=0.2,
                )
                assistant_reply = response.choices[0].message.content
                st.markdown(assistant_reply)

                # Store Turn in History
                st.session_state.display_messages.append({"role": "assistant", "content": assistant_reply})
                st.session_state.api_messages.append({"role": "assistant", "content": assistant_reply})

                # Rerun immediately to trigger terminal banner if assay ended
                if is_turn_terminal:
                    st.rerun()

            except Exception as e:
                err_msg = f"API Error: {str(e)}"
                st.error(err_msg)
                st.session_state.display_messages.append({"role": "assistant", "content": err_msg})
