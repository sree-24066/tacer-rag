"""
TACER-RAG Streamlit Interactive Demonstration.

Type-Aware and Confidence-Calibrated Hallucination Detection
and Evidence-Constrained Correction for RAG.

Usage:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

# Add src to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Offline flags removed — models are now cached locally after first download
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_OFFLINE", "1")

import streamlit as st
import torch

from tacer_rag.pipeline import TACERPipeline
from tacer_rag.detection.predict import PredictedSpan

CATEGORY_BADGE_STYLE = {
    "EVIDENT_CONFLICT": {
        "bg": "#d32f2f",
        "color": "#ffffff",
        "border": "#b71c1c",
        "label": "Evident Conflict",
    },
    "SUBTLE_CONFLICT": {
        "bg": "#f57c00",
        "color": "#ffffff",
        "border": "#e65100",
        "label": "Subtle Conflict",
    },
    "EVIDENT_BASELESS": {
        "bg": "#7b1fa2",
        "color": "#ffffff",
        "border": "#4a148c",
        "label": "Evident Baseless",
    },
    "SUBTLE_BASELESS": {
        "bg": "#0288d1",
        "color": "#ffffff",
        "border": "#01579b",
        "label": "Subtle Baseless",
    },
}

PRESET_EXAMPLES = {
    "✅  Example 1 — No Hallucinations (Clean Answer)": {
        "question": "What is the speed of light and what does it represent?",
        "context": (
            "The speed of light in a vacuum is exactly 299,792,458 metres per second, "
            "commonly denoted by the letter c. It is a fundamental physical constant and "
            "plays a central role in Einstein's theory of special relativity, expressed in the equation E=mc²."
        ),
        "answer": (
            "The speed of light in a vacuum is 299,792,458 metres per second, denoted by c. "
            "It is a fundamental physical constant and plays a central role in Einstein's theory "
            "of special relativity, expressed as E=mc²."
        ),
    },
    "🔴  Example 2 — Evident Conflict (Direct Factual Contradiction)": {
        "question": "Where is the Eiffel Tower located and when was it built?",
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. "
            "It has a height of 330 metres."
        ),
        "answer": (
            "The Eiffel Tower is located in Berlin, Germany. "
            "It was constructed between 1887 and 1889 and stands 330 metres tall."
        ),
    },
    "🟠  Example 3 — Subtle Conflict (Semantic Distortion)": {
        "question": "What effect does moderate exercise have on cardiovascular health?",
        "context": (
            "Studies consistently show that regular moderate-intensity exercise significantly reduces "
            "the risk of cardiovascular disease. Health guidelines recommend at least 150 minutes "
            "of moderate aerobic activity per week for adults."
        ),
        "answer": (
            "Regular moderate-intensity exercise completely eliminates the risk of cardiovascular disease. "
            "Health guidelines recommend at least 150 minutes of moderate aerobic activity per week for adults."
        ),
    },
    "🟣  Example 4 — Evident Baseless (Fabricated Specific Facts)": {
        "question": "Who was on the Apollo 11 mission and what did they find?",
        "context": (
            "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
            "Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. "
            "Command Module Pilot Michael Collins flew the command module Columbia alone in lunar orbit."
        ),
        "answer": (
            "Apollo 11 landed on the Moon in July 1969 with Neil Armstrong, Buzz Aldrin, and Michael Collins. "
            "The mission discovered vast underground water reservoirs and three ancient alien artefacts on the lunar surface."
        ),
    },
    "🔵  Example 5 — Subtle Baseless (Unjustified Inference)": {
        "question": "What are the first-line treatments for mild hypertension?",
        "context": (
            "Lifestyle modifications including reduced dietary sodium, regular physical exercise, and weight management "
            "are the first-line intervention for stage 1 hypertension without high cardiovascular risk. "
            "Medications such as ACE inhibitors or thiazide diuretics are initiated if lifestyle changes are insufficient after 3 to 6 months."
        ),
        "answer": (
            "For mild hypertension, lifestyle modifications such as reduced sodium intake and regular exercise are recommended first. "
            "These changes probably work by psychologically motivating patients to adopt healthier habits, "
            "which indirectly lowers blood pressure through stress reduction and improved mental wellbeing."
        ),
    },
}


def highlight_spans(answer: str, spans: list[PredictedSpan]) -> str:
    """Render HTML with high-contrast color-coded highlight tags for hallucination spans."""
    if not spans:
        return (
            "<div style='line-height: 1.8; font-size: 16px; padding: 16px; "
            "background-color: rgba(46, 160, 67, 0.12); border: 1px solid rgba(46, 160, 67, 0.4); "
            "border-radius: 8px; color: #3fb950;'>"
            f"✅ <b>Factual Grounding Verified:</b> No hallucinated spans detected. Answer is fully supported by context.</div>"
        )

    # Sort spans by start offset
    sorted_spans = sorted(spans, key=lambda s: s.start)
    html_parts = []
    last_idx = 0

    for s in sorted_spans:
        if s.start > last_idx:
            html_parts.append(answer[last_idx:s.start])

        cfg = CATEGORY_BADGE_STYLE.get(
            s.label_type,
            {"bg": "#d32f2f", "color": "#ffffff", "border": "#b71c1c", "label": s.label_type}
        )
        span_text = answer[s.start:s.end]
        badge = (
            f"<span style='background-color: {cfg['bg']}; color: {cfg['color']}; "
            f"border: 1px solid {cfg['border']}; padding: 3px 8px; border-radius: 6px; "
            f"font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,0.4); display: inline-block; margin: 2px 0;' "
            f"title='{cfg['label']} (Confidence: {s.confidence:.2f})'>"
            f"{span_text} <span style='font-size: 11px; opacity: 0.9; margin-left: 4px; padding: 1px 4px; "
            f"background: rgba(0,0,0,0.25); border-radius: 3px;'>{cfg['label']}</span></span>"
        )
        html_parts.append(badge)
        last_idx = s.end

    if last_idx < len(answer):
        html_parts.append(answer[last_idx:])

    content = "".join(html_parts)
    return (
        "<div style='line-height: 2.2; font-size: 16px; padding: 16px; "
        "background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; color: #f0f6fc;'>"
        f"{content}</div>"
    )


@st.cache_resource
def load_pipeline(ckpt_path_str: str):
    """Cache pipeline instance across Streamlit reruns. Auto-downloads from HF Hub if not local."""
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        from huggingface_hub import snapshot_download
        st.info("⏳ First run: downloading model weights from HuggingFace Hub (~570 MB). Please wait...")
        snapshot_download(
            repo_id="Sreekanth2005/tacer-rag-model",
            local_dir=str(ROOT / "experiments" / "checkpoints"),
        )
    return TACERPipeline.from_checkpoint(ckpt_path)


def main():
    st.set_page_config(
        page_title="TACER-RAG | Hallucination Detection & Correction",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ TACER-RAG")
    st.subheader("Type-Aware & Confidence-Calibrated Hallucination Detection and Evidence-Constrained Correction")

    st.markdown(
        """
        **TACER-RAG** reliably identifies hallucinated spans in RAG answers, classifies errors into 4 specific categories,
        calibrates confidence using Temperature Scaling, and applies evidence-constrained correction policies with post-correction verification.
        """
    )

    # Sidebar: Model and Configuration
    st.sidebar.header("⚙️ Configuration")

    DEFAULT_CKPT = "ModernBERT_trained_1500"
    checkpoints_dir = ROOT / "experiments" / "checkpoints"
    available_ckpts = [p.name for p in checkpoints_dir.glob("*") if (p / "best_model.pt").exists()]

    if not available_ckpts:
        # Cloud deployment: checkpoint will be auto-downloaded from HuggingFace Hub
        available_ckpts = [DEFAULT_CKPT]
        st.sidebar.info("⏳ Model not found locally — will auto-download from HuggingFace Hub (~570 MB) on first run.")

    default_idx = available_ckpts.index(DEFAULT_CKPT) if DEFAULT_CKPT in available_ckpts else 0
    selected_run = st.sidebar.selectbox("Checkpoint Model", available_ckpts, index=default_idx)
    ckpt_path = checkpoints_dir / selected_run / "best_model.pt"

    pipeline = load_pipeline(str(ckpt_path))

    # Threshold sliders
    st.sidebar.subheader("Confidence Operating Thresholds")
    use_cal = st.sidebar.checkbox("Apply Temperature Calibration", value=True)
    high_t = st.sidebar.slider("High Confidence Threshold", 0.5, 0.95, float(pipeline.controller.high_conf), 0.05)
    abstain_t = st.sidebar.slider("Abstain Below Cutoff", 0.1, 0.5, float(pipeline.controller.abstain_below), 0.05)
    run_pv = st.sidebar.checkbox("Post-Correction Verification (Max 2 Rounds)", value=True)

    pipeline.controller.high_conf = high_t
    pipeline.controller.abstain_below = abstain_t

    # Main Tabs
    tab_demo, tab_calibration, tab_about = st.tabs(["🚀 Live Interactive Demo", "📊 Calibration Analysis", "📖 Methodology & Architecture"])

    with tab_demo:
        # Preset selector
        preset_choice = st.selectbox("Choose a Preset Example or Enter Custom Text:", ["Custom"] + list(PRESET_EXAMPLES.keys()))

        if preset_choice != "Custom":
            default_q = PRESET_EXAMPLES[preset_choice]["question"]
            default_ctx = PRESET_EXAMPLES[preset_choice]["context"]
            default_ans = PRESET_EXAMPLES[preset_choice]["answer"]
        else:
            default_q = ""
            default_ctx = ""
            default_ans = ""

        col1, col2 = st.columns(2)
        with col1:
            q_input = st.text_input("User Question (Optional)", value=default_q)
            ctx_input = st.text_area("Retrieved Context (Evidence)", value=default_ctx, height=180)
        with col2:
            st.write("&nbsp;")
            ans_input = st.text_area("Generated RAG Answer (To Verify & Correct)", value=default_ans, height=180)

        run_btn = st.button("🔍 Detect and Correct Hallucinations", type="primary", use_container_width=True)

        if run_btn:
            if not ctx_input or not ans_input:
                st.warning("Please provide both Context and Answer.")
            else:
                with st.spinner("Executing TACER-RAG pipeline...\n(1st run may take 15–30s to load models)"):
                    result = pipeline.process(
                        context=ctx_input,
                        answer=ans_input,
                        question=q_input,
                        run_post_verification=run_pv,
                        apply_temperature=use_cal,
                    )

                st.divider()

                # Step 1: Detection Output
                st.subheader("1. Token-Level Hallucination Detection")
                st.markdown(highlight_spans(ans_input, result.detected_spans), unsafe_allow_html=True)

                # Legend
                st.markdown(
                    "<div style='margin-top: 12px; margin-bottom: 12px; font-size: 13px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;'>"
                    "<b style='color: #8b949e;'>Legend:</b> "
                    "<span style='background: #d32f2f; color: #fff; padding: 2px 7px; border-radius: 4px; font-weight: 500;'>Evident Conflict</span>"
                    "<span style='background: #f57c00; color: #fff; padding: 2px 7px; border-radius: 4px; font-weight: 500;'>Subtle Conflict</span>"
                    "<span style='background: #7b1fa2; color: #fff; padding: 2px 7px; border-radius: 4px; font-weight: 500;'>Evident Baseless</span>"
                    "<span style='background: #0288d1; color: #fff; padding: 2px 7px; border-radius: 4px; font-weight: 500;'>Subtle Baseless</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # Details Table
                if result.detected_spans:
                    st.write("#### Detected Suspicious Spans")
                    span_rows = []
                    for s in result.detected_spans:
                        span_rows.append({
                            "Span Text": s.text,
                            "Category": s.label_type,
                            "Confidence": f"{s.confidence:.4f}",
                            "Char Offsets": f"{s.start} - {s.end}",
                        })
                    st.table(span_rows)
                else:
                    st.success("✅ No hallucinations detected in the answer. Answer is fully supported by context.")

                # Step 2: Actions & Evidence
                st.subheader("2. Type-Aware Evidence Verification & Correction Actions")
                if result.actions_taken:
                    for idx, a in enumerate(result.actions_taken, 1):
                        with st.expander(f"Action {idx}: [{a.action.upper()}] for '{a.span_text}' ({a.label_type})"):
                            st.write(f"**Explanation:** {a.explanation}")
                            if a.evidence_text:
                                st.info(f"**Evidence Used:** {a.evidence_text}")
                            if a.nli_label:
                                st.write(f"**NLI Verification:** {a.nli_label} (confidence: {a.nli_confidence})")
                else:
                    st.write("No correction actions needed.")

                # Step 3: Final Answer & Post-Verification
                st.subheader("3. Final Corrected Answer & Post-Verification Audit")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info(f"**Original Answer:**\n\n{ans_input}")
                with res_col2:
                    st.success(f"**Final Answer:**\n\n{result.final_answer}")

                if result.post_verification:
                    pv = result.post_verification
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Rounds Executed", pv.rounds_executed)
                    m_col2.metric("Spans Removed", pv.removed_count)
                    m_col3.metric("New Hallucinations (NHR)", f"{pv.new_hallucination_rate:.2f}")
                    m_col4.metric("Regression Reverted", str(pv.reverted_to_original))

    with tab_calibration:
        st.subheader("Confidence Calibration & Reliability Analysis")
        cal_json_path = checkpoints_dir / selected_run / "calibration_params.json"
        fig_path = ROOT / "experiments" / "figures" / f"{selected_run}_reliability_diagram.png"

        if cal_json_path.exists():
            with open(cal_json_path, encoding="utf-8") as f:
                cal_data = json.load(f)

            c1, c2, c3 = st.columns(3)
            c1.metric("Learned Temperature (T)", f"{cal_data.get('temperature', 1.0):.4f}")
            val_m = cal_data.get("validation_metrics", {})
            c2.metric("Uncalibrated ECE", f"{val_m.get('uncalibrated_ece', 0.0):.4f}")
            c3.metric("Calibrated ECE", f"{val_m.get('calibrated_ece', 0.0):.4f}", f"-{val_m.get('ece_reduction_pct', 0.0)}%")

        if fig_path.exists():
            st.image(str(fig_path), caption="Reliability Diagram (Validation Set)", use_container_width=True)
        else:
            st.info("Run `python scripts/calibrate_detector.py` to generate the reliability diagram.")

    with tab_about:
        st.subheader("System Architecture & Research Design")
        st.markdown(
            """
            ### Pipeline Stages:
            1. **Input Processing:** Preserves exact character offsets and pairs context with response.
            2. **ModernBERT Detector:** Fine-tuned on RAGTruth with 9 BIO labels covering 4 error types.
            3. **Confidence Calibration:** Post-hoc Temperature Scaling minimizing negative log likelihood on validation data.
            4. **Evidence Retrieval:** TF-IDF relevance ranking extracts top evidence sentences from context without external web access.
            5. **Type-Aware Controller:** Decoupled policies (Replace with evidence, Rewrite meaning, Remove baseless, Qualify inference).
            6. **Post-Correction Verification:** Re-evaluates corrected answer, checks New Hallucination Rate, and reverts on regression.
            """
        )


if __name__ == "__main__":
    main()
