import sys
import os

# Fix the path — go up one level from app/ to reach src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import gradio as gr
from icd_mapper import suggest_icd10, format_icd10_output
from summarizer import summarize_note

EXAMPLES = [
    ["Patient is a 62-year-old male with chest pain, ST elevation, shortness of breath. History of hypertension. ECG shows STEMI. Troponin elevated."],
    ["55-year-old female with type 2 diabetes, HbA1c 9.8%, polyuria, polydipsia, blurred vision. On Metformin 1000mg BD."],
    ["78-year-old female with productive cough, fever 38.9°C, O2 sat 91%. CXR shows right lower lobe consolidation. Diagnosis: community-acquired pneumonia."],
]

def analyse_note(clinical_note: str):
    if not clinical_note or not clinical_note.strip():
        return "Please enter a clinical note.", "No input to classify."
    summary = summarize_note(clinical_note)
    suggestions = suggest_icd10(clinical_note, top_k=3)
    icd_output = format_icd10_output(suggestions)
    return summary, icd_output

with gr.Blocks(title="Clinical Note Summarizer + ICD-10 Tagger", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Clinical Note Summarizer + ICD-10 Suggestion App
    **BSc AI in Biomedical Engineering · FAU Erlangen-Nürnberg**

    Paste a clinical note → get an abstractive summary + ICD-10 diagnosis suggestions.
    > ⚠️ For educational purposes only. Not for clinical use.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            note_input = gr.Textbox(label="Clinical Note", placeholder="Paste a doctor's note here...", lines=12)
            analyse_btn = gr.Button("🔍 Analyse Note", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=note_input, label="Try an example")
        with gr.Column(scale=1):
            summary_output = gr.Textbox(label="📋 Summary (DistilBART-CNN)", lines=5, interactive=False)
            icd_output = gr.Textbox(label="🏷️ ICD-10 Suggestions (BART-large-MNLI zero-shot)", lines=10, interactive=False)
    analyse_btn.click(fn=analyse_note, inputs=note_input, outputs=[summary_output, icd_output])

if __name__ == "__main__":
    demo.launch()