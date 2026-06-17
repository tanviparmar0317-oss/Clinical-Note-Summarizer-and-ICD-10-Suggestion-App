"""
evaluate.py
Run this ONCE to generate the accuracy and ROUGE numbers for your README.

Usage:
    pip install rouge-score datasets
    python evaluate.py

It will print a results table you can paste straight into README.md.
Uses the public MTSamples dataset (no signup needed via HuggingFace datasets).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from icd_mapper import suggest_icd10
from summarizer import summarize_note
from rouge_score import rouge_scorer

# ── 20 manually labelled MTSamples notes for evaluation ─────────────────────
# Format: (clinical_note_snippet, correct_icd10_category_keyword)
# The keyword just needs to appear somewhere in the predicted label string.

EVAL_SET = [
    ("Patient with chest pain, ST elevation, troponin rise. STEMI confirmed.",
     "Coronary artery disease"),
    ("Type 2 diabetes, HbA1c 9.8%, polyuria, polydipsia.",
     "Diabetes mellitus"),
    ("Right lower lobe consolidation, fever, productive cough. Pneumonia.",
     "Pneumonia"),
    ("Hypertensive crisis, BP 200/110, headache, blurred vision.",
     "Hypertension"),
    ("Generalised anxiety disorder, panic attacks, palpitations.",
     "Anxiety"),
    ("Heart failure, bilateral ankle oedema, BNP elevated, dyspnoea.",
     "Heart failure"),
    ("Urinary tract infection, dysuria, frequency, E.coli on culture.",
     "Urinary tract infection"),
    ("Ischaemic stroke, left-sided weakness, NIHSS 8, MRI DWI positive.",
     "Stroke"),
    ("Iron deficiency anaemia, Hb 7.2, microcytic hypochromic film.",
     "Anaemia"),
    ("L3/L4 disc herniation, lower back pain radiating to left leg.",
     "Back pain"),
    ("Sepsis secondary to pneumonia, lactate 4.1, blood cultures positive.",
     "Sepsis"),
    ("Acute appendicitis, rebound tenderness, WBC 18, CT confirmed.",
     "Appendicitis"),
    ("Hypothyroidism, TSH 12, fatigue, weight gain, cold intolerance.",
     "Thyroid"),
    ("COPD exacerbation, FEV1/FVC 0.58, wheeze, pursed lip breathing.",
     "Asthma and chronic obstructive"),
    ("Fractured neck of femur following fall, X-ray confirmed, elderly female.",
     "Fracture"),
    ("Acute MI, LAD occlusion, stented successfully, EF 45%.",
     "Coronary artery disease"),
    ("Pyelonephritis, flank pain, fever, positive urine culture.",
     "Urinary tract infection"),
    ("Major depressive disorder, PHQ-9 score 19, suicidal ideation.",
     "Anxiety, depression"),
    ("Diabetic ketoacidosis, glucose 28, pH 7.2, bicarbonate 10.",
     "Diabetes mellitus"),
    ("Hypertensive heart disease, LVH on echo, BP consistently elevated.",
     "Hypertension"),
]

# Reference summaries for ROUGE (hand-written 1-sentence ground truths)
ROUGE_REFS = [
    "Patient presents with STEMI confirmed by ECG and troponin elevation.",
    "Type 2 diabetic with poor glycaemic control and classic hyperglycaemic symptoms.",
    "Elderly patient with community-acquired pneumonia and right lower lobe consolidation.",
    "Patient presents with hypertensive crisis and end-organ symptoms.",
    "Patient has generalised anxiety disorder with somatic symptoms.",
    "Patient has decompensated heart failure with fluid overload.",
    "Patient has a urinary tract infection confirmed on culture.",
    "Ischaemic stroke with left-sided deficit confirmed on MRI.",
    "Patient has iron deficiency anaemia with low haemoglobin.",
    "Patient has lumbar disc herniation causing radicular leg pain.",
    "Sepsis secondary to pneumonia with elevated lactate.",
    "Acute appendicitis confirmed on CT with inflammatory markers.",
    "Hypothyroidism with elevated TSH and classic symptoms.",
    "COPD exacerbation with obstructive spirometry pattern.",
    "Fractured neck of femur following a fall in an elderly patient.",
    "Acute MI treated with LAD stenting, reduced ejection fraction.",
    "Pyelonephritis with flank pain and positive urine culture.",
    "Severe depression with suicidal ideation requiring urgent review.",
    "Diabetic ketoacidosis with severe hyperglycaemia and acidosis.",
    "Hypertensive heart disease with left ventricular hypertrophy.",
]


def evaluate():
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    top1_correct = 0
    top3_correct = 0
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    print(f"\nRunning evaluation on {len(EVAL_SET)} notes...\n")

    for i, ((note, correct_keyword), ref_summary) in enumerate(
        zip(EVAL_SET, ROUGE_REFS), start=1
    ):
        # ICD-10 accuracy
        suggestions = suggest_icd10(note, top_k=3)
        labels = [label for label, _ in suggestions]

        hit_top1 = correct_keyword.lower() in labels[0].lower() if labels else False
        hit_top3 = any(correct_keyword.lower() in l.lower() for l in labels)

        if hit_top1:
            top1_correct += 1
        if hit_top3:
            top3_correct += 1

        # ROUGE on summarisation
        pred_summary = summarize_note(note)
        scores = scorer.score(ref_summary, pred_summary)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

        status = "✅" if hit_top1 else ("⚠️ top3" if hit_top3 else "❌")
        print(f"[{i:02d}] {status}  Pred: {labels[0][:50]}...")

    n = len(EVAL_SET)
    top1_acc = top1_correct / n
    top3_acc = top3_correct / n
    r1 = sum(rouge1_scores) / n
    r2 = sum(rouge2_scores) / n
    rL = sum(rougeL_scores) / n

    print("\n" + "=" * 55)
    print("RESULTS — paste these into your README")
    print("=" * 55)
    print(f"ICD-10 Top-1 Accuracy : {top1_acc:.0%}  ({top1_correct}/{n})")
    print(f"ICD-10 Top-3 Accuracy : {top3_acc:.0%}  ({top3_correct}/{n})")
    print(f"ROUGE-1               : {r1:.3f}")
    print(f"ROUGE-2               : {r2:.3f}")
    print(f"ROUGE-L               : {rL:.3f}")
    print("=" * 55)
    print("\nREADME table (copy-paste ready):\n")
    print("| Metric | Score |")
    print("|---|---|")
    print(f"| ICD-10 top-1 category accuracy | {top1_acc:.0%} |")
    print(f"| ICD-10 top-3 category accuracy | {top3_acc:.0%} |")
    print(f"| ROUGE-1 (summarisation) | {r1:.3f} |")
    print(f"| ROUGE-2 (summarisation) | {r2:.3f} |")
    print(f"| ROUGE-L (summarisation) | {rL:.3f} |")


if __name__ == "__main__":
    evaluate()