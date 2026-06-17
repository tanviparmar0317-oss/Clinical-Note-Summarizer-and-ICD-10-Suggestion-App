"""
icd10_classifier.py
Zero-shot ICD-10 category suggestion using BART-large-MNLI.
Model: facebook/bart-large-mnli  (~1.6 GB, runs on CPU)

No labelled medical data or fine-tuning required — the model infers
relevance from the natural-language ICD-10 category descriptions.
"""

from transformers import pipeline
from typing import List, Tuple

# ICD-10 category labels — written as natural-language descriptions
# so BART-MNLI can reason about them effectively
ICD10_LABELS = [
    "Diabetes mellitus and blood sugar disorders (E10-E14)",
    "Hypertension and high blood pressure (I10-I15)",
    "Coronary artery disease and chest pain (I20-I25)",
    "Heart failure and cardiac arrest (I50-I51)",
    "Pneumonia and lower respiratory tract infection (J18)",
    "Asthma and chronic obstructive pulmonary disease (J40-J47)",
    "Urinary tract infection and kidney disease (N10-N39)",
    "Stroke and cerebrovascular disease (I60-I69)",
    "Anxiety, depression and mental health disorders (F30-F48)",
    "Fractures and musculoskeletal injuries (S00-S99)",
    "Anaemia and blood disorders (D50-D64)",
    "Sepsis and systemic infection (A40-A41)",
    "Appendicitis and abdominal conditions (K35-K37)",
    "Thyroid disorders and endocrine diseases (E00-E07)",
    "Back pain and spinal disorders (M40-M54)",
]

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        print("[icd10] Loading BART-large-MNLI model...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
        print("[icd10] Model loaded.")
    return _classifier


def suggest_icd10(text: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Suggests the most likely ICD-10 categories for a clinical note.

    Args:
        text:  Raw clinical note or its summary.
        top_k: Number of top suggestions to return (default 3).

    Returns:
        List of (label, confidence_score) tuples, sorted by score descending.
        Example: [("Diabetes mellitus... (E10-E14)", 0.91), ...]
    """
    text = text.strip()
    if not text:
        return [("No input provided.", 0.0)]

    # Use at most 1000 chars — MNLI works best on shorter passages
    if len(text) > 1000:
        text = text[:1000]

    classifier = _get_classifier()

    result = classifier(
        text,
        candidate_labels=ICD10_LABELS,
        multi_label=True,   # a note can match multiple ICD categories
    )

    # Zip labels + scores and return top_k
    pairs = list(zip(result["labels"], result["scores"]))
    return pairs[:top_k]


def format_icd10_output(suggestions: List[Tuple[str, float]]) -> str:
    """
    Formats ICD-10 suggestions into a clean string for display in Gradio.
    """
    lines = []
    for rank, (label, score) in enumerate(suggestions, start=1):
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        lines.append(f"{rank}. {label}\n   Confidence: {bar} {score:.0%}")
    return "\n\n".join(lines)