"""
summarizer.py
Abstractive summarisation of clinical notes using DistilBART-CNN.
Model: sshleifer/distilbart-cnn-12-6  (~300 MB, runs on CPU)
"""

from transformers import pipeline

# Load once at import time so Gradio doesn't reload on every click
_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("[summarizer] Loading model...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        
        def run(text):
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=120, min_length=40, length_penalty=2.0, num_beams=4)
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        _summarizer = run
        print("[summarizer] Model loaded.")
    return _summarizer


def summarize_note(text: str) -> str:
    """
    Takes a free-text clinical note and returns a concise abstractive summary.

    Args:
        text: Raw clinical note string.

    Returns:
        A 2–3 sentence summary string.
    """
    text = text.strip()
    if not text:
        return "No input provided."

    # DistilBART handles up to ~1024 tokens; truncate very long notes
    max_input_chars = 3000
    if len(text) > max_input_chars:
        text = text[:max_input_chars]

    summarizer = _get_summarizer()

    return summarizer(text)