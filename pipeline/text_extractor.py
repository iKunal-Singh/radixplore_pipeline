import fitz  # PyMuPDF
import pdfplumber
import os

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extracts text content from each page of a given PDF file.
    
    Prioritizes PyMuPDF for speed and falls back to pdfplumber for robustness.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A dictionary mapping page numbers (1-based) to their extracted text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: The file was not found at {pdf_path}")
    
    page_texts = {}
    # --- Primary Method: PyMuPDF (fitz) ---
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                page_texts[i + 1] = page.get_text("text")
        if any(page_texts.values()):
            return page_texts
    except Exception as e:
        print(f"PyMuPDF failed on {os.path.basename(pdf_path)}: {e}. Falling back to pdfplumber.")

    # --- Secondary Method: pdfplumber ---
    page_texts = {} # Reset in case PyMuPDF returned partial but empty data
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if text := page.extract_text():
                    page_texts[i + 1] = text
    except Exception as e:
        print(f"pdfplumber also failed on {os.path.basename(pdf_path)}: {e}")

    return page_texts
