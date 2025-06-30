import os
import shutil
import sys

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from pipeline.text_extractor import extract_text_from_pdf
    from pipeline.ner_model import (
        convert_json_to_iob,
        fine_tune_ner_model,
        generate_ner_output,
    )
    from pipeline.geolocator import run_geolocation_pipeline
except ImportError as e:
    print(f"Error: A required module could not be imported: {e}")
    print("Please ensure you have run 'pip install -r requirements.txt' in your virtual environment.")
    sys.exit(1)


def main():
    """
    Orchestrates the entire RadiXplore intelligence pipeline from start to finish.
    """
    # --- Configuration: Define all paths relative to this script ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Ensure all necessary directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    ANNOTATIONS_FILE = os.path.join(DATA_DIR, "sample-annotations.json")
    PDF_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    
    MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, "ner_model")
    NER_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "projects_ner_output.jsonl")
    FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "projects_final.jsonl")
    
    print("--- Starting RadiXplore Intelligence Pipeline ---")

    # --- Step 1: Check for data files ---
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Error: Annotation file not found at '{ANNOTATIONS_FILE}'.")
        print("Please place 'sample-annotations.json' in the 'data' directory.")
        return
    if not PDF_FILES:
        print(f"Error: No PDF files found in the '{DATA_DIR}' directory.")
        return

    # --- Step 2: Prepare IOB data for NER training ---
    iob_data = convert_json_to_iob(ANNOTATIONS_FILE)
    if not iob_data:
        print("Could not generate IOB data from annotations. Halting.")
        return

    # --- Step 3: Fine-tune the NER model ---
    if os.path.exists(MODEL_OUTPUT_DIR):
        print(f"Model directory '{MODEL_OUTPUT_DIR}' already exists. Reusing it.")
    else:
        fine_tune_ner_model(iob_data, MODEL_OUTPUT_DIR)

    # --- Step 4: Generate intermediate NER output ---
    generate_ner_output(MODEL_OUTPUT_DIR, PDF_FILES, NER_OUTPUT_FILE)
    
    # --- Step 5: Run the Geolocation Pipeline ---
    run_geolocation_pipeline(NER_OUTPUT_FILE, FINAL_OUTPUT_FILE)

    # --- Step 6: Verification ---
    if os.path.exists(FINAL_OUTPUT_FILE):
        print("\n--- Pipeline Complete. Verifying Final Output... ---")
        with open(FINAL_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            print(f"First 5 entries in '{os.path.basename(FINAL_OUTPUT_FILE)}':")
            for i, line in enumerate(f):
                if i >= 5: break
                print(line.strip())
        print("--- End of Verification ---")
    else:
        print("--- Pipeline Finished, but the final output file was not created. Please check logs. ---")


if __name__ == '__main__':
    main()
