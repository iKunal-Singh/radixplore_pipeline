# RadiXplore Candidate Coding Challenge: Mining Project Intelligence System

## 1. Project Overview

This project implements a modular, automated pipeline to address the RadiXplore Candidate Coding Challenge. The system is designed to extract critical intelligence from unstructured geological and mining reports in PDF format. It performs two primary tasks:

1.  **Named Entity Recognition (NER)**: Identifies mining project names within the text of the reports.
2.  **Geolocation Inference**: Infers the most plausible geographic coordinates (latitude and longitude) for each identified project.

The pipeline processes a collection of PDF files, identifies "PROJECT" entities, enriches them with geographic data, and produces a final structured output in JSONL format. This solution prioritizes accuracy, robustness, and reproducibility, incorporating bonus features like confidence scoring for both NER and geolocation tasks.

## 2. Project Structure

The project is organized into a clean, modular structure for maintainability and ease of use:


radixplore_pipeline/
├── main.py                # Main script to orchestrate the full pipeline
├── pipeline/
│   ├── init.py
│   ├── text_extractor.py  # Module for PDF parsing
│   ├── ner_model.py       # Module for NER model training and inference
│   └── geolocator.py      # Module for geolocation and disambiguation
├── models/                  # Directory to store the fine-tuned NER model
├── data/                    # Input directory for PDFs and annotation files
├── output/                  # Output directory for the final JSONL files
├── requirements.txt         # List of all Python package dependencies
└── README.md                # This documentation file


## 3. Setup and Execution

Follow these steps carefully to set up the environment and run the full pipeline. This process has been designed to be reliable and avoid common `ModuleNotFoundError` issues.

### Step 1: Create a Virtual Environment (Crucial)

This isolates the project's dependencies and is the most important step to ensure the code runs correctly.

```bash
# Navigate to the project's root directory (radixplore_pipeline)
python3 -m venv venv

Step 2: Activate the Virtual Environment
You must activate the environment in your terminal session before installing packages or running the script.

On macOS and Linux:

source venv/bin/activate

On Windows:

.\venv\Scripts\activate

Your terminal prompt should now be prefixed with (venv).

Step 3: Install Dependencies
Install all required libraries from the requirements.txt file. This command uses the pip from your newly activated virtual environment.

pip install -r requirements.txt

Step 4: Place Data Files
Place your PDF reports and the sample-annotations.json file inside the data/ directory.

Step 5: Run the Pipeline
Execute the main script from the project's root directory.

python main.py

The script will:

Train and save a new NER model in the models/ directory.

Create an intermediate projects_ner_output.jsonl in the output/ directory.

Create the final projects_final.jsonl in the output/ directory.

This new structure and setup process should definitively resolve the ModuleNotFoundError and make the entire project much easier to run and manage.