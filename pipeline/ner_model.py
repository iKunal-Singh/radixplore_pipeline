import json
import os
import re

# Use a relative import to get the function from a sibling module
from .text_extractor import extract_text_from_pdf

try:
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
except ImportError:
    raise ImportError("Hugging Face libraries not found. Please run 'pip install -r requirements.txt'.")

def convert_json_to_iob(annotations_path: str) -> list:
    """Converts annotation JSON to IOB format for training."""
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotation file not found at: {annotations_path}")

    with open(annotations_path, 'r', encoding='utf-8') as f: data = json.load(f)
    
    processed_records = []
    for record in data:
        text = record.get("data", {}).get("text", "")
        if not text: continue
        
        annotations = record.get("annotations", [{}])[0].get("result", [])
        tokens = text.split()
        tags = ['O'] * len(tokens)
        
        current_pos, token_spans = 0, []
        for token in tokens:
            start = text.find(token, current_pos)
            if start == -1: continue
            token_spans.append((start, start + len(token)))
            current_pos = start + len(token)

        for ann in annotations:
            val = ann.get("value", {})
            start_char, end_char, label = val.get("start"), val.get("end"), val.get("labels", [None])[0]
            if label != "PROJECT" or start_char is None: continue
            
            first = True
            for i, (token_start, token_end) in enumerate(token_spans):
                if max(start_char, token_start) < min(end_char, token_end):
                    tags[i] = f"B-{label}" if first else f"I-{label}"
                    first = False
        processed_records.append({"tokens": tokens, "ner_tags": tags})
    return processed_records

def fine_tune_ner_model(iob_data: list, model_output_dir: str):
    """Fine-tunes and saves a BERT model for NER."""
    print("\n--- Starting NER Model Fine-Tuning ---")
    label_list = ["O", "B-PROJECT", "I-PROJECT"]
    label_encoding = {label: i for i, label in enumerate(label_list)}
    for record in iob_data:
        record["ner_tags"] = [label_encoding.get(tag, 0) for tag in record["ner_tags"]]

    ner_dataset = Dataset.from_list(iob_data)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_and_align(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_idx, label_ids = None, []
            for word_idx in word_ids:
                label_ids.append(-100 if word_idx is None or word_idx == previous_word_idx else label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = ner_dataset.map(tokenize_and_align, batched=True)
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))

    args = TrainingArguments(output_dir=model_output_dir, learning_rate=2e-5, per_device_train_batch_size=8, num_train_epochs=3, weight_decay=0.01, logging_steps=50)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset, tokenizer=tokenizer)
    
    print("Training the model..."); trainer.train(); print("Training complete.")
    print(f"Saving model to '{model_output_dir}'..."); trainer.save_model(model_output_dir); print("Model saved.")

def generate_ner_output(model_path: str, pdf_files: list, output_file: str):
    """Applies the NER model to PDFs and generates a JSONL output with confidence scores."""
    print("\n--- Applying NER Model and Generating Output ---")
    ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple")
    if os.path.exists(output_file): os.remove(output_file)

    for pdf_path in pdf_files:
        print(f"Processing '{os.path.basename(pdf_path)}'...")
        page_texts = extract_text_from_pdf(pdf_path)
        for page_num, text in page_texts.items():
            if not text.strip(): continue
            sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
            for sentence in filter(lambda s: len(s) > 10, sentences):
                try:
                    for entity in ner_pipeline(sentence):
                        if entity['entity_group'] == 'PROJECT':
                            project_name = entity['word'].strip().rstrip('.,')
                            ner_confidence = round(float(entity['score']), 4)
                            record = {"pdf_file": os.path.basename(pdf_path), "page_number": page_num, "project_name": project_name, "ner_confidence": ner_confidence, "context_sentence": sentence.strip(), "coordinates": None}
                            with open(output_file, 'a', encoding='utf-8') as f:
                                json.dump(record, f); f.write('\n')
                except Exception: pass
    print(f"NER processing complete. Output saved to '{output_file}'.")
