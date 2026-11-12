import os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from data_loader import load_dataset
from transformer_data import NERDataset, build_collator, map_chunk_predictions
from spans import bio_to_spans, normalize_bio
from evaluate import evaluate_spans
from transformers import TrainerCallback
import shutil
from analyze_errors import (analyze_dataset, print_error_summary, save_error_analysis)

# 1. KONFIGURASI
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "transformers"
MODEL_NAME = "indobenchmark/indobert-base-p1"
LABEL_LIST = ["O", "B-PPL", "I-PPL", "B-PLC", "I-PLC", "B-IND", "I-IND", "B-EVT", "I-EVT", "B-FNB", "I-FNB"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# 2. MEMUAT DATA
raw_data = load_dataset()
train_data = raw_data["train"]
valid_data = raw_data["validation"]
test_data = raw_data["test"]

print("Train:", len(train_data))
print("Validation:", len(valid_data))
print("Test:", len(test_data))

# 3. TOKENIZER DAN DATASET
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

train_dataset = NERDataset(train_data, tokenizer, LABEL2ID, MAX_LENGTH)
valid_dataset = NERDataset(valid_data, tokenizer, LABEL2ID, MAX_LENGTH)
test_dataset = NERDataset(test_data, tokenizer, LABEL2ID, MAX_LENGTH)

collator = build_collator(tokenizer)

# 4. EVALUASI EXACT-MATCH SPAN
def make_compute_metrics(dataset):
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        grouped = {}

        for i, metadata in enumerate(dataset.records):
            word_ids = dataset.word_ids[i]
            predictions = map_chunk_predictions(logits[i][:len(word_ids)], word_ids)

            record_id = metadata["id"]
            if record_id not in grouped:
                grouped[record_id] = {
                    "tokens": metadata["tokens"],
                    "labels": metadata["labels"],
                    "predictions": ["O"] * len(metadata["tokens"])
                }

            for word_idx, label_id in predictions.items():
                grouped[record_id]["predictions"][word_idx] = ID2LABEL[label_id]

        true_spans, pred_spans = [], []

        for item in grouped.values():
            true_spans.append(bio_to_spans(normalize_bio(item["labels"])))
            pred_spans.append(bio_to_spans(normalize_bio(item["predictions"])))

        results = evaluate_spans(true_spans, pred_spans)
        overall = results["micro_overall"]

        return {
            "micro_span_precision": overall["precision"],
            "micro_span_recall": overall["recall"],
            "micro_span_f1": overall["f1"]
        }

    return compute_metrics


# 5. MEMUAT MODEL PRETRAINED
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    label2id=LABEL2ID,
    id2label=ID2LABEL
)

# 6. KONFIGURASI TRAINING
training_args = TrainingArguments(
    output_dir="hasil_model",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_micro_span_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    logging_strategy="epoch",
    report_to="none",
    seed=SEED,
    data_seed=SEED,
    disable_tqdm=True
)


class TrainingDisplayCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("=" * 60)
        print("TRAINING TRANSFORMER NER")
        print("=" * 60)
        print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        print("Train:", len(train_dataset))
        print("Validation:", len(valid_dataset))
        print("Test:", len(test_dataset))
        print("Epochs:", EPOCHS)
        print("Batch Size:", BATCH_SIZE)
        print("Learning Rate:", LEARNING_RATE)
        print("=" * 60)

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = state.log_history
        epoch = int(state.epoch)

        train_logs = [
            item for item in logs
            if "loss" in item and "eval_loss" not in item
        ]

        val_logs = [
            item for item in logs
            if "eval_micro_span_f1" in item
        ]

        if train_logs and val_logs:
            train_loss = train_logs[-1]["loss"]
            val = val_logs[-1]

            print(f"\nEPOCH {epoch}/{EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Precision: {val['eval_micro_span_precision']:.4f}")
            print(f"Validation Recall: {val['eval_micro_span_recall']:.4f}")
            print(f"Validation Micro F1: {val['eval_micro_span_f1']:.4f}")
            print("-" * 60)

    def on_train_end(self, args, state, control, **kwargs):
        print("\nTRAINING SELESAI")
        print("Best Epoch:", state.best_model_checkpoint)
        print("Best Validation Micro F1:", f"{state.best_metric:.4f}")
        print("=" * 60)


# 7. TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=make_compute_metrics(valid_dataset),
    callbacks=[TrainingDisplayCallback()]
)

def run_error_analysis(trainer, test_dataset):
    print("\n" + "=" * 60)
    print("ANALISIS KESALAHAN")
    print("=" * 60)

    predictions = trainer.predict(test_dataset).predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    grouped = {}

    for i, metadata in enumerate(test_dataset.records):
        word_ids = test_dataset.word_ids[i]

        chunk_predictions = map_chunk_predictions(
            predictions[i][:len(word_ids)],
            word_ids
        )

        record_id = metadata["id"]

        if record_id not in grouped:
            grouped[record_id] = {
                "tokens": metadata["tokens"],
                "labels": metadata["labels"],
                "predictions": ["O"] * len(metadata["tokens"])
            }

        for word_idx, label_id in chunk_predictions.items():
            grouped[record_id]["predictions"][word_idx] = ID2LABEL[label_id]

    tokens_list = []
    y_true = []
    y_pred = []
    sentence_ids = []

    for record_id, item in grouped.items():
        tokens_list.append(item["tokens"])
        y_true.append(normalize_bio(item["labels"]))
        y_pred.append(normalize_bio(item["predictions"]))
        sentence_ids.append(record_id)

    errors = analyze_dataset(
        tokens_list=tokens_list,
        y_true=y_true,
        y_pred=y_pred,
        sentence_ids=sentence_ids
    )

    print_error_summary(errors)
    save_error_analysis(errors)

    # Pindahkan hasil dari folder CRF ke folder Transformer
    base_dir = Path(__file__).resolve().parent.parent

    source = base_dir / "results" / "crf" / "error_analysis.csv"
    destination = base_dir / "results" / "transformer" / "error_analysis.csv"

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.exists():
        shutil.move(str(source), str(destination))
        print(f"\nHasil analisis disimpan di: {destination}")
    else:
        print("\nFile hasil analisis tidak ditemukan.")

    return errors

# 8. FINE-TUNING
trainer.train()

# 9. SIMPAN MODEL TERBAIK
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

# 10. EVALUASI TEST SET
print("\n" + "=" * 60)
print("EVALUASI TEST SET")
print("=" * 60)

trainer.compute_metrics = make_compute_metrics(test_dataset)
test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

print("\nHASIL EXACT-MATCH SPAN")
print("-" * 60)
print(f"Test Loss: {test_results['test_loss']:.4f}")
print(f"Test Precision: {test_results['test_micro_span_precision']:.4f}")
print(f"Test Recall: {test_results['test_micro_span_recall']:.4f}")
print(f"Test Micro F1: {test_results['test_micro_span_f1']:.4f}")
print("-" * 60)

# 11. ANALISIS KESALAHAN
errors = run_error_analysis(trainer, test_dataset)
