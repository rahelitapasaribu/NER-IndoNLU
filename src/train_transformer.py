import csv
import json
import time
import torch
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, TrainerCallback)
from data_loader import load_dataset
from transformer_data import (NERDataset, build_collator, map_chunk_predictions)
from spans import normalize_bio, bio_to_spans
from evaluate import (evaluate_spans, evaluate_bio, ENTITY_TYPES, BIO_LABELS)


# 1. KONFIGURASI
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_BASE_DIR = BASE_DIR / "models" / "transformer"
RESULT_BASE_DIR = BASE_DIR / "results" / "transformer"
LABEL_LIST = BIO_LABELS
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 2. PERSIAPAN OUTPUT
def get_experiment_dirs(normalize_labels):

    if normalize_labels:
        name = "with_normalization"
    else:
        name = "without_normalization"

    model_dir = MODEL_BASE_DIR / name
    result_dir = RESULT_BASE_DIR / name

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return name, model_dir, result_dir


# 3. MEMBUAT DATASET TRANSFORMER
def prepare_datasets(dataset, tokenizer, normalize_labels):

    train_dataset = NERDataset(
        dataset["train"],
        tokenizer,
        LABEL2ID,
        max_length=MAX_LENGTH,
        normalize_labels=normalize_labels
    )

    valid_dataset = NERDataset(
        dataset["validation"],
        tokenizer,
        LABEL2ID,
        max_length=MAX_LENGTH,
        normalize_labels=normalize_labels
    )

    test_dataset = NERDataset(
        dataset["test"],
        tokenizer,
        LABEL2ID,
        max_length=MAX_LENGTH,
        normalize_labels=normalize_labels
    )

    print("\nJUMLAH DATASET")
    print("Train chunks     :", len(train_dataset))
    print("Validation chunks:", len(valid_dataset))
    print("Test chunks      :", len(test_dataset))

    return train_dataset, valid_dataset, test_dataset


# 4. MENGGABUNGKAN PREDIKSI CHUNK
def collect_predictions(logits, dataset):

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
            grouped[record_id]["predictions"][word_idx] = (
                ID2LABEL[label_id]
            )

    return list(grouped.values())


# 5. EVALUASI BIO DAN EXACT-MATCH SPAN
def calculate_metrics(grouped, normalize_labels):

    y_true = []
    y_pred = []

    for item in grouped:

        true_labels = list(item["labels"])
        pred_labels = list(item["predictions"])

        if normalize_labels:

            true_labels = normalize_bio(true_labels)
            pred_labels = normalize_bio(pred_labels)

        y_true.append(true_labels)
        y_pred.append(pred_labels)

    true_spans = [bio_to_spans(seq) for seq in y_true]
    pred_spans = [bio_to_spans(seq) for seq in y_pred]
    span_result = evaluate_spans(true_spans, pred_spans, entity_types=ENTITY_TYPES)
    bio_result = evaluate_bio(y_true, y_pred)

    return y_true, y_pred, span_result, bio_result


# 6. COMPUTE METRICS UNTUK HUGGINGFACE TRAINER
def make_compute_metrics(dataset, normalize_labels):
    def compute_metrics(eval_pred):
        grouped = collect_predictions(
            eval_pred.predictions,
            dataset
        )

        _, _, span_result, bio_result = calculate_metrics(
            grouped,
            normalize_labels
        )

        span = span_result["micro_overall"]
        macro = bio_result["macro_avg"]
        weighted = bio_result["weighted_avg"]
        micro = bio_result["micro_overall"]

        return {
            "micro_span_precision": span["precision"],
            "micro_span_recall": span["recall"],
            "micro_span_f1": span["f1"],

            "bio_accuracy": bio_result["accuracy"],

            "bio_macro_precision": macro["precision"],
            "bio_macro_recall": macro["recall"],
            "bio_macro_f1": macro["f1"],

            "bio_weighted_f1": weighted["f1"],
            "bio_micro_f1": micro["f1"]
        }

    return compute_metrics


# 7. MENAMPILKAN HASIL EVALUASI
def print_span_metrics(result, title):
    print(f"\n{title}")
    print(
        f"{'Entity':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'TP':>8}"
        f"{'FP':>8}"
        f"{'FN':>8}"
    )

    for entity_type in ENTITY_TYPES + ["micro_overall"]:
        metric = result[entity_type]
        print(
            f"{entity_type:<15}"
            f"{metric['precision']:>12.4f}"
            f"{metric['recall']:>12.4f}"
            f"{metric['f1']:>12.4f}"
            f"{metric['tp']:>8}"
            f"{metric['fp']:>8}"
            f"{metric['fn']:>8}"
        )


def print_bio_metrics(result, title):
    print(f"\n{title}")
    print(
        f"{'Label':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    for label in BIO_LABELS:
        metric = result[label]
        print(
            f"{label:<15}"
            f"{metric['precision']:>12.4f}"
            f"{metric['recall']:>12.4f}"
            f"{metric['f1']:>12.4f}"
            f"{metric['support']:>12}"
        )

    for avg_name in [
        "macro_avg",
        "weighted_avg",
        "micro_overall"
    ]:

        metric = result[avg_name]

        print(
            f"{avg_name:<15}"
            f"{metric['precision']:>12.4f}"
            f"{metric['recall']:>12.4f}"
            f"{metric['f1']:>12.4f}"
            f"{metric.get('support', '-'):>12}"
        )

    print(f"\nToken Accuracy: {result['accuracy']:.4f}")


# 8. MENYIMPAN HASIL SPAN
def save_span_results(result, output_path):
    fieldnames = [
        "entity",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for entity in ENTITY_TYPES + ["micro_overall"]:
            metric = result[entity]
            writer.writerow({
                "entity": entity,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "tp": metric["tp"],
                "fp": metric["fp"],
                "fn": metric["fn"]
            })

    print("Hasil span disimpan:", output_path)


# 9. MENYIMPAN HASIL BIO
def save_bio_results(result, output_path):
    fieldnames = [
        "label",
        "precision",
        "recall",
        "f1",
        "support"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for label in BIO_LABELS:
            metric = result[label]
            writer.writerow({
                "label": label,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "support": metric["support"]
            })

        for name in [
            "macro_avg",
            "weighted_avg",
            "micro_overall"
        ]:

            metric = result[name]
            
            writer.writerow({
                "label": name,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "support": metric.get("support", "")
            })

        writer.writerow({
            "label": "accuracy",
            "precision": "",
            "recall": "",
            "f1": result["accuracy"],
            "support": ""
        })

    print("Hasil BIO disimpan:", output_path)


# 10. CALLBACK TRAINING
class TrainingDisplayCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):

        logs = state.log_history

        train_logs = [item for item in logs if "loss" in item and "eval_loss" not in item]
        val_logs = [item for item in logs if "eval_micro_span_f1" in item]

        if train_logs and val_logs:
            train_loss = train_logs[-1]["loss"]
            val = val_logs[-1]

            print(f"\nEPOCH {int(state.epoch)}")
            print(f"Train Loss: {train_loss:.4f}")
            print("Validation Span Precision:", f"{val['eval_micro_span_precision']:.4f}")
            print("Validation Span Recall:", f"{val['eval_micro_span_recall']:.4f}")
            print("Validation Span F1:", f"{val['eval_micro_span_f1']:.4f}")


# 11. MENJALANKAN SATU EKSPERIMEN
def run_experiment(normalize_labels):
    name, model_dir, result_dir = get_experiment_dirs(normalize_labels)

    print("\n" + "=" * 70)
    print("EKSPERIMEN:", name.upper())
    print("Normalisasi BIO:", normalize_labels)
    print("=" * 70)

    # Memuat dataset
    dataset = load_dataset()

    print("\nJUMLAH DATA")
    print("Train:", len(dataset["train"]))
    print("Validation:", len(dataset["validation"]))
    print("Test:", len(dataset["test"]))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Dataset Transformer
    train_dataset, valid_dataset, test_dataset = prepare_datasets(
        dataset,
        tokenizer,
        normalize_labels
    )

    collator = build_collator(tokenizer)

    # Model pretrained
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(result_dir / "checkpoints"),
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(valid_dataset, normalize_labels),
        callbacks=[TrainingDisplayCallback()]
    )

    # Training
    print("\nMEMULAI TRAINING")
    start_time = time.perf_counter()
    trainer.train()
    training_time = time.perf_counter() - start_time

    print(f"\nWaktu training: {training_time:.2f} detik")
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation Span F1:", f"{trainer.state.best_metric:.4f}")

    # Menyimpan model terbaik
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Metadata
    metadata = {
        "model": MODEL_NAME,
        "experiment": name,
        "normalization": normalize_labels,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "training_time": training_time,
        "selection_metric": "validation_micro_span_f1",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_span_f1": trainer.state.best_metric,
        "entity_types": ENTITY_TYPES,
        "bio_labels": BIO_LABELS
    }

    with open(
        result_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    # EVALUASI VALIDATION
    print("\nEVALUASI VALIDATION")
    val_output = trainer.predict(valid_dataset)

    val_grouped = collect_predictions(val_output.predictions, valid_dataset)
    _, _, val_span, val_bio = calculate_metrics(val_grouped, normalize_labels)

    print_span_metrics(val_span, "VALIDATION EXACT-MATCH SPAN")
    print_bio_metrics(val_bio, "VALIDATION TOKEN-LEVEL BIO")
    save_span_results(val_span, result_dir / "validation_span_results.csv")
    save_bio_results(val_bio, result_dir / "validation_bio_results.csv")

    # EVALUASI TEST
    print("\n" + "=" * 70)
    print("EVALUASI TEST SET")
    print("=" * 70)

    trainer.compute_metrics = make_compute_metrics(test_dataset, normalize_labels)
    test_output = trainer.predict(test_dataset)
    test_grouped = collect_predictions(test_output.predictions, test_dataset)
    
    y_true, y_pred, test_span, test_bio = calculate_metrics(test_grouped, normalize_labels)

    print_span_metrics(test_span, "TEST EXACT-MATCH SPAN")
    print_bio_metrics(test_bio, "TEST TOKEN-LEVEL BIO")
    save_span_results(test_span, result_dir / "test_span_results.csv")
    save_bio_results(test_bio, result_dir / "test_bio_results.csv")

    # RINGKASAN EKSPERIMEN
    span = test_span["micro_overall"]
    macro = test_bio["macro_avg"]

    summary = {
        "experiment": name,
        "normalization": normalize_labels,
        "training_time": round(training_time, 2),
        "validation_span_f1": trainer.state.best_metric,
        "test_span_precision": span["precision"],
        "test_span_recall": span["recall"],
        "test_span_f1": span["f1"],
        "test_bio_accuracy": test_bio["accuracy"],
        "test_bio_macro_precision": macro["precision"],
        "test_bio_macro_recall": macro["recall"],
        "test_bio_macro_f1": macro["f1"],
        "test_bio_weighted_f1": test_bio["weighted_avg"]["f1"],
        "test_bio_micro_f1": test_bio["micro_overall"]["f1"]
    }

    with open(
        result_dir / "experiment_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys())
        )

        writer.writeheader()
        writer.writerow(summary)

    print("\nRINGKASAN EKSPERIMEN")
    print(summary)

    return summary


# 12. EKSEKUSI DUA EKSPERIMEN
if __name__ == "__main__":

    all_results = []

    # Eksperimen 1: dengan normalisasi
    result_normalized = run_experiment(
        normalize_labels=True
    )

    all_results.append(result_normalized)

    # Eksperimen 2: tanpa normalisasi
    result_not_normalized = run_experiment(
        normalize_labels=False
    )

    all_results.append(result_not_normalized)

    # Menyimpan perbandingan kedua eksperimen
    comparison_path = (
        RESULT_BASE_DIR / "comparison_results.csv"
    )

    RESULT_BASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        comparison_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("PERBANDINGAN EKSPERIMEN SELESAI")
    print("=" * 70)

    for result in all_results:
        print("\n", result["experiment"])
        print("Test Span F1:", f"{result['test_span_f1']:.4f}")
        print("Test BIO Macro F1:", f"{result['test_bio_macro_f1']:.4f}")

    print("\nHasil perbandingan:", comparison_path)
