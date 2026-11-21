from pathlib import Path
import token
import results
import joblib
import time
import csv

import sklearn_crfsuite

from data_loader import load_dataset
from crf_features import sentence2features
from spans import normalize_bio, bio_to_spans
from evaluate import evaluate_spans, evaluate_bio, ENTITY_TYPES, BIO_LABELS
from analyze_errors import analyze_dataset, print_error_summary, save_error_analysis


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "crf"
RESULT_DIR = BASE_DIR / "results" / "crf"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def split_xy(sentences, normalize_labels=False):
    """Mengubah sequence menjadi fitur X dan label y. Normalisasi hanya dilakukan apabila normalize_labels=True"""
    X = []
    y = []

    total_changed = 0

    for idx, sentence in enumerate(sentences):

        tokens = sentence["tokens"]
        original_labels = list(sentence["labels"])

        if not tokens:
            raise ValueError(f"Sequence kosong pada indeks {idx}")

        if len(tokens) != len(original_labels):
            raise ValueError(f"Jumlah token dan label berbeda pada indeks {idx}")

        labels = original_labels

        if normalize_labels:

            labels = normalize_bio(original_labels)
            total_changed += sum(original != normalized for original, normalized in zip(original_labels, labels))

        features = sentence2features(tokens)

        if len(features) != len(labels):
            raise ValueError(f"Jumlah fitur dan label berbeda pada indeks {idx}")

        X.append(features)
        y.append(labels)

    print(f"Jumlah sequence : {len(X)}")
    print(f"Jumlah token    : {sum(map(len, X))}")
    print(f"Label diubah    : {total_changed}")

    return X, y


def predict_evaluate(model, X, y_true):
    """Melakukan evaluasi entity-level span dan token-level BIO"""
    y_pred = model.predict(X)

    """Evaluasi entity-level exact-match span"""
    true_spans = [bio_to_spans(seq) for seq in y_true]
    pred_spans = [bio_to_spans(seq) for seq in y_pred]

    span_result = evaluate_spans(true_spans, pred_spans, entity_types=ENTITY_TYPES)

    """Evaluasi token-level BIO"""
    bio_result = evaluate_bio(y_true, y_pred)

    return y_pred, span_result, bio_result


def print_span_metrics(result, title):
    """Menampilkan hasil evaluasi exact-match span"""
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
    """Menampilkan hasil evaluasi token-level BIO"""
    print(f"\n{title}")
    print(
        f"{'Label':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    """Metrik untuk setiap label BIO"""
    for label in BIO_LABELS:
        if label not in result:
            continue

        metric = result[label]

        print(
            f"{label:<15}"
            f"{metric['precision']:>12.4f}"
            f"{metric['recall']:>12.4f}"
            f"{metric['f1']:>12.4f}"
            f"{metric['support']:>12}"
        )

    """Ringkasan rata-rata"""
    for avg_name in ["macro_avg", "weighted_avg", "micro_overall"]:
        if avg_name not in result:
            continue

        metric = result[avg_name]

        print(
            f"{avg_name:<15}"
            f"{metric['precision']:>12.4f}"
            f"{metric['recall']:>12.4f}"
            f"{metric['f1']:>12.4f}"
            f"{metric.get('support', '-'):>12}"
        )

    if "accuracy" in result:
        print(f"\nToken Accuracy : {result['accuracy']:.4f}")



""" Konfigurasi eksperimen CRF """
CONFIGURATIONS = [
    {
        "name": "CRF-01",
        "algorithm": "lbfgs",
        "max_iterations": 200,
        "c1": 0.01,
        "c2": 0.01,
        "all_possible_transitions": True
    },

    {
        "name": "CRF-02",
        "algorithm": "lbfgs",
        "max_iterations": 200,
        "c1": 0.1,
        "c2": 0.01,
        "all_possible_transitions": True
    },

    {
        "name": "CRF-03",
        "algorithm": "lbfgs",
        "max_iterations": 200,
        "c1": 0.5,
        "c2": 0.01,
        "all_possible_transitions": True
    },

    {
        "name": "CRF-04",
        "algorithm": "lbfgs",
        "max_iterations": 200,
        "c1": 0.1,
        "c2": 0.1,
        "all_possible_transitions": True
    },

    {
        "name": "CRF-05",
        "algorithm": "lbfgs",
        "max_iterations": 200,
        "c1": 0.5,
        "c2": 0.1,
        "all_possible_transitions": True
    }
]


def build_crf(config):
    """" Membangun model CRF berdasarkan konfigurasi"""
    model = sklearn_crfsuite.CRF(
        algorithm=config["algorithm"],
        max_iterations=config["max_iterations"],
        c1=config["c1"],
        c2=config["c2"],
        all_possible_transitions=(config["all_possible_transitions"]),
        verbose=False
    )

    return model


def save_experiment_results(results):
    """Menyimpan hasil eksperimen validation ke CSV"""
    output_path = RESULT_DIR / "validation_results.csv"

    fieldnames = [
        "experiment",
        "c1",
        "c2",
        "max_iterations",
        "training_time",

        "span_precision",
        "span_recall",
        "span_f1",

        "bio_accuracy",
        "bio_macro_precision",
        "bio_macro_recall",
        "bio_macro_f1",
        "bio_weighted_f1",
        "bio_micro_f1"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nHasil eksperimen disimpan: {output_path}")


def select_best_model(X_train, y_train, X_val, y_val):
    """Melakukan eksperimen beberapa konfigurasi CRF. Model terbaik dipilih berdasarkan exact-match span F1 pada validation"""
    best_model = None
    best_config = None
    best_span_result = None
    best_bio_result = None
    best_f1 = -1.0

    experiment_results = []

    for config in CONFIGURATIONS:

        print(f"\nTRAINING {config['name']}")
        print("Konfigurasi:", config)

        model = build_crf(config)

        start_time = time.perf_counter()
        
        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        print(f"Waktu training: {training_time:.2f} detik")
        print("Jumlah label yang dipelajari:", len(model.classes_))

        """Evaluasi validation menggunakan dua pendekatan"""
        y_pred, span_result, bio_result = predict_evaluate(model, X_val, y_val)

        print_span_metrics(span_result, f"VALIDATION EXACT-MATCH SPAN - {config['name']}")
        print_bio_metrics(bio_result, f"VALIDATION TOKEN-LEVEL BIO - {config['name']}")
        
        """Simpan detail evaluasi validation setiap konfigurasi"""
        save_span_results(span_result, f"validation_{config['name']}_span_results.csv")
        save_bio_results(bio_result, f"validation_{config['name']}_bio_results.csv")

        """Dasar pemilihan model: exact-match span F1"""
        span_metric = span_result["micro_overall"]
        span_f1 = span_metric["f1"]

        """Metrik BIO untuk pencatatan eksperimen"""
        bio_macro = bio_result.get("macro_avg", {})
        bio_weighted = bio_result.get("weighted_avg", {})
        bio_micro = bio_result.get("micro_overall", {})

        experiment_results.append({
            "experiment": config["name"],
            "c1": config["c1"],
            "c2": config["c2"],
            "max_iterations": config["max_iterations"],
            "training_time": round(training_time, 2),

            "span_precision": span_metric["precision"],
            "span_recall": span_metric["recall"],
            "span_f1": span_f1,

            "bio_accuracy": bio_result.get("accuracy", 0.0),
            "bio_macro_precision": bio_macro.get("precision", 0.0),
            "bio_macro_recall": bio_macro.get("recall", 0.0),
            "bio_macro_f1": bio_macro.get("f1", 0.0),
            "bio_weighted_f1": bio_weighted.get("f1", 0.0),
            "bio_micro_f1": bio_micro.get("f1", 0.0)
        })

        print(f"\nValidation Exact-match Span F1: {span_f1:.4f}")

        """Pemilihan model hanya berdasarkan span F1"""
        if span_f1 > best_f1:
            best_f1 = span_f1
            best_model = model
            best_config = config.copy()
            best_span_result = span_result
            best_bio_result = bio_result

            print("Menjadi model terbaik sementara.")

    save_experiment_results(experiment_results)

    print("\nKONFIGURASI CRF TERBAIK")
    print(best_config)
    print(f"Best validation exact-match span F1: {best_f1:.4f}")

    return (
        best_model,
        best_config,
        best_span_result,
        best_bio_result,
        experiment_results
    )


def save_best_model(model, config, validation_span_result, validation_bio_result):
    model_path = MODEL_DIR / "best_crf.joblib"

    joblib.dump(model, model_path)

    metadata = {
        "model": "CRF",
        "library": "sklearn-crfsuite",
        "config": config,

        "features": [
            "word.lower",
            "prefix2",
            "prefix3",
            "suffix2",
            "suffix3",
            "is_upper",
            "is_title",
            "is_digit",
            "contains_digit",
            "contains_hyphen",
            "word_shape",
            "previous_context",
            "next_context",
            "BOS",
            "EOS",
            "bias"
        ],

        "entity_types": ENTITY_TYPES,
        "bio_labels": BIO_LABELS,

        "training_label_normalization": True,
        "validation_label_normalization": True,
        "test_label_normalization": True,

        "selection_metric": "validation_exact_match_span_micro_f1",

        "validation_span_result": validation_span_result,
        "validation_bio_result": validation_bio_result,

        "validation_span_f1":validation_span_result["micro_overall"]["f1"]
    }
    
    metadata_path = MODEL_DIR / "crf_metadata.joblib"

    joblib.dump(metadata, metadata_path)

    print("\nModel terbaik berhasil disimpan.")
    print("Model:", model_path)
    print("Metadata:", metadata_path)

    return model_path, metadata_path


def save_span_results(result, filename):
    """Menyimpan hasil evaluasi exact-match span"""
    output_path = RESULT_DIR / filename

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

    print(f"Hasil span disimpan: {output_path}")


def save_bio_results(result, filename):
    """Menyimpan hasil evaluasi token-level BIO."""
    output_path = RESULT_DIR / filename

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

        """Hasil per label"""
        for label in BIO_LABELS:
            if label not in result:
                continue

            metric = result[label]

            writer.writerow({
                "label": label,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "support": metric["support"]
            })

        """Hasil rata-rata"""
        for avg_name in ["macro_avg", "weighted_avg", "micro_overall"]:
            if avg_name not in result:
                continue

            metric = result[avg_name]

            writer.writerow({
                "label": avg_name,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "support": metric.get("support", "")
            })

        if "accuracy" in result:
            writer.writerow({
                "label": "accuracy",
                "precision": "",
                "recall": "",
                "f1": result["accuracy"],
                "support": ""
            })

    print(f"Hasil BIO disimpan: {output_path}")


def load_best_model():
    """Load model terbaik dari file joblib"""
    model_path = MODEL_DIR / "best_crf.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    model = joblib.load(model_path)

    print("\nModel terbaik berhasil dimuat")
    print("Label:", model.classes_)

    return model


def prepare_test_data(dataset, normalize_labels=True):
    test_masked = dataset["test_masked"]
    test_gold = dataset["test"]

    """Validasi jumlah sequence"""
    if len(test_masked) != len(test_gold):
        raise ValueError("Jumlah sequence test masked dan test gold berbeda")

    """Validasi kesesuaian data"""
    for idx, (masked, gold) in enumerate(zip(test_masked, test_gold)):

        """ID split berbeda, tetapi nomor sequence harus sama"""
        masked_index = masked["id"].rsplit("_", 1)[-1]
        gold_index = gold["id"].rsplit("_", 1)[-1]

        if masked_index != gold_index:
            raise ValueError(f"Urutan ID berbeda pada indeks {idx}")

        """Token harus identik"""
        if masked["tokens"] != gold["tokens"]:
            raise ValueError(f"Token masked dan gold berbeda pada indeks {idx}")

        """Jumlah token dan label harus sama"""
        if len(masked["tokens"]) != len(gold["labels"]):
            raise ValueError(f"Jumlah token dan label berbeda pada indeks {idx}")

    """Ekstraksi fitur dari token masked"""
    X_test = [sentence2features(item["tokens"]) for item in test_masked]

    """Mengambil label gold"""
    y_test = []

    for item in test_gold:
        labels = list(item["labels"])

        """Normalisasi label jika diaktifkan"""
        if normalize_labels:
            labels = normalize_bio(labels)

        y_test.append(labels)

    return X_test, y_test


def evaluate_test(model, dataset, normalize_labels=True):
    X_test, y_test = prepare_test_data(dataset, normalize_labels=normalize_labels)

    print("\nJumlah sequence test:", len(X_test))
    print("Jumlah token test:", sum(len(seq) for seq in X_test))

    y_pred, span_result, bio_result = predict_evaluate(model, X_test, y_test)

    """Mengambil token dari test masked"""
    test_tokens = [item["tokens"] for item in dataset["test_masked"]]
    
    test_ids = [item["id"] for item in dataset["test_masked"]]

    """Analisis kesalahan"""
    errors = analyze_dataset(
        tokens_list=test_tokens,
        y_true=y_test,
        y_pred=y_pred,
        sentence_ids=test_ids
    )
    
    print_error_summary(errors)
    
    save_error_analysis(errors)

    """Menampilkan hasil span"""
    print_span_metrics(span_result, "HASIL EVALUASI AKHIR TEST - EXACT-MATCH SPAN")

    """Menampilkan hasil BIO"""
    print_bio_metrics(bio_result, "HASIL EVALUASI AKHIR TEST - TOKEN-LEVEL BIO")

    """Menyimpan hasil evaluasi secara terpisah"""
    save_span_results(span_result, "test_span_results.csv")
    save_bio_results(bio_result, "test_bio_results.csv")

    return y_pred, span_result, bio_result


def train():
    """Training, seleksi model, dan evaluasi akhir test"""
    print("\nMEMUAT DATASET")

    dataset = load_dataset()

    train_sentences = dataset["train"]
    val_sentences = dataset["validation"]

    """Persiapan train"""
    print("\nPREPARE TRAIN")

    X_train, y_train = split_xy(train_sentences, normalize_labels=True)

    """Persiapan validation"""
    print("\nPREPARE VALIDATION")

    X_val, y_val = split_xy(val_sentences, normalize_labels=True)

    """Eksperimen dan seleksi model"""
    (best_model, 
     best_config, 
     best_span_result, 
     best_bio_result, 
     results
     ) = select_best_model(X_train, y_train, X_val, y_val)

    """Menyimpan model terbaik dan kedua hasil validation"""
    save_best_model(
        best_model,
        best_config,
        best_span_result,
        best_bio_result
    )

    """Memuat model terbaik"""
    loaded_model = load_best_model()

    """Evaluasi akhir pada test"""
    y_test_pred, test_span_result, test_bio_result = evaluate_test(loaded_model, dataset)

    return (
        loaded_model,
        best_span_result,
        best_bio_result,
        test_span_result,
        test_bio_result
    )


if __name__ == "__main__":
    
    train()
