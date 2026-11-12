from pathlib import Path
import joblib
import time
import csv

import sklearn_crfsuite

from src.data_loader import load_dataset
from src.crf_features import sentence2features
from src.spans import normalize_bio, bio_to_spans
from src.evaluate import evaluate_spans, ENTITY_TYPES
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

            total_changed += sum(
                original != normalized
                for original, normalized
                in zip(original_labels, labels)
            )

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
    """Melakukan prediksi dan evaluasi span entity berdasarkan label BIO"""
    y_pred = model.predict(X)

    true_spans = [bio_to_spans(seq) for seq in y_true]
    pred_spans = [bio_to_spans(seq) for seq in y_pred]

    result = evaluate_spans(
        true_spans,
        pred_spans,
        entity_types=ENTITY_TYPES
    )

    return y_pred, result


def print_metrics(result, title):
    """Menampilkan metrik evaluasi span entity"""
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
    """"Menyimpan hasil eksperimen ke file CSV"""
    output_path = RESULT_DIR / "validation_results.csv"

    fieldnames = [
        "experiment",
        "c1",
        "c2",
        "max_iterations",
        "training_time",
        "precision",
        "recall",
        "f1"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:

            writer.writerow(result)

    print(f"\nHasil eksperimen disimpan: {output_path}")


def select_best_model(X_train, y_train, X_val, y_val):
    """Melakukan eksperimen dengan berbagai konfigurasi dan memilih model terbaik berdasarkan F1 score pada validation"""
    best_model = None
    best_config = None
    best_result = None
    best_f1 = -1.0

    experiment_results = []

    for config in CONFIGURATIONS:

        print(f"\nTRAINING {config['name']}")
        print("Konfigurasi:", config)

        model = build_crf(config)

        start_time = time.perf_counter()

        """Training hanya menggunakan train"""
        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        print(f"Waktu training: {training_time:.2f} detik")
        print("Jumlah label yang dipelajari:", len(model.classes_))

        """Evaluasi menggunakan validation"""
        _, validation_result = predict_evaluate(model, X_val, y_val)

        print_metrics(validation_result, f"HASIL VALIDASI {config['name']}")

        val_metric = validation_result["micro_overall"]

        val_f1 = val_metric["f1"]

        experiment_results.append({
            "experiment": config["name"],
            "c1": config["c1"],
            "c2": config["c2"],
            "max_iterations": config["max_iterations"],
            "training_time": round(training_time, 2),
            "precision": val_metric["precision"],
            "recall": val_metric["recall"],
            "f1": val_f1
        })

        print(f"\nValidation span F1: {val_f1:.4f}")

        """Memilih konfigurasi terbaik"""
        if val_f1 > best_f1:

            best_f1 = val_f1
            best_model = model
            best_config = config.copy()
            best_result = validation_result

            print("Menjadi model terbaik sementara")

    save_experiment_results(experiment_results)

    print("\nKONFIGURASI CRF TERBAIK")
    print(best_config)
    print(f"Best validation F1: {best_f1:.4f}")

    return (
        best_model,
        best_config,
        best_result,
        experiment_results
    )


def save_best_model(model, config, validation_result):
    """Menyimpan model terbaik dan metadata ke file joblib"""
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
        "training_label_normalization": True,
        "validation_label_normalization": False,
        "validation_result": validation_result,
        "validation_f1": validation_result["micro_overall"]["f1"]
    }

    metadata_path = MODEL_DIR / "crf_metadata.joblib"

    joblib.dump(metadata, metadata_path)

    print("\nModel terbaik berhasil disimpan")
    print("Model:", model_path)
    print("Metadata:", metadata_path)

    return model_path, metadata_path


def load_best_model():
    """Load model terbaik dari file joblib"""
    model_path = MODEL_DIR / "best_crf.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    model = joblib.load(model_path)

    print("\nModel terbaik berhasil dimuat")
    print("Label:", model.classes_)

    return model


def prepare_test_data(dataset):
    """Mempersiapkan data test untuk evaluasi akhir"""
    test_masked = dataset["test_masked"]
    test_gold = dataset["test"]

    if len(test_masked) != len(test_gold):
        raise ValueError("Jumlah sequence test masked dan test gold berbeda")

    for idx, (masked, gold) in enumerate(zip(test_masked, test_gold)):

        """ID split berbeda, tetapi nomor sequence harus sama"""
        masked_index = masked["id"].rsplit("_", 1)[-1]
        gold_index = gold["id"].rsplit("_", 1)[-1]

        if masked_index != gold_index:
            raise ValueError(f"Urutan ID berbeda pada indeks {idx}")

        """Token harus identik"""
        if masked["tokens"] != gold["tokens"]:
            raise ValueError(f"Token masked dan gold berbeda pada indeks {idx}")

        if len(masked["tokens"]) != len(gold["labels"]):
            raise ValueError(f"Jumlah token dan label berbeda pada indeks {idx}")

    X_test = [sentence2features(item["tokens"]) for item in test_masked]

    y_test = [list(item["labels"]) for item in test_gold]

    return X_test, y_test


def evaluate_test(model, dataset):
    """Melakukan evaluasi akhir pada data test"""
    X_test, y_test = prepare_test_data(dataset)

    print("\nJumlah sequence test:", len(X_test))
    print("Jumlah token test:", sum(len(seq) for seq in X_test))

    """Test tidak dinormalisasi"""
    y_pred, test_result = predict_evaluate(model, X_test, y_test)

    """Mengambil token dari test masked"""
    test_tokens = [item["tokens"] for item in dataset["test_masked"]]

    """Mengambil ID sequence"""
    test_ids = [item["id"] for item in dataset["test_masked"]]

    """Analisis kesalahan"""
    errors = analyze_dataset(
        tokens_list=test_tokens,
        y_true=y_test,
        y_pred=y_pred,
        sentence_ids=test_ids
    )

    """Menampilkan ringkasan"""
    print_error_summary(errors)

    """Menyimpan detail kesalahan"""
    save_error_analysis(errors)

    print_metrics(test_result, "HASIL EVALUASI AKHIR TEST CRF")

    """Simpan hasil evaluasi test"""
    output_path = RESULT_DIR / "test_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "entity",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn"
        ])

        for entity in ENTITY_TYPES + ["micro_overall"]:

            metric = test_result[entity]

            writer.writerow([
                entity,
                metric["precision"],
                metric["recall"],
                metric["f1"],
                metric["tp"],
                metric["fp"],
                metric["fn"]
            ])

    print("\nHasil test disimpan:", output_path)

    return y_pred, test_result


def train():
    """Melakukan training, seleksi model, dan evaluasi akhir test"""
    print("\nMEMUAT DATASET")

    dataset = load_dataset()

    train_sentences = dataset["train"]
    val_sentences = dataset["validation"]

    """Memisahkan fitur dan label untuk train dan validation"""
    print("\nPREPARE TRAIN")

    X_train, y_train = split_xy(train_sentences, normalize_labels=True)

    """Memisahkan fitur dan label untuk validation"""
    print("\nPREPARE VALIDATION")

    X_val, y_val = split_xy(val_sentences, normalize_labels=False)

    """Melakukan eksperimen dengan berbagai konfigurasi dan memilih model terbaik berdasarkan F1 score pada validation"""
    best_model, best_config, best_result, results = (select_best_model(X_train, y_train, X_val, y_val))

    """Menyimpan model terbaik dan metadata ke file joblib"""
    save_best_model(best_model, best_config, best_result)


    """Load model terbaik dari file joblib"""
    loaded_model = load_best_model()

    """Melakukan evaluasi akhir pada data test"""
    y_test_pred, test_result = evaluate_test(loaded_model, dataset)

    return loaded_model, best_result, test_result


if __name__ == "__main__":

    train()
