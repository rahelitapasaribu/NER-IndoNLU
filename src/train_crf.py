from pathlib import Path
import joblib
import sklearn_crfsuite

from data_loader import load_dataset
from crf_features import sentence2features
from spans import normalize_bio, bio_to_spans
from evaluate import evaluate_spans


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "crf"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def split_xy(sentences, normalize_labels=False):
    """Mengubah data kalimat menjadi fitur X dan label y"""
    X = []
    y = []

    for sentence in sentences:
        tokens = sentence["tokens"]
        labels = list(sentence["labels"])

        """Ekstraksi fitur CRF dari kalimat"""
        if normalize_labels:
            labels = normalize_bio(labels)

        X.append(sentence2features(tokens))
        y.append(labels)

    return X, y

def predict_evaluate(model, X, y_true):
    """Memprediksi label dan mengevaluasi entitas berdasarkan span"""
    y_pred = model.predict(X)

    true_spans = [bio_to_spans(seq) for seq in y_true]
    pred_spans = [bio_to_spans(seq) for seq in y_pred]

    result = evaluate_spans(true_spans, pred_spans)

    return y_pred, result

def train():
    """Melatih model dengan data train dan menvalidasi model terbaik dengan data validation"""
    dataset = load_dataset()

    train_sentences = dataset["train"]
    val_sentences = dataset["validation"]

    """Ekstraksi fitur dan pengambilan label BIO"""
    X_train, y_train = split_xy(train_sentences, normalize_labels=True)
    X_val, y_val = split_xy(val_sentences, normalize_labels=False)

    """Konfigurasi model CRF"""
    config = {
        "algorithm": "lbfgs",
        "max_iterations": 200,
        "c1": 0.1,
        "c2": 0.01,
        "all_possible_transitions": True
    }

    crf = sklearn_crfsuite.CRF(**config)

    """Melatih model CRF"""
    crf.fit(X_train, y_train)

    print("\nModel CRF berhasil dilatih")

    """Mengevaluasi model dengan data validation"""
    _, validation_result = predict_evaluate(crf, X_val, y_val)

    print("\nHasil Evaluasi Model CRF dengan Data Validation")
    print(validation_result)

    """Ambil span F1 dari hasil evaluasi"""
    val_f1 = validation_result["micro_overall"]["f1"]

    print(f"\nF1 Score pada Data Validation: {val_f1:.4f}")

    """Menyimpan model CRF ke file"""
    model_path = MODEL_DIR / "crf_model.joblib"
    joblib.dump(crf, model_path)

    """Simpan informasi konfigurasi model ke file"""
    metadata = {
        "algorithm": config["algorithm"],
        "max_iterations": config["max_iterations"],
        "c1": config["c1"],
        "c2": config["c2"],
        "all_possible_transitions": config["all_possible_transitions"],
        "validation_result": validation_result,
        "validation_f1": val_f1
    }

    metadata_path = MODEL_DIR / "crf_metadata.joblib"
    joblib.dump(metadata, metadata_path)

    print(f"\nModel CRF disimpan ke file: {model_path}")
    print(f"Metadata disimpan ke file: {metadata_path}")

    """Uji pemuatan model dari file"""
    loaded_crf = joblib.load(model_path)

    print(f"\nModel CRF berhasil dimuat dari file: {model_path}")
    print(f"Label yang dipelajari: {loaded_crf.classes_}")

    return loaded_crf, validation_result

if __name__ == "__main__":
    train()


