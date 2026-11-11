from pathlib import Path
import csv

from spans import bio_to_spans


BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results" / "crf"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


ERROR_TYPES = [
    "missed_entity",
    "false_entity",
    "wrong_entity_type",
    "boundary_error",
    "partial_entity"
]


def calculate_overlap(true_span, pred_span):
    """Menghitung jumlah token yang saling overlap"""
    true_start, true_end, _ = true_span
    pred_start, pred_end, _ = pred_span

    return max(0, min(true_end, pred_end) - max(true_start, pred_start))


def is_partial_entity(true_span, pred_span):
    """Memeriksa apakah salah satu span merupakan bagian dari span lainnya"""
    true_start, true_end, true_type = true_span
    pred_start, pred_end, pred_type = pred_span

    if true_type != pred_type:
        return False

    """Prediksi sepenuhnya berada di dalam gold"""
    pred_inside_true = (true_start <= pred_start and pred_end <= true_end)

    """Gold sepenuhnya berada di dalam prediksi"""
    true_inside_pred = (pred_start <= true_start and true_end <= pred_end)

    return (pred_inside_true or true_inside_pred) and (true_start != pred_start or true_end != pred_end)


def classify_error(true_span, pred_span):
    """Mengklasifikasikan kesalahan berdasarkan batas dan tipe entitas"""
    true_start, true_end, true_type = true_span
    pred_start, pred_end, pred_type = pred_span

    """Batas sama, tipe berbeda"""
    if (
        true_start == pred_start
        and true_end == pred_end
        and true_type != pred_type
    ):
        return "wrong_entity_type"

    """Tipe sama, salah satu span merupakan bagian lainnya"""
    if is_partial_entity(true_span, pred_span):
        return "partial_entity"

    """Overlap tetapi batas tidak sesuai"""
    return "boundary_error"


def analyze_sentence(tokens, true_labels, pred_labels, sentence_id):
    """Analisis kesalahan untuk satu sequence. Menghasilkan daftar kasus kesalahan"""
    if len(tokens) != len(true_labels):
        raise ValueError(f"Jumlah token dan gold label tidak sama: {sentence_id}")

    if len(tokens) != len(pred_labels):
        raise ValueError(f"Jumlah token dan predicted label tidak sama: {sentence_id}")

    true_spans = bio_to_spans(true_labels)
    pred_spans = bio_to_spans(pred_labels)

    """Semua kandidat pasangan yang memiliki overlap"""
    candidates = []

    for true_span in true_spans:
        for pred_span in pred_spans:

            overlap = calculate_overlap(true_span, pred_span)

            if overlap > 0:

                candidates.append({
                    "true_span": true_span,
                    "pred_span": pred_span,
                    "overlap": overlap
                })

    """Prioritaskan pasangan dengan overlap terbesar"""
    candidates.sort(key=lambda item: item["overlap"], reverse=True)

    matched_true = set()
    matched_pred = set()

    errors = []

    def create_record(error_type, true_span=None, pred_span=None):
        """Membuat dictionary untuk satu kasus kesalahan"""
        true_text = ""
        pred_text = ""

        true_type = ""
        pred_type = ""

        true_start = ""
        true_end = ""

        pred_start = ""
        pred_end = ""

        if true_span is not None:
            true_start, true_end, true_type = true_span
            true_text = " ".join(tokens[true_start:true_end])

        if pred_span is not None:
            pred_start, pred_end, pred_type = pred_span
            pred_text = " ".join(tokens[pred_start:pred_end])

        return {
            "sentence_id": sentence_id,
            "error_type": error_type,
            "true_text": true_text,
            "predicted_text": pred_text,
            "true_type": true_type,
            "predicted_type": pred_type,
            "true_start": true_start,
            "true_end": true_end,
            "pred_start": pred_start,
            "pred_end": pred_end,
            "sentence": " ".join(tokens)
        }

    """Pasangkan span yang overlap secara satu-ke-satu"""
    for candidate in candidates:

        true_span = candidate["true_span"]
        pred_span = candidate["pred_span"]

        if true_span in matched_true:
            continue

        if pred_span in matched_pred:
            continue

        matched_true.add(true_span)
        matched_pred.add(pred_span)

        """Jika sama persis, bukan kesalahan"""
        if true_span == pred_span:
            continue

        error_type = classify_error(true_span, pred_span)

        errors.append(create_record(error_type, true_span, pred_span))

    """Gold span yang tidak memiliki pasangan"""
    for true_span in true_spans - matched_true:

        errors.append(create_record("missed_entity", true_span=true_span))

    """Predicted span yang tidak memiliki pasangan"""
    for pred_span in pred_spans - matched_pred:

        errors.append(create_record("false_entity", pred_span=pred_span))

    return errors


def analyze_dataset(tokens_list, y_true, y_pred, sentence_ids=None):
    """Analisis kesalahan seluruh dataset"""
    if not (len(tokens_list) == len(y_true) == len(y_pred)):
        raise ValueError("Jumlah sequence tokens, y_true, dan y_pred berbeda")

    if sentence_ids is None:
        sentence_ids = [f"sentence_{i:06d}" for i in range(len(tokens_list))]

    if len(sentence_ids) != len(tokens_list):
        raise ValueError("Jumlah sentence_ids tidak sesuai")

    all_errors = []

    for tokens, true_labels, pred_labels, sentence_id in zip(tokens_list, y_true, y_pred, sentence_ids):

        errors = analyze_sentence(tokens, true_labels, pred_labels, sentence_id)

        all_errors.extend(errors)

    return all_errors


def summarize_errors(errors):
    """Menghitung jumlah setiap kategori kesalahan"""
    summary = {error_type: 0 for error_type in ERROR_TYPES}

    for error in errors:
        summary[error["error_type"]] += 1

    return summary


def save_error_analysis(errors):
    """Menyimpan hasil analisis kesalahan ke file CSV"""
    output_path = RESULT_DIR / "error_analysis.csv"

    fieldnames = [
        "sentence_id",
        "error_type",
        "true_text",
        "predicted_text",
        "true_type",
        "predicted_type",
        "true_start",
        "true_end",
        "pred_start",
        "pred_end",
        "sentence"
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(errors)

    print("\nFile analisis kesalahan disimpan:", output_path)


def print_error_summary(errors):
    """Mencetak ringkasan kesalahan ke console"""
    summary = summarize_errors(errors)

    print("\nRINGKASAN ANALISIS KESALAHAN")

    for error_type, count in summary.items():

        print(f"{error_type:<25}: {count}")

    print("Total kasus kesalahan:", len(errors))


if __name__ == "__main__":

    print("Gunakan analyze_dataset() setelah memperoleh prediksi test")
