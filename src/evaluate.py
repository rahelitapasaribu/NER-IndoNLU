from collections import defaultdict


ENTITY_TYPES = ["PPL", "PLC", "IND", "EVT", "FNB"]

BIO_LABELS = [
    "O",
    "B-PPL", "I-PPL",
    "B-PLC", "I-PLC",
    "B-IND", "I-IND",
    "B-EVT", "I-EVT",
    "B-FNB", "I-FNB"
]


def evaluate_bio(y_true, y_pred):
    """Evaluasi token-level berdasarkan 11 label BIO. Menghitung precision, recall, F1, support untuk setiap label dan micro overall."""
    if len(y_true) != len(y_pred):
        raise ValueError("Jumlah sequence berbeda")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for true_seq, pred_seq in zip(y_true, y_pred):
        if len(true_seq) != len(pred_seq):
            raise ValueError("Jumlah token dan prediksi berbeda")

        for true, pred in zip(true_seq, pred_seq):
            if true not in BIO_LABELS:
                raise ValueError(f"Label asli tidak dikenal: {true}")

            if pred not in BIO_LABELS:
                raise ValueError(f"Label prediksi tidak dikenal: {pred}")

            support[true] += 1

            if true == pred:
                tp[true] += 1
            else:
                fn[true] += 1
                fp[pred] += 1

    results = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in BIO_LABELS:
        t = tp[label]
        f_pos = fp[label]
        f_neg = fn[label]

        precision = (t / (t + f_pos) if t + f_pos else 0.0)
        recall = (t / (t + f_neg) if t + f_neg else 0.0)
        f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0)

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[label],
            "tp": t,
            "fp": f_pos,
            "fn": f_neg
        }

        total_tp += t
        total_fp += f_pos
        total_fn += f_neg

    precision = (total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0)
    recall = (total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0)
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    results["micro_overall"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

    # Macro average: rata-rata seluruh label tanpa bobot
    macro_precision = sum(results[label]["precision"] for label in BIO_LABELS) / len(BIO_LABELS)
    macro_recall = sum(results[label]["recall"] for label in BIO_LABELS) / len(BIO_LABELS)
    macro_f1 = sum(results[label]["f1"] for label in BIO_LABELS) / len(BIO_LABELS)

    results["macro_avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }

    # Weighted average: rata-rata berdasarkan support
    total_support = sum(support.values())

    if total_support:
        weighted_precision = sum(results[label]["precision"] * support[label] for label in BIO_LABELS) / total_support
        weighted_recall = sum(results[label]["recall"] * support[label] for label in BIO_LABELS) / total_support
        weighted_f1 = sum(results[label]["f1"] * support[label] for label in BIO_LABELS) / total_support

    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    results["weighted_avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "support": total_support
    }

    # Accuracy token-level
    total_tokens = sum(support.values())
    correct_tokens = sum(tp.values())

    results["accuracy"] = (correct_tokens / total_tokens if total_tokens else 0.0)

    return results


def evaluate_spans(true_spans, pred_spans, entity_types=None):
    """
    Evaluasi exact-match entity span.

    Input:
        true_spans: list of sets (start, end, type)
        pred_spans: list of sets (start, end, type)

    Output:
        Precision, Recall, F1, TP, FP, FN per kategori dan micro overall.
    """
    if len(true_spans) != len(pred_spans):
        raise ValueError("Jumlah true_spans dan pred_spans harus sama")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    observed_types = set()

    for true, pred in zip(true_spans, pred_spans):

        true = set(true)
        pred = set(pred)

        for span in true | pred:

            if not isinstance(span, tuple) or len(span) != 3:
                raise ValueError(f"Span tidak valid: {span}")

            start, end, entity_type = span

            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"Indeks span harus integer: {span}")

            if start < 0 or end <= start:
                raise ValueError(f"Batas span tidak valid: {span}")

            if not isinstance(entity_type, str) or not entity_type:
                raise ValueError(f"Tipe entitas tidak valid: {span}")

            observed_types.add(entity_type)

        # Exact match
        for span in true & pred:
            tp[span[2]] += 1

        # FP
        for span in pred - true:
            fp[span[2]] += 1

        # FN
        for span in true - pred:
            fn[span[2]] += 1

    if entity_types is None:
        entity_types = sorted(observed_types)

    else:
        entity_types = sorted(set(entity_types) | observed_types)

    results = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entity_type in entity_types:

        t = tp[entity_type]
        f_pos = fp[entity_type]
        f_neg = fn[entity_type]

        precision = (t / (t + f_pos) if t + f_pos else 0.0)
        recall = (t / (t + f_neg) if t + f_neg else 0.0)
        f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0)

        results[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": t,
            "fp": f_pos,
            "fn": f_neg
        }

        total_tp += t
        total_fp += f_pos
        total_fn += f_neg

    precision_overall = (total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0)
    recall_overall = (total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0)
    f1_overall = (2 * precision_overall * recall_overall / (precision_overall + recall_overall) if precision_overall + recall_overall else 0.0)

    results["micro_overall"] = {
        "precision": precision_overall,
        "recall": recall_overall,
        "f1": f1_overall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

    return results