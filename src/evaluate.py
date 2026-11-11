
from collections import defaultdict


ENTITY_TYPES = ["PPL", "PLC", "IND", "EVT", "FNB"]

def evaluate_spans(true_spans, pred_spans, entity_types=None):
    """
    Evaluasi exact-match entity span.

    Input:
        true_spans: list of sets (start, end, type)
        pred_spans: list of sets (start, end, type)

    Output:
        Precision, Recall, F1, TP, FP, FN
        per kategori dan micro overall.
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

        # False Positive
        for span in pred - true:
            fp[span[2]] += 1

        # False Negative
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

    precision_overall = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )

    recall_overall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )

    f1_overall = (
        2 * precision_overall * recall_overall
        / (precision_overall + recall_overall)
        if precision_overall + recall_overall else 0.0
    )

    results["micro_overall"] = {
        "precision": precision_overall,
        "recall": recall_overall,
        "f1": f1_overall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

    return results
