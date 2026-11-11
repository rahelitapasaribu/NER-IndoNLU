from collections import defaultdict


def evaluate_spans(true_spans, pred_spans, entity_types=None):
    """
    Mengevaluasi prediksi entitas berdasarkan span
    Mengembalikan metrik precision, recall, dan F1-score untuk setiap tipe entitas
    """

    if len(true_spans) != len(pred_spans):
        raise ValueError("Jumlah true_spans dan pred_spans harus sama")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    observed_entity_types = set()

    for true, pred in zip(true_spans, pred_spans):
        true = set(true)
        pred = set(pred)

        for span in true | pred:
            if len(span) != 3:
                raise ValueError(f"Span tidak valid: {span}. Span harus berupa tuple (start, end, entity_type)")
            observed_entity_types.add(span[2])

        for span in true & pred:
            tp[span[2]] += 1

        for span in pred - true:
            fp[span[2]] += 1

        for span in true - pred:
            fn[span[2]] += 1

    if entity_types is None:
        entity_types = sorted(observed_entity_types)
    else:
        entity_types = sorted(set(entity_types) | observed_entity_types)

    results = {}
    total_tp =  0
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
            "fn": f_neg,
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
        "fn": total_fn,
    }

    return results