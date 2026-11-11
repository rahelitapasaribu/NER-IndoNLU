
def normalize_bio(labels):
    """Memperbaiki label I-TYPE yang tidak memiliki entitas sebelumnya dengan tipe yang sama"""
    normalized = []
    previous_type = None
    previous_prefix = "O"

    for label in labels:
        if label == "O":
            normalized.append("O")
            previous_type = None
            previous_prefix = "O"
            continue

        if "-" not in label:
            normalized.append("O")
            previous_type = None
            previous_prefix = "O"
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix not in {"B", "I"}:
            normalized.append("O")
            previous_type = None
            previous_prefix = "O"
            continue

        if (
            prefix == "I"
            and not (
                previous_prefix in {"B", "I"}
                and previous_type == entity_type
            )
        ):
            prefix = "B"

        normalized.append(f"{prefix}-{entity_type}")
        previous_type = entity_type
        previous_prefix = prefix

    return normalized


def bio_to_spans(labels):
    """
    Mengubah label BIO menjadi set: (start, end, entity_type), dengan end eksklusif
    Kebijakan untuk label BIO tidak valid:
    - I-TYPE tanpa span aktif dianggap awal entitas.
    - I-TYPE dengan tipe berbeda dari span aktif menutup span sebelumnya dan memulai span baru.
    - Label dengan format tidak valid memunculkan ValueError.
    """
    spans = set()
    start = None
    entity_type = None

    def close_span(end):
        nonlocal start, entity_type

        if start is not None:
            spans.add((start, end, entity_type))
            start = None
            entity_type = None

    for i, label in enumerate(labels):
        if label == "O":
            close_span(i)
            continue

        if "-" not in label:
            raise ValueError(f"Format label BIO tidak valid: {label}")

        prefix, current_type = label.split("-", 1)

        if prefix not in {"B", "I"} or not current_type:
            raise ValueError(f"Format label BIO tidak valid: {label}")

        if prefix == "B":
            close_span(i)
            start = i
            entity_type = current_type

        elif prefix == "I":
            if start is None or current_type != entity_type:
                close_span(i)
                start = i
                entity_type = current_type

    close_span(len(labels))

    return spans
