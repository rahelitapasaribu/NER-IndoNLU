def normalize_bio(labels):
    """
    Menormalisasi urutan BIO.

    I-TYPE yang tidak memiliki entitas sebelumnya
    dengan tipe sama akan diubah menjadi B-TYPE.
    """

    normalized = []

    previous_type = None
    previous_prefix = "O"

    for label in labels:

        if label == "O":
            normalized.append("O")
            previous_type = None
            previous_prefix = "O"
            continue

        if not isinstance(label, str) or "-" not in label:
            raise ValueError(f"Label BIO tidak valid: {label}")

        prefix, entity_type = label.split("-", 1)

        if prefix not in {"B", "I"} or not entity_type:
            raise ValueError(f"Label BIO tidak valid: {label}")

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
    Mengubah BIO menjadi set:
    (start, end, entity_type).

    End menggunakan indeks eksklusif.

    I-TYPE tanpa entitas sebelumnya dianggap
    sebagai awal entitas.

    I-TYPE dengan tipe berbeda akan memulai
    entitas baru.
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

        if not isinstance(label, str) or "-" not in label:
            raise ValueError(f"Format BIO tidak valid: {label}")

        prefix, current_type = label.split("-", 1)

        if prefix not in {"B", "I"} or not current_type:
            raise ValueError(f"Format BIO tidak valid: {label}")

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
