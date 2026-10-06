"""
Modul untuk melakukan validasi anotasi BIO pada dataset NERP.
Memeriksa kesesuaian token-label, validitas label, konsistensi transisi BIO,
mengekstrak entitas, serta menyediakan fungsi pendukung inspeksi anotasi.
"""
import random

VALID_LABELS = {
    "O",
    "B-EVT", "I-EVT",
    "B-FNB", "I-FNB",
    "B-IND", "I-IND",
    "B-PLC", "I-PLC",
    "B-PPL", "I-PPL",
}

def validate_token_label_length(tokens, labels):
    return len(tokens) == len(labels)

def validate_labels(labels):
    invalid_labels = []

    for label in labels:
        if label not in VALID_LABELS:
            invalid_labels.append(label)

    return invalid_labels

def validate_bio_sequence(tokens, labels):
    errors = []

    for i, label in enumerate(labels):
        if not label.startswith("I-"):
            continue

        entity_type = label[2:]

        if i == 0:
            errors.append(
                f"Token '{tokens[i]}' memiliki label {label} "
                "di awal sequence"
            )
            continue

        previous_label = labels[i - 1]

        valid_previous = {
            f"B-{entity_type}",
            f"I-{entity_type}",
        }

        if previous_label not in valid_previous:
            errors.append(
                f"Token '{tokens[i]}' memiliki label {label} "
                f"setelah {previous_label}"
            )

    return errors

def extract_entities(tokens, labels):
    entities = []

    current_tokens = []
    current_type = None

    for token, label in zip(tokens, labels):

        if label == "O":
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "type": current_type,
                })

                current_tokens = []
                current_type = None

        elif label.startswith("B-"):
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "type": current_type,
                })

            current_tokens = [token]
            current_type = label[2:]

        elif label.startswith("I-"):
            entity_type = label[2:]

            if current_tokens and current_type == entity_type:
                current_tokens.append(token)

            else:
                if current_tokens:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "type": current_type,
                    })

                current_tokens = [token]
                current_type = entity_type

    if current_tokens:
        entities.append({
            "text": " ".join(current_tokens),
            "type": current_type,
        })

    return entities

def validate_sample(sample):
    tokens = sample["tokens"]
    labels = sample["labels"]

    errors = []

    if not validate_token_label_length(tokens, labels):
        errors.append(
            f"Jumlah token ({len(tokens)}) dan "
            f"label ({len(labels)}) tidak sama"
        )

    invalid_labels = validate_labels(labels)

    if invalid_labels:
        errors.append(
            f"Label tidak valid: {sorted(set(invalid_labels))}"
        )

    if len(tokens) == len(labels):
        errors.extend(
            validate_bio_sequence(tokens, labels)
        )

    return {
        "id": sample["id"],
        "valid": len(errors) == 0,
        "errors": errors,
        "entities": extract_entities(tokens, labels),
    }

def validate_dataset(data):
    return [
        validate_sample(sample)
        for sample in data
    ]

def sample_for_manual_inspection(data, n=20, seed=42):
    random.seed(seed)

    n = min(n, len(data))
    samples = random.sample(data, n)

    results = []

    for sample in samples:
        validation = validate_sample(sample)

        results.append({
            "id": sample["id"],
            "tokens": sample["tokens"],
            "labels": sample["labels"],
            "entities": validation["entities"],
            "bio_valid": validation["valid"],
            "errors": validation["errors"],
        })

    return results

def categorize_bio_errors(data):
    error_counts = {
        "I-X setelah O": 0,
        "I-X setelah tipe berbeda": 0,
        "I-X di awal sequence": 0,
    }

    for sample in data:
        labels = sample["labels"]

        for i, label in enumerate(labels):
            if not label.startswith("I-"):
                continue

            entity_type = label[2:]

            if i == 0:
                error_counts["I-X di awal sequence"] += 1
                continue

            previous_label = labels[i - 1]

            if previous_label == "O":
                error_counts["I-X setelah O"] += 1

            elif previous_label not in {
                f"B-{entity_type}",
                f"I-{entity_type}",
            }:
                error_counts[
                    "I-X setelah tipe berbeda"
                ] += 1

    return error_counts