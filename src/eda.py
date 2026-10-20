from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_dataset
from annotation import (validate_sample, extract_entities, categorize_bio_errors)

OUTPUT_DIR = Path("hasil_eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDA_SPLITS = {"train", "validation"}


def save_csv(data, filename):
    """Menyimpan hasil analisis ke file CSV."""
    pd.DataFrame(data).to_csv(
        OUTPUT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )


def word_shape(token):
    """Mengelompokkan bentuk kapitalisasi token."""
    letters = [c for c in token if c.isalpha()]

    if not letters:
        return "tanpa_huruf"
    if all(c.isupper() for c in letters):
        return "kapital_semua"
    if all(c.islower() for c in letters):
        return "huruf_kecil"
    if token.istitle():
        return "kapital_awal"

    return "campuran"


def analyze_quality(dataset):
    summary = []
    issue_rows = []
    duplicate_counter = Counter()
    sequence_rows = []

    for split, records in dataset.items():
        lengths = []

        for sample in records:
            tokens = sample["tokens"]
            labels = sample["labels"]
            sample_id = sample["id"]

            lengths.append(len(tokens))

            # Catat sequence untuk analisis duplikasi
            key = (tuple(tokens), tuple(labels))
            duplicate_counter[key] += 1

            sequence_rows.append({
                "split": split,
                "id": sample_id,
                "text": " ".join(tokens),
                "n_tokens": len(tokens),
            })

            result = validate_sample(sample)

            for error in result["errors"]:
                issue_rows.append({
                    "split": split,
                    "id": sample_id,
                    "error": error,
                })

        summary.append({
            "split": split,
            "jumlah_sequence": len(records),
            "total_token": sum(lengths),
            "rata_rata_token": (
                round(sum(lengths) / len(lengths), 2)
                if lengths else 0
            ),
            "minimum_token": min(lengths) if lengths else 0,
            "maksimum_token": max(lengths) if lengths else 0,
            "token_label_sesuai": sum(
                len(s["tokens"]) == len(s["labels"])
                for s in records
            ),
        })

    duplicates = [
        {
            "text": " ".join(tokens),
            "labels": " ".join(labels),
            "jumlah": count,
        }
        for (tokens, labels), count in duplicate_counter.items()
        if count > 1
    ]

    outliers = []

    for split in dataset:
        lengths = pd.Series([
            row["n_tokens"]
            for row in sequence_rows
            if row["split"] == split
        ])

        if len(lengths) < 4:
            continue

        q1 = lengths.quantile(0.25)
        q3 = lengths.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for row in sequence_rows:
            if row["split"] != split:
                continue

            if row["n_tokens"] < lower or row["n_tokens"] > upper:
                outliers.append(row)

    save_csv(summary, "01_struktur_split.csv")
    save_csv(issue_rows, "02_kualitas_anotasi.csv")
    save_csv(duplicates, "03_duplikasi_sequence.csv")
    save_csv(outliers, "04_outlier_sequence.csv")

    print("\n=== STRUKTUR DATA ===")
    print(pd.DataFrame(summary).to_string(index=False))

    print(f"\nJumlah temuan validasi anotasi: {len(issue_rows)}")
    print(f"Jumlah kelompok sequence duplikat: {len(duplicates)}")
    print(f"Jumlah sequence outlier: {len(outliers)}")

    return sequence_rows


def analyze_labels(dataset):
    token_rows = []
    span_rows = []
    capitalization_rows = []
    ambiguous = {}

    for split in EDA_SPLITS:
        for sample in dataset.get(split, []):
            tokens = sample["tokens"]
            labels = sample["labels"]

            for token, label in zip(tokens, labels):
                entity_type = (
                    label[2:]
                    if label.startswith(("B-", "I-"))
                    else None
                )

                token_rows.append({
                    "split": split,
                    "token": token,
                    "label": label,
                    "entity_type": entity_type,
                })

                if entity_type:
                    capitalization_rows.append({
                        "split": split,
                        "entity_type": entity_type,
                        "word_shape": word_shape(token),
                    })

            entities = extract_entities(tokens, labels)

            for entity in entities:
                entity_text = entity["text"]
                entity_type = entity["type"]

                span_rows.append({
                    "split": split,
                    "id": sample["id"],
                    "entity": entity_text,
                    "entity_type": entity_type,
                    "panjang_token": len(entity_text.split()),
                })

                key = entity_text.casefold()
                ambiguous.setdefault(key, set()).add(entity_type)

    token_df = pd.DataFrame(token_rows)
    span_df = pd.DataFrame(span_rows)

    label_counts = (
        token_df.groupby(["split", "label"])
        .size()
        .reset_index(name="jumlah_token")
    )

    save_csv(label_counts, "05_distribusi_label.csv")

    o_summary = []

    for split, group in token_df.groupby("split"):
        total = len(group)
        jumlah_o = (group["label"] == "O").sum()

        o_summary.append({
            "split": split,
            "total_token": total,
            "jumlah_O": int(jumlah_o),
            "proporsi_O_persen": round(
                jumlah_o / total * 100, 2
            ) if total else 0,
        })

    save_csv(o_summary, "06_proporsi_label_O.csv")

    if not span_df.empty:
        span_counts = (
            span_df.groupby(["split", "entity_type"])
            .size()
            .reset_index(name="jumlah_entitas")
        )

        span_lengths = (
            span_df.groupby(["entity_type", "panjang_token"])
            .size()
            .reset_index(name="jumlah_entitas")
        )
    else:
        span_counts = pd.DataFrame(
            columns=["split", "entity_type", "jumlah_entitas"]
        )
        span_lengths = pd.DataFrame(
            columns=["entity_type", "panjang_token", "jumlah_entitas"]
        )

    save_csv(span_counts, "07_jumlah_entitas.csv")
    save_csv(span_lengths, "08_panjang_entitas.csv")

    cap_df = pd.DataFrame(capitalization_rows)

    if not cap_df.empty:
        cap_summary = (
            cap_df.groupby(["split", "entity_type", "word_shape"])
            .size()
            .reset_index(name="jumlah_token")
        )
    else:
        cap_summary = pd.DataFrame(
            columns=["split", "entity_type", "word_shape", "jumlah_token"]
        )

    save_csv(cap_summary, "09_kapitalisasi.csv")

    ambiguous_rows = [
        {
            "permukaan_entitas": surface,
            "tipe_entitas": ", ".join(sorted(types)),
        }
        for surface, types in ambiguous.items()
        if len(types) > 1
    ]

    save_csv(ambiguous_rows, "10_entitas_ambigu.csv")

    boundary = token_df[
        token_df["entity_type"].notna()
        & token_df["token"].astype(str).str.fullmatch(r"[^\w\s]+")
    ]

    save_csv(
        boundary.to_dict("records"),
        "11_kandidat_boundary.csv",
    )

    bio_errors = []

    for split in EDA_SPLITS:
        for sample in dataset.get(split, []):
            for error in validate_sample(sample)["errors"]:
                if (
                    "setelah" in error
                    or "di awal sequence" in error
                ):
                    bio_errors.append({
                        "split": split,
                        "id": sample["id"],
                        "error": error,
                    })

    save_csv(bio_errors, "12_pelanggaran_BIO.csv")

    if not token_df.empty:
        (
            token_df.groupby("label")
            .size()
            .sort_values(ascending=False)
            .plot(kind="bar", figsize=(10, 5))
        )

        plt.title("Distribusi Label BIO - Train dan Validation")
        plt.xlabel("Label")
        plt.ylabel("Jumlah token")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "distribusi_label.png",
            dpi=150,
        )
        plt.close()

    print("\n=== DISTRIBUSI LABEL BIO ===")
    print(label_counts.to_string(index=False))

    print("\n=== PROPORSI LABEL O ===")
    print(pd.DataFrame(o_summary).to_string(index=False))

    print("\n=== JUMLAH ENTITAS PER TIPE ===")
    print(span_counts.to_string(index=False))

    print(f"\nJumlah entitas: {len(span_df)}")
    print(f"Jumlah pelanggaran BIO: {len(bio_errors)}")
    print(f"Kandidat entitas ambigu: {len(ambiguous_rows)}")
    print(f"Kandidat boundary tanda baca: {len(boundary)}")


def main():
    print("=== EDA DATASET NER INDONLU ===")

    dataset = load_dataset()

    analyze_quality(dataset)

    analyze_labels(dataset)

    print("\n=== RINGKASAN ===")

    for split, records in dataset.items():
        print(f"{split}: {len(records)} sequence")

    print(f"\nHasil EDA tersimpan di: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
