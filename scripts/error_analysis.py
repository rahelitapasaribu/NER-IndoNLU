"""
Script untuk menganalisis jenis pelanggaran struktur BIO pada dataset NERP.
Menghitung jumlah error seperti I-X setelah O, setelah tipe entitas berbeda,
dan I-X yang muncul pada awal sequence.
"""

from src.data_loader import load_dataset
from src.annotation import categorize_bio_errors

dataset = load_dataset()

print("\n=== ANALISIS JENIS ERROR BIO ===")

total_errors = {
    "I-X setelah O": 0,
    "I-X setelah tipe berbeda": 0,
    "I-X di awal sequence": 0
}

for split in ["train", "validation"]:
    errors = categorize_bio_errors(dataset[split])

    print(f"\n{split.upper()}")

    for error_type, count in errors.items():
        print(f"{error_type}: {count}")
        total_errors[error_type] += count

print("\nTOTAL ERROR BIO")

for error_type, count in total_errors.items():
    print(f"{error_type}: {count}")

print(f"\nTotal: {sum(total_errors.values())}")