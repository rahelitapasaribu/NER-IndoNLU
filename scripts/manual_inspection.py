"""
Script untuk mengambil 20 sampel data train secara acak dengan seed tetap.
Menampilkan kalimat, entitas, dan hasil validasi BIO sebagai bahan
inspeksi manual terhadap kualitas anotasi dataset.
"""
from src.data_loader import load_dataset
from src.annotation import sample_for_manual_inspection


dataset = load_dataset()

samples = sample_for_manual_inspection(
    dataset["train"],
    n=20,
    seed=42
)

print("\n=== 20 SAMPEL INSPEKSI MANUAL ===")

for i, sample in enumerate(samples, start=1):
    print(f"\n--- Sampel {i} ---")
    print("ID       :", sample["id"])
    print("Kalimat  :", " ".join(sample["tokens"]))
    print("BIO Valid:", sample["bio_valid"])

    print("Entities :")

    if sample["entities"]:
        for entity in sample["entities"]:
            print(
                f"  - {entity['text']} "
                f"({entity['type']})"
            )
    else:
        print("  - Tidak ada entity")

    if sample["errors"]:
        print("Errors   :", sample["errors"])