"""
Script pengujian untuk memastikan fungsi pada annotation.py bekerja dengan benar.
Menguji contoh anotasi BIO yang valid dan invalid serta memeriksa
apakah kesalahan struktur BIO dapat terdeteksi.
"""
from src.data_loader import load_dataset
from src.annotation import validate_dataset

dataset = load_dataset()

print("\n=== HASIL VALIDASI ANOTASI ===")

total_data = 0
total_valid = 0
total_invalid = 0

for split in ["train", "validation", "test"]:
    results = validate_dataset(dataset[split])

    valid_count = sum(result["valid"] for result in results)
    invalid_count = len(results) - valid_count

    total_data += len(results)
    total_valid += valid_count
    total_invalid += invalid_count

    print(f"\n{split.upper()}")
    print(f"Total   : {len(results)}")
    print(f"Valid   : {valid_count}")
    print(f"Invalid : {invalid_count}")

    if invalid_count > 0:
        print("\nContoh error:")

        shown = 0

        for result in results:
            if not result["valid"]:
                print(f"- {result['id']}: {result['errors']}")
                shown += 1

                if shown == 5:
                    break

print("\nTOTAL DATASET")
print(f"Total   : {total_data}")
print(f"Valid   : {total_valid}")
print(f"Invalid : {total_invalid}")