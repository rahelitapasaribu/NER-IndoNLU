import requests

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "indobenchmark/indonlu/master/dataset/nerp_ner-prosa/"
)

FILES = {
    "train": "train_preprocess.txt",
    "validation": "valid_preprocess.txt",
    "test": "test_preprocess_masked_label.txt"
}

def download_split(split):

    if split not in FILES:
        raise ValueError(
            f"Split tidak valid: {split}. "
            f"Gunakan salah satu dari {list(FILES.keys())}."
        )

    url = BASE_URL + FILES[split]

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return response.text

def parse_dataset(text, split):
    sentences = []
    raw_sentences = text.strip().split("\n\n")

    for idx, raw_sentence in enumerate(raw_sentences):

        tokens = []
        labels = []

        for line in raw_sentence.splitlines():
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1]

                tokens.append(token)
                labels.append(label)

        if tokens:
            sentences.append({
                "id": f"{split}_{idx:06d}",
                "tokens": tokens,
                "labels": labels
            })

    return sentences

def filter_dataset(data):
    filtered_data = []

    for item in data:
        if not item.get("id"):
            continue
        if not item.get("tokens"):
            continue
        if not item.get("labels"):
            continue
        if len(item["tokens"]) != len(item["labels"]):
            continue

        filtered_data.append(item)

    return filtered_data

def load_dataset():
    dataset = {}

    for split in FILES:
        print(f"Memuat split: {split}")

        text = download_split(split)
        data = parse_dataset(text, split)
        data = filter_dataset(data)

        dataset[split] = data

        print(f"Jumlah data setelah filter: {len(data)}")

    return dataset

if __name__ == "__main__":

    dataset = load_dataset()

    print("\nJumlah data:")
    for split, data in dataset.items():
        print(f"{split}: {len(data)}")

    print("\nContoh data:")
    print(dataset["train"][0])

    print("\nLabel:")
    all_labels = []

    for split_data in dataset.values():
        for item in split_data:
            all_labels.extend(item["labels"])

    unique_labels = sorted(set(all_labels))

    for label in unique_labels:
        print(label)