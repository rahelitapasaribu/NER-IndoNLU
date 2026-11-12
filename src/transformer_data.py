import unicodedata
from torch.utils.data import Dataset
from transformers import DataCollatorForTokenClassification
from src.spans import normalize_bio

def clean_record(record, include_labels=True):
    tokens = record["tokens"]
    labels = record.get("labels", [])
    if include_labels and len(tokens) != len(labels):
        raise ValueError(f"Jumlah token dan label berbeda: {record['id']}")

    clean_tokens, clean_labels = [], []
    for i, token in enumerate(tokens):
        token = "".join(c for c in str(token) if unicodedata.category(c) not in {"Cf", "Cc"}).strip()
        if not token:
            if include_labels and labels[i] != "O":
                raise ValueError(f"Token entitas kosong: {record['id']}")
            continue
        clean_tokens.append(token)
        if include_labels:
            clean_labels.append(labels[i])

    if include_labels:
        clean_labels = normalize_bio(clean_labels)

    result = record.copy()
    result["tokens"] = clean_tokens
    result["labels"] = clean_labels if include_labels else []
    return result


def build_units(labels):
    units, i = [], 0
    while i < len(labels):
        label = labels[i]
        if label == "O":
            units.append([i])
            i += 1
        else:
            prefix, entity_type = label.split("-", 1)
            if prefix != "B":
                raise ValueError(f"Urutan BIO tidak valid pada indeks {i}: {label}")
            unit = [i]
            i += 1
            while i < len(labels) and labels[i] == f"I-{entity_type}":
                unit.append(i)
                i += 1
            units.append(unit)
    return units


class NERDataset(Dataset):
    def __init__(self, records, tokenizer, label2id, max_length=256, include_labels=True):
        self.features, self.word_ids, self.records = [], [], []
        self.include_labels = include_labels

        if not tokenizer.is_fast:
            raise ValueError("Gunakan fast tokenizer.")
        if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
            raise ValueError("Special token CLS/SEP tidak ditemukan.")

        content_limit = max_length - 2

        for original in records:
            record = clean_record(original, include_labels)
            tokens, labels = record["tokens"], record["labels"]
            if not tokens:
                continue

            encoding = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False, truncation=False)
            all_ids = encoding["input_ids"]
            word_groups = [[] for _ in tokens]

            for pos, word_id in enumerate(encoding.word_ids()):
                if word_id is not None:
                    word_groups[word_id].append(pos)

            if any(not group for group in word_groups):
                raise ValueError(f"Ada token yang tidak memiliki subword: {record['id']}")

            units = build_units(labels) if include_labels else [[i] for i in range(len(tokens))]
            chunks, current, length = [], [], 0

            for unit in units:
                unit_length = sum(len(word_groups[i]) for i in unit)
                if unit_length > content_limit:
                    raise ValueError(f"Satu unit melebihi batas input: {record['id']}")
                if current and length + unit_length > content_limit:
                    chunks.append(current)
                    current, length = [], 0
                current.extend(unit)
                length += unit_length

            if current:
                chunks.append(current)

            for chunk in chunks:
                start = word_groups[chunk[0]][0]
                end = word_groups[chunk[-1]][-1] + 1
                chunk_ids = all_ids[start:end]

                input_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)

                chunk_word_ids = [None]
                for word_idx in chunk:
                    chunk_word_ids.extend([word_idx] * len(word_groups[word_idx]))
                chunk_word_ids.append(None)

                if len(input_ids) > max_length or len(input_ids) != len(chunk_word_ids):
                    raise ValueError(f"Chunk atau alignment tidak valid: {record['id']}")

                feature = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }

                if include_labels:
                    aligned = [-100]
                    for word_idx in chunk:
                        aligned.append(label2id[labels[word_idx]])
                        aligned.extend([-100] * (len(word_groups[word_idx]) - 1))
                    aligned.append(-100)
                    feature["labels"] = aligned

                self.features.append(feature)
                self.word_ids.append(chunk_word_ids)
                self.records.append({
                    "id": record["id"],
                    "tokens": tokens,
                    "labels": labels,
                    "chunk_words": chunk
                })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def build_collator(tokenizer):
    return DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )


def map_chunk_predictions(logits, word_ids):
    predictions = logits.argmax(axis=-1)
    result = {}
    for pos, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in result:
            result[word_id] = int(predictions[pos])
    return result
