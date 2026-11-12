
import torch

from src.inference.crf_predictor import (
    tokenize_with_offsets,
    split_sentences_with_offsets,
    ENTITY_MAPPING
)
from src.transformer_data import (
    NERDataset,
    map_chunk_predictions
)
from src.spans import bio_to_spans, normalize_bio


def predict_transformer(
    model,
    tokenizer,
    device,
    text,
    max_length=128
):
    if not isinstance(text, str):
        raise TypeError("Input harus berupa string.")

    if not text.strip():
        return []

    model.eval()

    entities = []
    sentence_offsets = split_sentences_with_offsets(text)

    with torch.inference_mode():

        for sentence_start, sentence_end in sentence_offsets:

            sentence_text = text[sentence_start:sentence_end]

            tokens, token_offsets = tokenize_with_offsets(
                sentence_text
            )

            if not tokens:
                continue

            # Menyiapkan data untuk Transformer
            records = [{
                "id": "inference",
                "tokens": tokens,
                "labels": []
            }]

            dataset = NERDataset(
                records=records,
                tokenizer=tokenizer,
                label2id={},
                max_length=max_length,
                include_labels=False
            )

            predicted_labels = ["O"] * len(tokens)

            # Mengambil mapping label dari model
            id2label = {
                int(key): value
                for key, value in model.config.id2label.items()
            }

            # Prediksi setiap chunk
            for idx in range(len(dataset)):

                feature = dataset[idx]
                word_ids = dataset.word_ids[idx]

                inputs = {
                    "input_ids": torch.tensor(
                        [feature["input_ids"]],
                        dtype=torch.long,
                        device=device
                    ),
                    "attention_mask": torch.tensor(
                        [feature["attention_mask"]],
                        dtype=torch.long,
                        device=device
                    ),
                    "token_type_ids": torch.tensor(
                        [feature["token_type_ids"]],
                        dtype=torch.long,
                        device=device
                    )
                }

                outputs = model(**inputs)

                logits = outputs.logits[0].cpu().numpy()

                chunk_predictions = map_chunk_predictions(
                    logits,
                    word_ids
                )

                for word_idx, label_id in chunk_predictions.items():

                    if label_id not in id2label:
                        raise ValueError(
                            f"Label ID tidak ditemukan: {label_id}"
                        )

                    predicted_labels[word_idx] = id2label[label_id]

            # Memperbaiki urutan label BIO
            predicted_labels = normalize_bio(
                predicted_labels
            )

            spans = bio_to_spans(predicted_labels)

            # Mengubah indeks token menjadi posisi karakter
            for start_idx, end_idx, entity_type in spans:

                if entity_type not in ENTITY_MAPPING:
                    continue

                char_start = (
                    sentence_start +
                    token_offsets[start_idx][0]
                )

                char_end = (
                    sentence_start +
                    token_offsets[end_idx - 1][1]
                )

                entities.append({
                    "entity": text[char_start:char_end],
                    "type": ENTITY_MAPPING[entity_type],
                    "start": char_start,
                    "end": char_end
                })

    return sorted(
        entities,
        key=lambda item: item["start"]
    )
