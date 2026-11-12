
import re

from src.crf_features import sentence2features
from src.spans import bio_to_spans, normalize_bio


# ==================================================
# ENTITY MAPPING
# ==================================================

ENTITY_MAPPING = {
    "PPL": "Person",
    "PLC": "Location",
    "IND": "Product/Brand",
    "EVT": "Event",
    "FNB": "Food/Beverage"
}


# ==================================================
# TEXT TOKENIZATION
# ==================================================

TOKEN_PATTERN = re.compile(
    r"\w+(?:[-’']\w+)*|[^\w\s]",
    flags=re.UNICODE
)


def tokenize_with_offsets(text):
    """
    Memecah teks menjadi token sekaligus menyimpan
    posisi awal dan akhir setiap token pada teks asli.
    """

    tokens = []
    offsets = []

    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))

    return tokens, offsets


# ==================================================
# SENTENCE SEGMENTATION
# ==================================================

def split_sentences_with_offsets(text):
    """
    Memisahkan teks berdasarkan tanda baca akhir kalimat
    dan baris baru, sambil mempertahankan posisi karakter.
    """

    sentences = []

    for match in re.finditer(r"[^.!?\n]+[.!?]*|[.!?]+", text):
        start, end = match.span()

        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        if start < end:
            sentences.append((start, end))

    return sentences


# ==================================================
# CRF PREDICTION
# ==================================================

def predict_crf(model, text):
    """
    Melakukan prediksi NER menggunakan model CRF.

    Output:
        [
            {
                "entity": "...",
                "type": "...",
                "start": 0,
                "end": 10
            }
        ]
    """

    if not isinstance(text, str):
        raise TypeError("Input harus berupa string.")

    if not text.strip():
        return []

    entities = []

    sentence_offsets = split_sentences_with_offsets(text)

    for sentence_start, sentence_end in sentence_offsets:

        sentence_text = text[sentence_start:sentence_end]

        tokens, token_offsets = tokenize_with_offsets(sentence_text)

        if not tokens:
            continue

        # Ekstraksi fitur harus sama dengan proses training
        features = sentence2features(tokens)

        # Prediksi label BIO
        predicted_labels = model.predict_single(features)

        if len(predicted_labels) != len(tokens):
            raise ValueError(
                "Jumlah prediksi label tidak sesuai dengan jumlah token."
            )

        predicted_labels = normalize_bio(predicted_labels)

        # Konversi label BIO menjadi indeks token
        spans = bio_to_spans(predicted_labels)

        for start_idx, end_idx, entity_type in spans:

            if entity_type not in ENTITY_MAPPING:
                continue

            # Konversi indeks token menjadi posisi karakter
            char_start = (
                sentence_start + token_offsets[start_idx][0]
            )

            char_end = (
                sentence_start + token_offsets[end_idx - 1][1]
            )

            entities.append({
                "entity": text[char_start:char_end],
                "type": ENTITY_MAPPING[entity_type],
                "start": char_start,
                "end": char_end
            })

    return sorted(entities, key=lambda item: item["start"])
