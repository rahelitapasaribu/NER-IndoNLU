
from src.inference.model_loader import (
    load_crf_model,
    load_transformer_model
)

from src.inference.crf_predictor import predict_crf
from src.inference.transformer_predictor import predict_transformer


def load_selected_model(model_name):
    """
    Memuat model berdasarkan pilihan pengguna.
    """

    if model_name == "CRF":
        model, metadata = load_crf_model()

        return {
            "model": model,
            "metadata": metadata
        }

    elif model_name == "Transformer":
        model, tokenizer, device = load_transformer_model()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }

    else:
        raise ValueError(
            f"Model tidak dikenal: {model_name}"
        )


def predict_entities(model_name, text, loaded_model):
    """
    Melakukan prediksi entitas menggunakan model terpilih.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    if model_name == "CRF":
        return predict_crf(
            model=loaded_model["model"],
            text=text
        )

    elif model_name == "Transformer":
        return predict_transformer(
            model=loaded_model["model"],
            tokenizer=loaded_model["tokenizer"],
            device=loaded_model["device"],
            text=text
        )

    else:
        raise ValueError(
            f"Model tidak dikenal: {model_name}"
        )
