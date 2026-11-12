
from pathlib import Path

import joblib
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)


# ==================================================
# PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CRF_MODEL_PATH = BASE_DIR / "models" / "crf" / "best_crf.joblib"
CRF_METADATA_PATH = BASE_DIR / "models" / "crf" / "crf_metadata.joblib"

TRANSFORMER_MODEL_DIR = BASE_DIR / "models" / "transformers"


# ==================================================
# DEVICE CONFIGURATION
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# LOAD CRF MODEL
# ==================================================

def load_crf_model():

    if not CRF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model CRF tidak ditemukan: {CRF_MODEL_PATH}"
        )

    model = joblib.load(CRF_MODEL_PATH)

    metadata = None

    if CRF_METADATA_PATH.exists():
        metadata = joblib.load(CRF_METADATA_PATH)

    return model, metadata


# ==================================================
# LOAD TRANSFORMER MODEL
# ==================================================

def load_transformer_model():

    if not TRANSFORMER_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model Transformer tidak ditemukan: "
            f"{TRANSFORMER_MODEL_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TRANSFORMER_MODEL_DIR),
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise ValueError(
            "Transformer membutuhkan fast tokenizer."
        )

    model = AutoModelForTokenClassification.from_pretrained(
        str(TRANSFORMER_MODEL_DIR)
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer, DEVICE
