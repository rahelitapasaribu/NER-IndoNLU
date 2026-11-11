# NER IndoNLU

Proyek eksperimen Named Entity Recognition (NER) untuk bahasa Indonesia dengan dua pendekatan utama:

- Conditional Random Field (CRF)
- Model transformer IndoBERT berdasarkan `indobenchmark/indobert-base-p1`

Proyek ini memanfaatkan dataset NERP dari IndoNLU dan mengevaluasi hasil pada level span entitas dengan format BIO.

## Tujuan proyek

Tujuan utama dari repositori ini adalah membangun dan membandingkan model NER untuk mengenali entitas dalam teks bahasa Indonesia, seperti:

- PPL: Person / Orang
- PLC: Place / Tempat
- IND: Individual / Organisasi / Institusi
- EVT: Event / Peristiwa
- FNB: Food / Makanan

Model dikembangkan dengan fokus pada validasi label BIO, pelatihan model, evaluasi exact-match span, serta analisis kesalahan prediksi.

## Fitur utama

- Pengunduhan dan parsing dataset NERP dari IndoNLU
- Validasi struktur anotasi BIO untuk memastikan label konsisten
- Ekstraksi dan evaluasi entitas pada level span
- Pelatihan CRF dengan beberapa konfigurasi hyperparameter
- Pelatihan model token-classification IndoBERT
- Analisis kesalahan label BIO dan hasil prediksi
- Hasil eksperimen disimpan ke file CSV di folder `results/`

## Struktur proyek

```text
NER-IndoNLU/
├── README.md
├── models/
│   ├── crf/
│   │   ├── best_crf.joblib
│   │   └── crf_metadata.joblib
│   └── transformers/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
├── results/
│   └── crf/
│       ├── error_analysis.csv
│       ├── test_results.csv
│       └── validation_results.csv
├── scripts/
│   ├── error_analysis.py
│   └── manual_inspection.py
├── src/
│   ├── analyze_errors.py
│   ├── annotation.py
│   ├── crf_features.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── spans.py
│   ├── train_crf.py
│   ├── train_transformer.py
│   └── transformer_data.py
├── tests/
│   └── test_annotation.py
└── .gitignore
```

## Persiapan lingkungan

Pastikan Python 3.10+ sudah terinstall. Kemudian install dependency berikut:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install numpy torch transformers scikit-learn sklearn-crfsuite joblib requests
```

## Jalankan validasi dataset

Script ini memeriksa apakah anotasi BIO pada dataset valid atau terdapat inkonsistensi.

```bash
python tests/test_annotation.py
```

## Inspeksi data manual

Script ini menampilkan beberapa sampel acak dari dataset untuk inspeksi manual terhadap token, label BIO, dan entitas yang terdeteksi.

```bash
python scripts/manual_inspection.py
```

## Analisis error BIO

Script ini menghitung jenis kesalahan struktur BIO pada dataset, misalnya:

- `I-X setelah O`
- `I-X setelah tipe berbeda`
- `I-X di awal sequence`

```bash
python scripts/error_analysis.py
```

## Melatih model CRF

Model CRF dilatih pada fitur token yang dibangun dari teks dan label BIO. Script ini juga memilih konfigurasi terbaik berdasarkan metrik F1 pada validation set.

```bash
python src/train_crf.py
```

Hasil eksperimen disimpan di:

- `results/crf/validation_results.csv`
- `results/crf/test_results.csv`
- `results/crf/error_analysis.csv`

Model terbaik juga disimpan ke:

- `models/crf/best_crf.joblib`
- `models/crf/crf_metadata.joblib`

## Melatih model Transformer

Model transformer menggunakan IndoBERT sebagai encoder dan di-training untuk token classification dengan label BIO.

```bash
python src/train_transformer.py
```

Model yang dihasilkan akan disimpan pada direktori `models/transformers/`.

## Evaluasi dan analisis hasil

Modul evaluasi dibuat untuk menghitung precision, recall, dan F1 berdasarkan exact-match span entity. Implementasi evaluasi dapat dilihat di `src/evaluate.py`.

Untuk menjalankan analisis lebih lanjut, Anda dapat memanggil fungsi-fungsi di modul `src/analyze_errors.py` dan `src/evaluate.py` melalui Python.

## Dataset

Dataset yang dipakai adalah dataset NERP dari IndoNLU yang diunduh secara otomatis melalui URL GitHub resmi. Struktur data terdiri dari:

- `id`
- `tokens`
- `labels`

Untuk tiap split: `train`, `validation`, dan `test`.

## Catatan penting

- Label yang digunakan mengikuti format BIO dan hanya mendukung tipe entitas berikut: `PPL`, `PLC`, `IND`, `EVT`, `FNB`.
- Kode ditulis untuk eksperimen NER bahasa Indonesia dan cocok untuk pengembangan lebih lanjut seperti hyperparameter tuning, error analysis, atau deployment model.
- Proyek ini bersifat penelitian/eksperimen; struktur kode masih dapat disesuaikan untuk kebutuhan produksi.

## Lisensi

Repositori ini tidak mencantumkan lisensi khusus di file terpisah. Jika Anda akan menggunakan proyek ini untuk kebutuhan komersial atau publik, pastikan untuk menyesuaikan lisensi sesuai kebijakan dataset dan dependency yang digunakan.

## Referensi

- IndoNLU dataset untuk NERP
- IndoBERT (`indobenchmark/indobert-base-p1`)
- CRF dengan `sklearn-crfsuite`
