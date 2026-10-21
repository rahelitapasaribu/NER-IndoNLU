from pprint import pprint
from data_loader import load_dataset


def word_shape(word):
    """Mengubah karakter menjadi pola X, x, d, atau tanda baca"""
    result = []

    for char in word:
        if char.isupper():
            result.append("X")
        elif char.islower():
            result.append("x")
        elif char.isdigit():
            result.append("d")
        else:
            result.append(char)

    return "".join(result)

def token2features(tokens, i):
    """Mengubah satu token beserta konteksnya menjadi fitur CRF"""
    if not tokens:
        raise ValueError("Daftar token tidak boleh kosong")

    if i < 0 or i >= len(tokens):
        raise IndexError("Indeks token berada di luar batas")

    word = tokens[i]

    features = {
        "bias": 1.0,
        "word.lower": word.lower(),

        "prefix2": word[:2],
        "prefix3": word[:3],

        "suffix2": word[-2:],
        "suffix3": word[-3:],

        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),

        "contains_digit": any(c.isdigit() for c in word),
        "contains_hyphen": "-" in word,

        "word_shape": word_shape(word),

        "BOS": i == 0,
        "EOS": i == len(tokens) - 1,
    }

    """Konteks satu token sebelumnya"""
    if i > 0:
        prev = tokens[i - 1]

        features.update({
            "-1:word.lower": prev.lower(),
            "-1:is_title": prev.istitle(),
            "-1:word_shape": word_shape(prev),
        })

    """Konteks satu token sesudahnya"""
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]

        features.update({
            "+1:word.lower": nxt.lower(),
            "+1:is_title": nxt.istitle(),
            "+1:word_shape": word_shape(nxt),
        })

    return features

def sentence2features(tokens):
    """Mengubah semua token dalam sequence menjadi fitur CRF"""
    return [token2features(tokens, i) for i in range(len(tokens))]


if __name__ == "__main__":

    """Memuat dataset asli"""
    dataset = load_dataset()

    train_data = dataset["train"]

    """Mengambil dua contoh dari data train"""
    examples = train_data[:2]

    for idx, example in enumerate(examples, start=1):
        tokens = example["tokens"]
        labels = example["labels"]

        print(f"\nContoh Data Train Ke-{idx}")

        print(f"Token")
        print(tokens)

        print(f"Label BIO")
        print(labels)

        X = sentence2features(tokens)

        print(f"Fitur CRF")
        pprint(X)

        print("\nJumlah token:", len(tokens))
        print("Jumlah dictionary fitur:", len(X))
        print("Jumlah label BIO:", len(labels))
    