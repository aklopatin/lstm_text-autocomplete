import re
import random
from typing import List, Tuple
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    """
    Базовая очистка текста:
    - приведение к нижнему регистру;
    - удаление ссылок, упоминаний, эмодзи;
    - замена нестандартных символов на более стандартные аналоги.
    """
    # к нижнему регистру
    text = text.lower()

    # удаляем ссылки (http/https, www и т.п.)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # удаляем упоминания вида @username
    text = re.sub(r"@\w+", " ", text)

    # замена некоторых нестандартных символов на стандартные аналоги
    replacements = {
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "’": "'",
        "‘": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "✓": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # удаляем эмодзи и прочие пиктограммы (символы вне базовой многоязычной плоскости)
    text = re.sub(r"[\U00010000-\U0010FFFF]", "", text)

    # схлопываем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_preprocess_text(path: str = "./data/raw_dataset.txt") -> List[str]:
    """Загружает файл, очищает текст и возвращает список токенов (splittedtext)."""
    with open(path, "r", encoding="latin-1") as file:
        text = file.read()
        corpus = [
            clean_text(line.split(";")[0]) for line in text.strip().split("\n")
        ]
        print(corpus[:5])
        word_tokens = [message.split() for message in corpus if message]
        return word_tokens


def train_val_test_split(
    tokens: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Разбивает список токенов на train / val / test.

    По умолчанию: 80% / 10% / 10%. Можно менять доли параметрами.
    """
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("train_ratio, val_ratio и test_ratio должны быть в диапазоне (0, 1).")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Сумма train_ratio, val_ratio и test_ratio должна быть равна 1.")

    tokens_copy = list(tokens)

    if shuffle:
        random.seed(seed)
        random.shuffle(tokens_copy)

    n = len(tokens_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_tokens = tokens_copy[:train_end]
    val_tokens = tokens_copy[train_end:val_end]
    test_tokens = tokens_copy[val_end:]

    return train_tokens, val_tokens, test_tokens


if __name__ == "__main__":
    splittedtext = load_and_preprocess_text()
    train_tokens, val_tokens, test_tokens = train_val_test_split(splittedtext)

    # сохраняем разбиение в файлы в папке data
    with open("./data/train_tokens.txt", "w", encoding="utf-8") as f_train:
        for sent in train_tokens:
            f_train.write(" ".join(sent) + "\n")

    with open("./data/val_tokens.txt", "w", encoding="utf-8") as f_val:
        for sent in val_tokens:
            f_val.write(" ".join(sent) + "\n")

    with open("./data/test_tokens.txt", "w", encoding="utf-8") as f_test:
        for sent in test_tokens:
            f_test.write(" ".join(sent) + "\n")
