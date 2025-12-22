from typing import List, Tuple

from torch.utils.data import DataLoader

from data_utils import train_val_test_split
from next_token_dataset import (
    NextTokenDataset,
    build_vocab,
    sentences_to_indices,
)


def read_tokens_file(path: str) -> List[List[str]]:
    """
    Читает файл с токенами (одна строка — одно предложение, токены разделены пробелами)
    и возвращает список предложений.
    """
    sentences: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def create_dataloaders(
    train_path: str = "./data/train_tokens.txt",
    val_path: str = "./data/val_tokens.txt",
    context_size: int = 5,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Создаёт DataLoader для обучающей и валидационной выборок.

    :return: (train_loader, val_loader, vocab_size)
    """
    # Загружаем токенизированные предложения
    train_sentences = read_tokens_file(train_path)
    val_sentences = read_tokens_file(val_path)

    # Строим словарь по train + val (можно оставить только train, по задаче)
    all_sentences = train_sentences + val_sentences
    stoi, itos = build_vocab(all_sentences)
    vocab_size = len(itos)

    # Преобразуем токены в индексы
    train_indexed = sentences_to_indices(train_sentences, stoi)
    val_indexed = sentences_to_indices(val_sentences, stoi)

    # Создаём датасеты
    train_dataset = NextTokenDataset(train_indexed, context_size=context_size)
    val_dataset = NextTokenDataset(val_indexed, context_size=context_size)

    # И соответствующие DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, vocab_size


if __name__ == "__main__":
    train_loader, val_loader, vocab_size = create_dataloaders()
    print(f"Размер словаря: {vocab_size}")
    print(f"Количество батчей в train: {len(train_loader)}")
    print(f"Количество батчей в val: {len(val_loader)}")


