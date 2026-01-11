from typing import List, Tuple, Dict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from next_token_dataset import (
    NextTokenDataset,
    build_vocab,
    sentences_to_indices,
)
from lstm_model import NNAutocomplete
from eval_lstm import compute_rouge


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
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, int, Dict[str, int], List[str]]:
    """
    Создаёт DataLoader для обучающей и валидационной выборок.

    :return: (train_loader, val_loader, vocab_size, stoi, itos)
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

    return train_loader, val_loader, vocab_size, stoi, itos


class SentenceDataset(Dataset):
    """Dataset, который возвращает целые предложения в виде последовательностей индексов."""

    def __init__(self, indexed_sentences: List[List[int]]) -> None:
        self.sentences = [
            torch.tensor(s, dtype=torch.long) for s in indexed_sentences if len(s) > 1
        ]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sentences[idx]


def evaluate_model_rouge(
    model: NNAutocomplete,
    tokenized_sentences: List[List[str]],
    stoi: Dict[str, int],
    itos: List[str],
    device: torch.device,
    context_size: int = 5,
    max_new_tokens: int | None = None,
    max_examples: int = 5,
) -> Tuple[Dict[str, float], List[Tuple[str, str, str]]]:
    """
    Оценивает модель с помощью ROUGE на списке предложений.

    Для каждого предложения берём первые 3/4 токенов как вход (префикс),
    модель генерирует оставшуюся часть, и мы сравниваем её с истинными
    последними 1/4 токенов.
    """
    vocab_size = len(itos)
    pad_idx = stoi.get("<pad>", 0)

    indexed_sentences = sentences_to_indices(tokenized_sentences, stoi)
    dataset = SentenceDataset(indexed_sentences)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions: List[str] = []
    references: List[str] = []
    examples: List[Tuple[str, str, str]] = []

    model.eval()
    with torch.no_grad():
        for sent_tensor in loader:
            # sent_tensor: (1, seq_len)
            sent_tensor = sent_tensor.squeeze(0).to(device)  # (seq_len,)
            sent_indices = sent_tensor.tolist()

            if len(sent_indices) < 4:
                # Слишком короткие предложения пропускаем
                continue

            split_idx = max(1, int(len(sent_indices) * 3 / 4))
            prefix = sent_indices[:split_idx]
            target_suffix = sent_indices[split_idx:]

            if not target_suffix:
                continue

            steps = len(target_suffix) if max_new_tokens is None else max_new_tokens

            generated_suffix: List[int] = []
            cur_sequence = prefix.copy()

            for _ in range(steps):
                cur_context = cur_sequence[-context_size:]
                if len(cur_context) < context_size:
                    cur_context = [pad_idx] * (context_size - len(cur_context)) + cur_context

                context_tensor = torch.tensor(
                    cur_context, dtype=torch.long, device=device
                ).unsqueeze(0)  # (1, context_size)

                # one-hot кодировка токенов
                context_one_hot = F.one_hot(
                    context_tensor, num_classes=vocab_size
                ).float()  # (1, context_size, vocab_size)

                logits = model(context_one_hot)  # (1, vocab_size)
                next_idx = int(torch.argmax(logits, dim=-1).item())

                generated_suffix.append(next_idx)
                cur_sequence.append(next_idx)

            pred_tokens = [itos[i] for i in generated_suffix]
            ref_tokens = [itos[i] for i in target_suffix]

            pred_str = " ".join(pred_tokens)
            ref_str = " ".join(ref_tokens)

            predictions.append(pred_str)
            references.append(ref_str)

            if len(examples) < max_examples:
                prefix_tokens = [itos[i] for i in prefix]
                prefix_str = " ".join(prefix_tokens)
                examples.append((prefix_str, ref_str, pred_str))

    if not predictions:
        raise ValueError(
            "Не удалось сгенерировать ни одного примера для ROUGE. "
            "Возможно, все предложения слишком короткие."
        )

    return compute_rouge(predictions, references), examples


def train_lstm_model(
    train_path: str = "./data/train_tokens.txt",
    val_path: str = "./data/val_tokens.txt",
    context_size: int = 5,
    batch_size: int = 256,
    hidden_dim: int = 16,
    num_epochs: int = 5,
    lr: float = 1e-3,
    model_path: str = "./models/lstm_autocomplete.pt",
) -> None:
    """
    Обучает LSTM‑модель автодополнения и после каждой эпохи
    выводит средний loss по train и метрики ROUGE на val.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, vocab_size, stoi, itos = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        context_size=context_size,
        batch_size=batch_size,
    )

    model = NNAutocomplete(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        output_dim=vocab_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Используемое устройство: {device}")
    print(f"Размер словаря: {vocab_size}")

    last_examples: List[Tuple[str, str, str]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0

        for contexts, targets in train_loader:
            contexts = contexts.to(device)  # (batch, context_size)
            targets = targets.to(device)    # (batch,)

            # Переводим индексы токенов в one‑hot представление
            inputs_one_hot = F.one_hot(
                contexts, num_classes=vocab_size
            ).float()  # (batch, context_size, vocab_size)

            logits = model(inputs_one_hot)  # (batch, vocab_size)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = contexts.size(0)
            running_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        avg_loss = running_loss / max(1, num_samples)

        # Оценка ROUGE на валидационных предложениях
        val_sentences = read_tokens_file(val_path)
        rouge_scores, examples = evaluate_model_rouge(
            model=model,
            tokenized_sentences=val_sentences,
            stoi=stoi,
            itos=itos,
            device=device,
            context_size=context_size,
        )
        last_examples = examples

        print(
            f"Эпоха {epoch}/{num_epochs} | "
            f"train loss: {avg_loss:.4f} | "
            f"ROUGE-1: {rouge_scores['rouge1']:.4f} | "
            f"ROUGE-2: {rouge_scores['rouge2']:.4f} | "
            f"ROUGE-L: {rouge_scores['rougeL']:.4f}"
        )

    # Примеры предсказаний LSTM на валидационной выборке
    if last_examples:
        print("\nПримеры предсказаний LSTM на валидационной выборке:")
        for i, (prefix_str, true_suffix, pred_suffix) in enumerate(last_examples, start=1):
            print(f"=== Пример {i} ===")
            print(f"Префикс (3/4): {prefix_str}")
            print(f"Истинное продолжение (1/4): {true_suffix}")
            print(f"Предсказанное продолжение:   {pred_suffix}")
            print()

    # Сохраняем обученную модель на диск
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "stoi": stoi,
            "itos": itos,
            "context_size": context_size,
            "hidden_dim": hidden_dim,
        },
        model_path,
    )
    print(f"Модель сохранена в файл: {model_path}")


if __name__ == "__main__":
    train_lstm_model()


