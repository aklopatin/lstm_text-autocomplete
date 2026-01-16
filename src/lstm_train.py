from typing import List, Tuple, Dict
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from next_token_dataset import (
    NextTokenDataset,
    build_vocab,
    sentences_to_indices,
)
from lstm_model import NNAutocomplete
from eval_lstm import compute_rouge


def read_tokens_file(path: str) -> List[List[str]]:
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
    test_path: str = "./data/test_tokens.txt",
    context_size: int = 15,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Dict[str, int], List[str]]:
    train_sentences = read_tokens_file(train_path)
    val_sentences = read_tokens_file(val_path)
    test_sentences = read_tokens_file(test_path)
    all_sentences = train_sentences + val_sentences + test_sentences
    stoi, itos = build_vocab(all_sentences)
    vocab_size = len(itos)
    train_indexed = sentences_to_indices(train_sentences, stoi)
    val_indexed = sentences_to_indices(val_sentences, stoi)
    test_indexed = sentences_to_indices(test_sentences, stoi)
    train_dataset = NextTokenDataset(train_indexed, context_size=context_size)
    val_dataset = NextTokenDataset(val_indexed, context_size=context_size)
    test_dataset = NextTokenDataset(test_indexed, context_size=context_size)
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, vocab_size, stoi, itos


class SentenceDataset(Dataset):
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
    context_size: int = 15,
    max_new_tokens: int | None = None,
    max_examples: int = 5,
    max_samples: int = 1000,
) -> Tuple[Dict[str, float], List[Tuple[str, str, str]]]:
    pad_idx = stoi.get("<pad>", 0)
    if max_samples > 0:
        tokenized_sentences = tokenized_sentences[:max_samples]
    indexed_sentences = sentences_to_indices(tokenized_sentences, stoi)
    dataset = SentenceDataset(indexed_sentences)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions: List[str] = []
    references: List[str] = []
    examples: List[Tuple[str, str, str]] = []
    model.eval()
    with torch.no_grad():
        for sent_tensor in loader:
            sent_tensor = sent_tensor.squeeze(0).to(device)
            sent_indices = sent_tensor.tolist()
            if len(sent_indices) < 4:
                continue
            split_idx = max(1, int(len(sent_indices) * 3 / 4))
            prefix = sent_indices[:split_idx]
            target_suffix = sent_indices[split_idx:]
            if not target_suffix:
                continue
            steps = len(target_suffix) if max_new_tokens is None else max_new_tokens
            generated_suffix = model.generate(
                context_indices=prefix,
                context_size=context_size,
                max_new_tokens=steps,
                pad_idx=pad_idx,
                device=device,
            )
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
    rouge_scores = compute_rouge(predictions, references)
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
    }, examples


def train_lstm_model(
    train_path: str = "./data/train_tokens.txt",
    val_path: str = "./data/val_tokens.txt",
    test_path: str = "./data/test_tokens.txt",
    context_size: int = 15,
    batch_size: int = 256,
    hidden_dim: int = 256,
    embed_dim: int | None = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    model_path: str = "./models/lstm_autocomplete.pt",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, vocab_size, stoi, itos = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        context_size=context_size,
        batch_size=batch_size,
    )
    if embed_dim is None:
        embed_dim = hidden_dim
    model = NNAutocomplete(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
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
        print(f"Эпоха: {epoch}")
        for contexts, targets in train_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            logits = model(contexts)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size_actual = contexts.size(0)
            running_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual
        avg_loss = running_loss / max(1, num_samples)
        test_sentences = read_tokens_file(test_path)
        test_max_samples = 1000
        print(f"ROUGE считаем на фиксированных {test_max_samples} примерах теста.")
        test_rouge_scores, examples = evaluate_model_rouge(
            model=model,
            tokenized_sentences=test_sentences,
            stoi=stoi,
            itos=itos,
            device=device,
            context_size=context_size,
            max_samples=test_max_samples,
        )
        last_examples = examples
        print(
            f"Эпоха {epoch}/{num_epochs} | "
            f"train loss: {avg_loss:.4f} | "
            f"Test ROUGE-1: {test_rouge_scores['rouge1']:.4f} | "
            f"Test ROUGE-2: {test_rouge_scores['rouge2']:.4f}"
        )
    if last_examples:
        print("\nПримеры предсказаний LSTM на тестовой выборке:")
        for i, (prefix_str, true_suffix, pred_suffix) in enumerate(last_examples, start=1):
            print(f"=== Пример {i} ===")
            print(f"Префикс (3/4): {prefix_str}")
            print(f"Истинное продолжение (1/4): {true_suffix}")
            print(f"Предсказанное продолжение:   {pred_suffix}")
            print()
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
            "embed_dim": embed_dim,
        },
        model_path,
    )
    print(f"Модель сохранена в файл: {model_path}")


if __name__ == "__main__":
    train_lstm_model()
