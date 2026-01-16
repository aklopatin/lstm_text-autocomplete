from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from lstm_model import NNAutocomplete


def read_tokenized_sentences(path: str) -> List[List[str]]:
    sentences: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def split_prefix_suffix_tokens(tokens: List[str]) -> Tuple[List[str], List[str]] | None:
    if len(tokens) < 4:
        return None
    split_idx = max(1, int(len(tokens) * 3 / 4))
    if split_idx >= len(tokens):
        return None
    prefix_tokens = tokens[:split_idx]
    suffix_tokens = tokens[split_idx:]
    return prefix_tokens, suffix_tokens


def load_lstm_checkpoint(
    model_path: str,
    device: torch.device,
) -> Tuple[NNAutocomplete, dict[str, int], List[str], int]:
    checkpoint = torch.load(model_path, map_location=device)
    vocab_size = int(checkpoint["vocab_size"])
    hidden_dim = int(checkpoint["hidden_dim"])
    context_size = int(checkpoint["context_size"])
    embed_dim = int(checkpoint.get("embed_dim", hidden_dim))
    model = NNAutocomplete(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=vocab_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["stoi"], checkpoint["itos"], context_size


def generate_distil_prediction(
    prefix_text: str,
    target_len: int,
    text_gen,
    tokenizer,
    max_new_tokens_factor: float,
) -> str:
    max_new_tokens = max(1, int(target_len * max_new_tokens_factor))
    outputs = text_gen(
        prefix_text,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=5,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = outputs[0]["generated_text"]
    if full_text.startswith(prefix_text):
        completion = full_text[len(prefix_text) :].strip()
    else:
        completion = full_text.strip()
    completion_tokens = completion.split()
    completion_tokens = completion_tokens[:target_len]
    return " ".join(completion_tokens)


def generate_lstm_prediction(
    prefix_tokens: List[str],
    target_len: int,
    model: NNAutocomplete,
    stoi: dict[str, int],
    itos: List[str],
    context_size: int,
    device: torch.device,
) -> str:
    unk_idx = stoi.get("<unk>")
    if unk_idx is None:
        raise ValueError("В словаре отсутствует токен <unk>.")
    pad_idx = stoi.get("<pad>", 0)
    prefix_indices = [stoi.get(tok, unk_idx) for tok in prefix_tokens]
    generated_indices = model.generate(
        context_indices=prefix_indices,
        context_size=context_size,
        max_new_tokens=target_len,
        pad_idx=pad_idx,
        device=device,
    )
    pred_tokens = [itos[i] for i in generated_indices]
    return " ".join(pred_tokens)


def compare_distil_lstm_predictions(
    val_path: str = "./data/val_tokens.txt",
    model_name: str = "distilgpt2",
    model_path: str = "./models/lstm_autocomplete.pt",
    num_examples: int = 20,
    max_new_tokens_factor: float = 1.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_idx = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_idx,
    )
    lstm_model, stoi, itos, context_size = load_lstm_checkpoint(model_path, device)
    sentences = read_tokenized_sentences(val_path)
    examples: List[Tuple[str, str, str, str]] = []
    for tokens in sentences:
        split = split_prefix_suffix_tokens(tokens)
        if split is None:
            continue
        prefix_tokens, suffix_tokens = split
        if not suffix_tokens:
            continue
        prefix_text = " ".join(prefix_tokens)
        suffix_text = " ".join(suffix_tokens)
        target_len = len(suffix_tokens)
        distil_pred = generate_distil_prediction(
            prefix_text=prefix_text,
            target_len=target_len,
            text_gen=text_gen,
            tokenizer=tokenizer,
            max_new_tokens_factor=max_new_tokens_factor,
        )
        lstm_pred = generate_lstm_prediction(
            prefix_tokens=prefix_tokens,
            target_len=target_len,
            model=lstm_model,
            stoi=stoi,
            itos=itos,
            context_size=context_size,
            device=device,
        )
        examples.append((prefix_text, suffix_text, distil_pred, lstm_pred))
        if len(examples) >= num_examples:
            break
    if not examples:
        raise ValueError(
            "Не удалось получить ни одного примера для сравнения. "
            "Проверьте валидационный файл и модель LSTM."
        )
    print(
        f"Сравнение предсказаний distilgpt2 и LSTM "
        f"на {len(examples)} одинаковых выражениях:"
    )
    for i, (prefix, true_suffix, distil_pred, lstm_pred) in enumerate(examples, start=1):
        print(f"=== Пример {i} ===")
        print(f"Префикс (3/4): {prefix}")
        print(f"Истинное продолжение (1/4): {true_suffix}")
        print(f"distilgpt2: {distil_pred}")
        print(f"LSTM:       {lstm_pred}")
        print()


if __name__ == "__main__":
    compare_distil_lstm_predictions()
