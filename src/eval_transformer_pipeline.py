from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from eval_lstm import compute_rouge


def read_sentences(path: str) -> List[str]:
    """
    Читает файл с предложениями (одна строка — одно предложение) и
    возвращает список строк.
    """
    sentences: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                sentences.append(text)
    return sentences


def split_prefix_suffix(text: str) -> Tuple[str, str] | None:
    """
    Делит предложение на первые 3/4 токенов (префикс) и последние 1/4 (суффикс).
    Возвращает (prefix, suffix) или None, если предложение слишком короткое.
    """
    tokens = text.split()
    if len(tokens) < 4:
        return None

    split_idx = max(1, int(len(tokens) * 3 / 4))
    if split_idx >= len(tokens):
        return None

    prefix_tokens = tokens[:split_idx]
    suffix_tokens = tokens[split_idx:]

    prefix = " ".join(prefix_tokens)
    suffix = " ".join(suffix_tokens)
    return prefix, suffix


def evaluate_distilgpt2_on_val(
    val_path: str = "./data/val_tokens.txt",
    model_name: str = "distilgpt2",
    max_samples: int | None = 200,
    max_new_tokens_factor: float = 1.0,
) -> None:
    """
    Оценивает качество distilgpt2 на валидационной выборке по ROUGE.

    Для каждого предложения:
      - берём первые 3/4 слов как вход (промпт),
      - модель генерирует продолжение,
      - сравниваем с истинной последней 1/4 предложения.

    :param val_path: путь к файлу с валидационными предложениями
    :param model_name: имя модели в Transformers (по умолчанию distilgpt2)
    :param max_samples: максимальное количество примеров из валидации
        (None — использовать все)
    :param max_new_tokens_factor: во сколько раз максимум генерируемых токенов
        превышает длину истинного суффикса (1.0 — ровно столько же)
    """
    device_idx = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # У GPT‑2 по умолчанию нет pad_token, поэтому используем eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_idx,
    )

    sentences = read_sentences(val_path)
    if max_samples is not None:
        sentences = sentences[:max_samples]

    predictions: List[str] = []
    references: List[str] = []
    examples: List[tuple[str, str, str]] = []

    for text in sentences:
        split = split_prefix_suffix(text)
        if split is None:
            continue
        prefix, suffix = split

        suffix_tokens = suffix.split()
        target_len = len(suffix_tokens)
        max_new_tokens = max(1, int(target_len * max_new_tokens_factor))

        # Генерация продолжения
        outputs = text_gen(
            prefix,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # жадная генерация для стабильности
            num_beams=5,              # beam‑поиск
            no_repeat_ngram_size=3,   # избегаем повторов
            pad_token_id=tokenizer.eos_token_id,
        )

        full_text = outputs[0]["generated_text"]

        # Отделяем только сгенерированное продолжение
        if full_text.startswith(prefix):
            completion = full_text[len(prefix):].strip()
        else:
            # На случай, если модель слегка изменила начало
            completion = full_text.strip()

        # Ограничиваем длину продолжения длиной истинного суффикса
        completion_tokens = completion.split()
        completion_tokens = completion_tokens[:target_len]
        completion_trimmed = " ".join(completion_tokens)

        predictions.append(completion_trimmed)
        references.append(suffix)

        if len(examples) < 5:
            examples.append((prefix, suffix, completion_trimmed))

    if not predictions:
        raise ValueError(
            "Не удалось получить ни одного примера для оценки. "
            "Проверьте содержимое валидационного файла и длину предложений."
        )

    rouge_scores = compute_rouge(predictions, references)

    print(f"Оценка модели {model_name} на валидационной выборке ({len(predictions)} примеров):")
    print(
        f"ROUGE-1: {rouge_scores['rouge1']:.4f} | "
        f"ROUGE-2: {rouge_scores['rouge2']:.4f} | "
        f"ROUGE-L: {rouge_scores['rougeL']:.4f}"
    )

    print("\nПримеры предсказаний:\n")
    for i, (prefix, true_suffix, pred_suffix) in enumerate(examples, start=1):
        print(f"=== Пример {i} ===")
        print(f"Префикс (3/4): {prefix}")
        print(f"Истинное продолжение (1/4): {true_suffix}")
        print(f"Предсказанное продолжение:   {pred_suffix}")
        print()


if __name__ == "__main__":
    evaluate_distilgpt2_on_val()


