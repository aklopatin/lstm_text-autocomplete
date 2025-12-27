from __future__ import annotations

"""
Простой сценарий, который последовательно выполняет все ключевые шаги:

1. Подготовка датасета из raw_dataset.txt (train / val / test).
2. Обучение LSTM-модели автодополнения и замер ROUGE.
3. Оценка трансформера distilgpt2 и замер ROUGE + примеры предсказаний.

Запуск:
    python src/run_pipeline.py
"""

from pathlib import Path

from data_utils import load_and_preprocess_text, train_val_test_split
from lstm_train import train_lstm_model
from eval_transformer_pipeline import evaluate_distilgpt2_on_val


DATA_DIR = Path("./data")


def prepare_dataset() -> None:
    """Готовит файлы train_tokens.txt, val_tokens.txt, test_tokens.txt."""
    raw_path = DATA_DIR / "raw_dataset.txt"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Не найден файл с сырыми данными: {raw_path}. "
            f"Поместите raw_dataset.txt в папку data."
        )

    print("1/3: Загрузка и предобработка исходного текста...")
    splittedtext = load_and_preprocess_text(str(raw_path))
    train_tokens, val_tokens, test_tokens = train_val_test_split(splittedtext)

    print("   Сохранение train/val/test токенов...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with (DATA_DIR / "train_tokens.txt").open("w", encoding="utf-8") as f_train:
        for sent in train_tokens:
            f_train.write(" ".join(sent) + "\n")

    with (DATA_DIR / "val_tokens.txt").open("w", encoding="utf-8") as f_val:
        for sent in val_tokens:
            f_val.write(" ".join(sent) + "\n")

    with (DATA_DIR / "test_tokens.txt").open("w", encoding="utf-8") as f_test:
        for sent in test_tokens:
            f_test.write(" ".join(sent) + "\n")

    print(
        f"   Готово. Кол-во предложений: "
        f"train={len(train_tokens)}, val={len(val_tokens)}, test={len(test_tokens)}"
    )


def train_lstm() -> None:
    """Обучает LSTM и замеряет ROUGE на валидации."""
    print("\n2/3: Обучение LSTM-модели и замер ROUGE...")
    train_lstm_model()
    print("   Обучение LSTM завершено и модель сохранена.")


def eval_transformer() -> None:
    """Оценивает distilgpt2 на валидации и выводит ROUGE + примеры."""
    print("\n3/3: Оценка distilgpt2 на валидации...")
    evaluate_distilgpt2_on_val()
    print("   Оценка distilgpt2 завершена.")


def main() -> None:
    prepare_dataset()
    train_lstm()
    eval_transformer()


if __name__ == "__main__":
    main()


