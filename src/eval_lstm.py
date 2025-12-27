from typing import List, Dict

import evaluate


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Считает метрику ROUGE по спискам предсказаний и эталонных текстов.

    :param predictions: список строк — предсказанные моделью продолжения/предложения
    :param references: список строк — реальные (целевые) продолжения/предложения
    :return: словарь с ключами 'rouge1', 'rouge2', 'rougeL', 'rougeLsum' и значениями метрик
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Длины списков не совпадают: predictions={len(predictions)}, "
            f"references={len(references)}"
        )

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    # Приводим к float, чтобы не тянуть за собой numpy / Tensor типы
    return {k: float(v) for k, v in scores.items()}


if __name__ == "__main__":
    # Пример использования: здесь вместо заглушек нужно подставить
    # реальные предсказания LSTM-модели и соответствующие таргеты.
    example_predictions = [
        "это пример предсказанного текста",
        "вторая строка предсказания",
    ]
    example_references = [
        "это пример истинного текста",
        "вторая тестовая строка",
    ]

    metrics = compute_rouge(example_predictions, example_references)
    print("ROUGE метрики для примера:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")



