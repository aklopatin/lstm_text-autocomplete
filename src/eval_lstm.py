from typing import List, Dict

import evaluate


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"Длины списков не совпадают: predictions={len(predictions)}, "
            f"references={len(references)}"
        )
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    return {k: float(v) for k, v in scores.items()}


if __name__ == "__main__":
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
