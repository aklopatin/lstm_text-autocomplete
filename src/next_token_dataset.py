from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """
    torch.utils.data.Dataset для задачи автодополнения текста:
    на вход подаются индексы токенов контекста фиксированной длины,
    а таргетом является индекс следующего токена.

    Ожидается, что токены уже преобразованы в индексы по словарю.
    """

    def __init__(
        self,
        indexed_sentences: List[List[int]],
        context_size: int = 5,
    ) -> None:
        """
        :param indexed_sentences: список предложений, где каждое предложение —
            список индексов токенов (int).
        :param context_size: длина контекста (количество предыдущих токенов),
            на основе которого нужно предсказывать следующий токен.
        """
        if context_size < 1:
            raise ValueError("context_size должен быть >= 1")

        self.context_size = context_size
        self._samples: List[Tuple[List[int], int]] = []

        # Для каждого предложения строим пары (контекст, следующий токен).
        for sent in indexed_sentences:
            if len(sent) <= context_size:
                continue

            for i in range(context_size, len(sent)):
                context = sent[i - context_size : i]
                target = sent[i]
                self._samples.append((context, target))

        if not self._samples:
            raise ValueError(
                "После подготовки данных не осталось ни одного обучающего примера. "
                "Проверьте длину предложений и значение context_size."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self._samples[idx]
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor


def build_vocab(
    tokenized_sentences: List[List[str]],
    min_freq: int = 1,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> Tuple[Dict[str, int], List[str]]:
    """
    Строит словарь токенов по токенизированным предложениям.

    :param tokenized_sentences: список предложений (списков токенов-строк).
    :param min_freq: минимальная частота токена для попадания в словарь.
    :param pad_token: специальный токен для паддинга.
    :param unk_token: специальный токен для неизвестных токенов.
    :return: кортеж (stoi, itos), где
        stoi — dict: token -> index,
        itos — list: index -> token.
    """
    from collections import Counter

    counter = Counter()
    for sent in tokenized_sentences:
        counter.update(sent)

    itos: List[str] = [pad_token, unk_token]
    for token, freq in counter.items():
        if freq >= min_freq and token not in (pad_token, unk_token):
            itos.append(token)

    stoi: Dict[str, int] = {tok: idx for idx, tok in enumerate(itos)}
    return stoi, itos


def sentences_to_indices(
    tokenized_sentences: List[List[str]],
    stoi: Dict[str, int],
    unk_token: str = "<unk>",
) -> List[List[int]]:
    """
    Преобразует токены в индексы согласно словарю.

    :param tokenized_sentences: список предложений (списков токенов-строк).
    :param stoi: словарь token -> index.
    :param unk_token: токен для неизвестных слов.
    :return: список предложений, где каждый токен заменён на индекс.
    """
    unk_idx = stoi.get(unk_token)
    if unk_idx is None:
        raise ValueError(f"Токен неизвестных слов '{unk_token}' отсутствует в словаре.")

    indexed_sentences: List[List[int]] = []
    for sent in tokenized_sentences:
        indexed_sent = [stoi.get(tok, unk_idx) for tok in sent]
        if indexed_sent:
            indexed_sentences.append(indexed_sent)

    return indexed_sentences


