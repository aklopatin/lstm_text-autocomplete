import torch
import torch.nn as nn


class NNAutocomplete(nn.Module):
    """
    Простейшая LSTM‑модель для автодополнения текста.

    Ожидается, что на вход подаются one‑hot представления токенов
    размерности (batch_size, seq_len, input_dim), где input_dim == vocab_size.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        # LSTM по последовательности контекста
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Линейный слой проецирует последнее скрытое состояние в логиты по словарю
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: тензор формы (batch_size, seq_len, input_dim)
        :return: логиты для следующего токена формы (batch_size, output_dim)
        """
        # output: (batch_size, seq_len, hidden_dim)
        output, _ = self.rnn(x)
        # Берём представление последнего токена в последовательности
        last_hidden = output[:, -1, :]  # (batch_size, hidden_dim)
        logits = self.fc(last_hidden)   # (batch_size, output_dim)
        return logits