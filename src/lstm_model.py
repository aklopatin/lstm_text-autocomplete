import torch
import torch.nn as nn
import torch.nn.functional as F


class NNAutocomplete(nn.Module):
    """
    Простейшая LSTM‑модель для автодополнения текста.

    Ожидается, что на вход подаются one‑hot представления токенов
    размерности (batch_size, seq_len, input_dim), где input_dim == vocab_size.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.vocab_size = input_dim
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

    def generate(
        self,
        context_indices: list[int],
        context_size: int,
        max_new_tokens: int,
        pad_idx: int,
        device: torch.device,
    ) -> list[int]:
        """
        Генерирует несколько следующих токенов по заданному контексту.

        :param context_indices: индексы токенов контекста
        :param context_size: размер контекста
        :param max_new_tokens: сколько токенов сгенерировать
        :param pad_idx: индекс токена паддинга
        :param device: устройство для вычислений
        :return: список индексов сгенерированных токенов
        """
        generated_indices: list[int] = []
        sequence = context_indices.copy()

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur_context = sequence[-context_size:]
                if len(cur_context) < context_size:
                    cur_context = [pad_idx] * (context_size - len(cur_context)) + cur_context

                context_tensor = torch.tensor(
                    cur_context, dtype=torch.long, device=device
                ).unsqueeze(0)  # (1, context_size)

                context_one_hot = F.one_hot(
                    context_tensor, num_classes=self.vocab_size
                ).float()  # (1, context_size, vocab_size)

                logits = self.forward(context_one_hot)  # (1, vocab_size)
                next_idx = int(torch.argmax(logits, dim=-1).item())

                generated_indices.append(next_idx)
                sequence.append(next_idx)

        return generated_indices