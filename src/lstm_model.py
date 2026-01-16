import torch
import torch.nn as nn


class NNAutocomplete(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(x)
        output, _ = self.rnn(embeddings)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits

    def generate(
        self,
        context_indices: list[int],
        context_size: int,
        max_new_tokens: int,
        pad_idx: int,
        device: torch.device,
    ) -> list[int]:
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
                ).unsqueeze(0)
                logits = self.forward(context_tensor)
                next_idx = int(torch.argmax(logits, dim=-1).item())
                generated_indices.append(next_idx)
                sequence.append(next_idx)
        return generated_indices
