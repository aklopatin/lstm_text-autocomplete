import torch

from lstm_model import NNAutocomplete


def load_trained_model(
    model_path: str = "./models/lstm_autocomplete.pt",
) -> tuple[NNAutocomplete, dict, list[str], int, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    vocab_size: int = checkpoint["vocab_size"]
    stoi: dict = checkpoint["stoi"]
    itos: list[str] = checkpoint["itos"]
    context_size: int = checkpoint["context_size"]
    hidden_dim: int = checkpoint["hidden_dim"]
    embed_dim: int = checkpoint.get("embed_dim", hidden_dim)
    model = NNAutocomplete(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=vocab_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, stoi, itos, context_size, device


def generate_completion(
    model: NNAutocomplete,
    stoi: dict,
    itos: list[str],
    context_size: int,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 10,
) -> str:
    pad_idx = stoi.get("<pad>", 0)
    unk_idx = stoi.get("<unk>", 1)
    tokens = prompt.strip().split()
    if not tokens:
        return ""
    sequence = [stoi.get(tok, unk_idx) for tok in tokens]
    generated_indices: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            cur_context = sequence[-context_size:]
            if len(cur_context) < context_size:
                cur_context = [pad_idx] * (context_size - len(cur_context)) + cur_context
            context_tensor = torch.tensor(
                cur_context, dtype=torch.long, device=device
            ).unsqueeze(0)
            logits = model(context_tensor)
            next_idx = int(torch.argmax(logits, dim=-1).item())
            if next_idx == pad_idx:
                break
            generated_indices.append(next_idx)
            sequence.append(next_idx)
    generated_tokens = [itos[i] for i in generated_indices]
    return " ".join(generated_tokens)


def main() -> None:
    model, stoi, itos, context_size, device = load_trained_model()
    print(f"Модель загружена. Устройство: {device}")
    print("Введите начальную фразу (пустая строка — выход).")
    while True:
        try:
            prompt = input("\nФраза: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if not prompt:
            print("Выход.")
            break
        completion = generate_completion(
            model=model,
            stoi=stoi,
            itos=itos,
            context_size=context_size,
            device=device,
            prompt=prompt,
            max_new_tokens=10,
        )
        if completion:
            print(f"Продолжение: {completion}")
        else:
            print("Не удалось сгенерировать продолжение (проверьте словарь/ввод).")


if __name__ == "__main__":
    main()
