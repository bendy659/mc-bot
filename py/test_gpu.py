"""Проверка, что PyTorch встал и видит GPU (если он есть)."""

import torch


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Устройство: {torch.cuda.get_device_name(0)}")
        # Мини-проверка реального вычисления на GPU.
        x = torch.rand(64, 4, 8, 8, device="cuda")
        print(f"Тестовый тензор на GPU: {x.sum().item():.1f} — OK")
    else:
        print("GPU не найден — обучение пойдёт на CPU (будет медленнее).")


if __name__ == "__main__":
    main()
