import torch


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
