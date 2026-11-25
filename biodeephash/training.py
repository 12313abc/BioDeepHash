"""Shared DFH training, runtime settings, and binary encoding."""

import sys
from math import isfinite

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dfh import DFHLoss, DFHNet


def experiment_device(args):
    if not args.data_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {args.data_root}")
    if args.batch_size < 2 or args.epochs < 1 or args.workers < 0:
        raise ValueError("Use a batch size of at least two, positive epochs, and nonnegative workers.")
    if args.lr_step < 1:
        raise ValueError("The learning-rate step must be positive.")
    values = (args.variance, args.eta, args.learning_rate)
    if not all(map(isfinite, values)) or args.variance <= 0 or args.eta < 0 or args.learning_rate <= 0:
        raise ValueError("Use positive variance and learning rate, and nonnegative eta.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Use distinct seeds in the range 0 through 2^32 - 1.")

    has_cuda = torch.cuda.is_available()
    if args.device == "cuda" and not has_cuda:
        raise ValueError("CUDA is unavailable. Use --device cpu or a CUDA-enabled PyTorch installation.")
    device = torch.device(("cuda" if has_cuda else "cpu") if args.device == "auto" else args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device


def data_loader(dataset, args, *, seed=None, batch_size=None):
    size = args.batch_size if batch_size is None else batch_size
    if seed is None:
        return DataLoader(dataset, batch_size=size, num_workers=args.workers)
    return DataLoader(
        dataset,
        batch_size=size,
        num_workers=args.workers,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=len(dataset) % size == 1,
    )


def train_model(dataset, args, num_classes, device, seed, *, image_size):
    torch.manual_seed(seed)
    loader = data_loader(dataset, args, seed=seed)
    model = DFHNet(args.bits, image_size=image_size).to(device)
    objective = DFHLoss(num_classes, args.bits, args.variance, args.eta, args.objective).to(device)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "amsgrad": True}, {"params": objective.parameters()}],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = scheduler.get_last_lr()[0]
        total_loss = 0.0
        sample_count = 0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = objective(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            sample_count += labels.numel()
        scheduler.step()
        print(
            f"Seed {seed}, epoch {epoch}/{args.epochs}, loss {total_loss / sample_count:.6f}, lr {learning_rate:.8g}",
            file=sys.stderr,
        )
    return model


def encode_batch(model, images, device):
    bits = (model(images.to(device)) > 0).to(torch.uint8).cpu().numpy()
    return np.packbits(bits, axis=1, bitorder="big")


@torch.inference_mode()
def encode_samples(model, loader, device):
    model.eval()
    for images, labels in loader:
        for row, label in zip(encode_batch(model, images, device), labels.tolist()):
            yield row.tobytes(), label


def training_configuration(args, device):
    return {
        "dataset": args.command,
        "code_bits": args.bits,
        "objective": args.objective,
        "loss_numerics": "legacy",
        "variance": args.variance,
        "eta": args.eta,
        "learning_rate": args.learning_rate,
        "lr_step": args.lr_step,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "workers": args.workers,
        "device": str(device),
    }
