"""Main experiment protocols for Thousand, YouTube Faces, and Extended Yale B."""

from collections import Counter, defaultdict
from statistics import mean

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from .metrics import evaluate, evaluate_crop_votes
from .protection import enroll
from .training import data_loader, encode_batch, encode_samples, experiment_device, train_model, training_configuration
from .yale import YALE_CROP_COUNT, yale_crops


class YaleProbes(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.targets = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            crops = [self.transform(crop) for crop in yale_crops(image.convert("RGB"))]
        return torch.stack(crops), label


def _image_datasets(root, name):
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    tensor = transforms.ToTensor()
    if name == "thousand":
        train_dir = root / "train"
        size = 48
        resize = transforms.Resize((52, 52))
        train_transform = transforms.Compose([
            resize, transforms.ColorJitter(brightness=0.1), transforms.RandomCrop(size), tensor,
            transforms.RandomErasing(1, (0.015, 0.02), (0.5, 2), 0), normalize,
        ])
        eval_transform = transforms.Compose([resize, transforms.CenterCrop(size), tensor, normalize])
    else:
        train_dir = root / "preprocess_train"
        size = 56
        resize = transforms.Resize((size, size))
        train_transform = transforms.Compose([
            resize, transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(), tensor, normalize,
        ])
        eval_transform = transforms.Compose([resize, tensor, normalize])

    training = datasets.ImageFolder(train_dir, transform=train_transform)
    enrollment = datasets.ImageFolder(train_dir, transform=eval_transform)
    testing = datasets.ImageFolder(root / "test", transform=eval_transform)
    if training.classes != testing.classes:
        raise ValueError("Training and testing must contain the same identity folders.")
    if name == "yaleb":
        testing = YaleProbes(testing.samples, eval_transform)
    return training, enrollment, testing, training.targets, testing.targets, size


def _youtube_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        images = archive["images"]
        labels = archive["labels"].reshape(-1)
    if images.shape != (len(labels), 3, 32, 32) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected NCHW 32 px RGB images and integer labels in {path.name}.")
    images = images.astype(np.float32, copy=False)
    center = images.mean(dtype=np.float64)
    scale = images.std(dtype=np.float64)
    images -= center
    images *= 0.1 / (scale + 1e-5)
    np.clip(images, -1.0, 1.0, out=images)
    return torch.from_numpy(images), labels


def _youtube_datasets(root):
    train_images, train_ids = _youtube_arrays(root / "train.npz")
    test_images, test_ids = _youtube_arrays(root / "test.npz")
    classes, train_labels = np.unique(train_ids, return_inverse=True)
    if not np.array_equal(np.unique(test_ids), classes):
        raise ValueError("YouTube training and testing must contain the same identities.")
    test_labels = np.searchsorted(classes, test_ids)
    training = TensorDataset(train_images, torch.from_numpy(train_labels))
    testing = TensorDataset(test_images, torch.from_numpy(test_labels))
    return training, training, testing, train_labels.tolist(), test_labels.tolist(), 32


@torch.inference_mode()
def _yale_codes(model, loader, device):
    model.eval()
    for crops, labels in loader:
        packed = encode_batch(model, crops.flatten(0, 1), device).reshape(len(labels), YALE_CROP_COUNT, -1)
        for views, label in zip(packed, labels.tolist()):
            yield [view.tobytes() for view in views], label


def _run_seed(training, enrollment, testing, count, size, args, device, seed):
    model = train_model(training, args, count, device, seed, image_size=size)
    enrollment_loader = data_loader(enrollment, args)
    references = {label: enroll(code) for code, label in encode_samples(model, enrollment_loader, device)}
    if args.command == "yaleb":
        testing_loader = data_loader(testing, args, batch_size=max(1, args.batch_size // YALE_CROP_COUNT))
        return evaluate_crop_votes(_yale_codes(model, testing_loader, device), references, args.minimum_crop_votes)

    testing_loader = data_loader(testing, args)
    probes = defaultdict(list)
    for code, label in encode_samples(model, testing_loader, device):
        probes[label].append(code)
    return evaluate(probes, references)


def run(args):
    device = experiment_device(args)
    name = args.command
    if name == "yaleb" and not 1 <= args.minimum_crop_votes <= YALE_CROP_COUNT:
        raise ValueError(f"Minimum crop votes must be between 1 and {YALE_CROP_COUNT}.")
    if name == "youtube":
        training, enrollment, testing, train_labels, test_labels, size = _youtube_datasets(args.data_root)
        if set(Counter(train_labels).values()) != {40} or set(Counter(test_labels).values()) != {5}:
            raise ValueError("YouTube requires 40 training and 5 test images per identity.")
    else:
        training, enrollment, testing, train_labels, test_labels, size = _image_datasets(args.data_root, name)
        if name == "thousand" and set(Counter(test_labels).values()) != {2}:
            raise ValueError("Thousand requires two test images per eye.")

    first_per_class = {}
    for index, label in enumerate(train_labels):
        first_per_class.setdefault(label, index)
    count = len(first_per_class)
    if count < 2:
        raise ValueError("Provide at least two enrolled identities.")
    enrollment = Subset(enrollment, list(first_per_class.values()))
    configuration = {
        **training_configuration(args, device), "input_size": size,
        "identities": count, "training_images": len(training), "test_images": len(testing),
    }
    if name == "yaleb":
        configuration.update(
            crops_per_probe=YALE_CROP_COUNT, minimum_crop_votes=args.minimum_crop_votes,
        )
    results = [
        {"seed": seed, **_run_seed(training, enrollment, testing, count, size, args, device, seed)}
        for seed in args.seeds
    ]
    keys = ("gar_percent", "far_percent")
    return {"configuration": configuration, "runs": results, "mean": {key: mean(r[key] for r in results) for key in keys}}
