"""Train DFH and measure protected-template verification on CASIA-Iris-Lamp."""

import random
from collections import defaultdict
from statistics import mean

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .metrics import evaluate
from .protection import enroll
from .training import data_loader, encode_samples, experiment_device, train_model, training_configuration


TEST_IMAGES_PER_EYE = 4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _eye_images(root):
    classes = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        eyes = sorted(path for path in folder.iterdir() if path.is_dir() and path.name.upper() in {"L", "R"})
        if eyes:
            candidates = {f"{folder.name}_{eye.name.upper()}": eye for eye in eyes}
        elif folder.name.upper().endswith(("_L", "_R")):
            candidates = {folder.name[:-1] + folder.name[-1].upper(): folder}
        else:
            continue
        for identity, eye in candidates.items():
            images = sorted(path for path in eye.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if identity in classes:
                raise ValueError(f"Duplicate eye class: {identity}.")
            if images:
                classes[identity] = images
    return classes


def _split(classes, seed):
    rng = random.Random(seed)
    training, enrollment, testing = [], [], []
    for label, identity in enumerate(sorted(classes)):
        images = classes[identity]
        held_out = set(rng.sample(images, TEST_IMAGES_PER_EYE))
        train_images = [path for path in images if path not in held_out]
        training.extend((path, label) for path in train_images)
        enrollment.append((train_images[0], label))
        testing.extend((path, label) for path in images if path in held_out)
    return training, enrollment, testing


class IrisImages(Dataset):
    def __init__(self, samples, *, augment):
        self.samples = samples
        operations = [transforms.Resize((56, 56))]
        if augment:
            operations.extend((transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)))
        operations.extend((transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)))
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def _run_seed(classes, args, device, seed):
    training, enrollment, testing = _split(classes, seed)
    model = train_model(
        IrisImages(training, augment=True), args, len(classes), device, seed, image_size=56,
    )
    enrollment_loader = data_loader(IrisImages(enrollment, augment=False), args)
    testing_loader = data_loader(IrisImages(testing, augment=False), args)
    references = {label: enroll(code) for code, label in encode_samples(model, enrollment_loader, device)}
    probes = defaultdict(list)
    for code, label in encode_samples(model, testing_loader, device):
        probes[label].append(code)
    return {"seed": seed, **evaluate(probes, references)}


def run(args):
    device = experiment_device(args)
    all_classes = _eye_images(args.data_root)
    classes = {identity: images for identity, images in all_classes.items() if len(images) > TEST_IMAGES_PER_EYE}
    if len(classes) < 2:
        raise ValueError("Provide at least two eye classes, with at least five images per eye.")

    results = [_run_seed(classes, args, device, seed) for seed in args.seeds]
    return {
        "configuration": {
            **training_configuration(args, device),
            "eye_classes": len(classes),
            "excluded_eye_classes": len(all_classes) - len(classes),
            "training_images": sum(len(images) - TEST_IMAGES_PER_EYE for images in classes.values()),
            "test_images_per_eye": TEST_IMAGES_PER_EYE,
        },
        "runs": results,
        "mean": {key: mean(result[key] for result in results) for key in ("gar_percent", "far_percent", "consistency_rate")},
    }
