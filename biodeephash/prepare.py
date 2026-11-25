"""Prepare user-supplied benchmark data for the experiment entry points."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory


IRIS_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _prepare_lamp(source, output):
    inputs = []
    for split in ("test", "train"):
        directory = source / split
        if not directory.is_dir():
            raise ValueError(f"Missing Lamp split directory: {directory}")
        inputs.extend(
            (split, path) for path in sorted(directory.glob("*/*"))
            if path.is_file() and path.suffix.lower() in IRIS_SUFFIXES
        )
    if not inputs:
        raise ValueError("No prepared Lamp images found.")
    for split, image in inputs:
        target = output / image.parent.name / f"{split}_{image.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(image, target)
    return {"images": len(inputs), "eye_classes": len({image.parent.name for _, image in inputs})}


def _prepare_yaleb(source, output):
    from PIL import Image

    from .yale import YALE_CROP_COUNT, yale_crops

    splits = {}
    for split in ("train", "test"):
        directory = source / split
        splits[split] = sorted(path for path in directory.glob("*/*.pgm") if path.is_file())
        if not splits[split]:
            raise ValueError(f"Expected Yale B PGM images in {directory}/<identity>/.")
    training, testing = splits["train"], splits["test"]
    if {path.parent.name for path in training} != {path.parent.name for path in testing}:
        raise ValueError("Yale B training and testing must contain the same identity folders.")
    train_names = {path.relative_to(source / "train") for path in training}
    if any(path.relative_to(source / "test") in train_names for path in testing):
        raise ValueError("Yale B training and test paths must be disjoint.")

    for image in training:
        destination = output / "preprocess_train" / image.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as original:
            for crop_index, crop in enumerate(yale_crops(original.convert("L"))):
                crop.save(destination / f"_{image.stem}_{crop_index}.pgm")

    for image in testing:
        target = output / "test" / image.relative_to(source / "test")
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(image, target)

    return {
        "training_images": len(training),
        "training_crops": len(training) * YALE_CROP_COUNT,
        "test_images": len(testing),
    }


def _prepare_youtube(source, output):
    import numpy as np

    summary = {}
    for split, expected_count in (("train", 40), ("test", 5)):
        image_shards, label_shards = [], []
        for shard in sorted((source / split).glob("*.pkl")):
            with shard.open("rb") as stream:
                images, labels = pickle.load(stream, encoding="bytes")
            labels = np.asarray(labels).reshape(-1)
            identities = labels.astype(np.int64)
            if not np.array_equal(labels, identities):
                raise ValueError(f"Expected integer identity labels in {shard.name}.")
            image_shards.append(np.asarray(images, dtype=np.float32).reshape(len(labels), 3, 32, 32))
            label_shards.append(identities)
        if not image_shards:
            raise ValueError(f"No YouTube Faces pickle shards found under {source / split}.")
        labels = np.concatenate(label_shards)
        counts = Counter(labels)
        if set(counts.values()) != {expected_count}:
            raise ValueError(f"The {split} split must contain {expected_count} images per identity.")
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / f"{split}.npz", images=np.concatenate(image_shards), labels=labels)
        summary[split] = {"images": len(labels), "identities": len(counts)}
    return summary


def main():
    parser = argparse.ArgumentParser(
        prog="python -m biodeephash.prepare", description=__doc__,
        epilog="Lamp and Yale B take train/<identity>/ and test/<identity>/ inputs. "
               "YouTube Faces takes trusted pickle shards in train/ and test/.",
    )
    formats = {"lamp": _prepare_lamp, "yaleb": _prepare_yaleb, "youtube-from-pickle": _prepare_youtube}
    parser.add_argument("format", choices=formats)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"Source directory does not exist: {args.source}")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".biodeephash-", dir=args.output.parent.resolve()) as staging:
            temporary_output = Path(staging) / "data"
            summary = formats[args.format](args.source, temporary_output)
            temporary_output.rename(args.output)
    except (ValueError, OSError, pickle.UnpicklingError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
