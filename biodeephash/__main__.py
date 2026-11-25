"""Command-line entry points for the standalone demo and benchmark experiments."""

import argparse
import json
from pathlib import Path

from .protection import enroll, verify


EXPERIMENT_DEFAULTS = {
    "lamp": {"epochs": 350, "variance": 16.0, "lr_step": 50, "workers": 4},
    "thousand": {"epochs": 350, "variance": 8.0, "lr_step": 50, "workers": 4},
    "youtube": {"epochs": 350, "variance": 16.0, "lr_step": 50, "workers": 0},
    "yaleb": {"epochs": 60, "variance": 8.0, "lr_step": 30, "workers": 4, "minimum_crop_votes": 1},
}


def demo():
    code = bytes.fromhex("0123456789abcdeffedcba98")
    different_code = code[:-1] + bytes([code[-1] ^ 1])
    reference = enroll(code)
    replacement = enroll(code)
    result = {
        "same_code_accepted": verify(code, reference),
        "one_bit_change_rejected": not verify(different_code, reference),
        "reissued_digest_changed": reference.digest != replacement.digest,
        "reissued_template_accepted": verify(code, replacement),
    }
    if not all(result.values()):
        raise RuntimeError(f"Template demo failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(prog="python -m biodeephash", description="BioDeepHash research implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run the standard-library template demo")
    for dataset, defaults in EXPERIMENT_DEFAULTS.items():
        experiment = commands.add_parser(
            dataset, help=f"Train and evaluate the {dataset} experiment",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        experiment.add_argument("--data-root", type=Path, required=True)
        experiment.add_argument("--seeds", type=int, nargs="+", default=[0], help="Independent training seeds")
        experiment.add_argument("--objective", choices=("full", "classwise-quantization", "classwise"), default="full")
        experiment.add_argument("--bits", type=int, choices=(96, 120, 144), default=96)
        experiment.add_argument("--epochs", type=int, help="Training epochs")
        experiment.add_argument("--batch-size", type=int, default=512)
        experiment.add_argument("--learning-rate", type=float, default=0.005)
        experiment.add_argument("--lr-step", type=int, help="Epochs between learning-rate halvings")
        experiment.add_argument("--variance", type=float, help="Gaussian class-wise variance")
        experiment.add_argument("--eta", type=float, default=0.02)
        experiment.add_argument("--workers", type=int, help="Data-loading workers")
        experiment.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if dataset == "yaleb":
            experiment.add_argument("--minimum-crop-votes", type=int, help="Minimum matching crops (1-64)")
        experiment.set_defaults(**defaults)
    args = parser.parse_args()

    if args.command == "demo":
        result = demo()
    else:
        if args.command == "lamp":
            from .lamp import run
        else:
            from .benchmarks import run

        try:
            result = run(args)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
