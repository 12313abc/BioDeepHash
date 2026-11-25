# BioDeepHash

Reference implementation of *BioDeepHash: Generating Consistent Templates for Secure Biometric Recognition*. The code includes Deep Fuzzy Hashing (DFH), protected-template generation, and the main iris and face verification experiments. [Paper](https://doi.org/10.1109/TDSC.2026.3660294)

## Quick start

Run all commands from the repository root. The synthetic demo uses Python's standard library and checks template issuance, exact matching, rejection after a one-bit change, and reissuance.

```sh
python -m biodeephash demo
```

## Installation

The experiment environment uses Python 3.10, PyTorch 2.2.2, and torchvision 0.17.2. For CUDA 12.1, install the matching PyTorch wheels first. 

```sh
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Prepare data

| Dataset | Input to `--data-root` |
| --- | --- |
| [CASIA-Iris-Lamp](https://hycasia.github.io/dataset/casia-irisv4/) | Prepared crops grouped as `<subject>_L/` and `<subject>_R/`, or `<subject>/L/` and `<subject>/R/`. |
| [CASIA-Iris-Thousand](https://hycasia.github.io/dataset/casia-irisv4/) | Prepared crops in `train/<eye>/` and `test/<eye>/`, with two test images per eye. |
| [YouTube Faces](https://www.cs.tau.ac.il/~wolf/ytfaces/) | `train.npz` and `test.npz` with `images` in `N × 3 × 32 × 32` order and integer `labels`. Use 40 training and 5 test images per identity. |
| [Extended Yale B](https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b) | Apply illumination normalization before preparing the training crops in `preprocess_train/<identity>/` and test images in `test/<identity>/`. |

Use `python -m biodeephash.prepare --help` for optional conversion of prepared Lamp images, YouTube Faces shards, or Yale B PGM images from an existing train/test split.

## Run the main experiments

Each command trains and evaluates BioDeepHash on the selected dataset. Training progress goes to standard error, and the JSON report goes to standard output.

```sh
python -m biodeephash lamp --data-root data/Lamp
python -m biodeephash thousand --data-root data/Thousand
python -m biodeephash youtube --data-root data/YouTubeFaces
python -m biodeephash yaleb --data-root data/ExtendedYaleB
```

Refer to the paper for the experiment settings. Run `python -m biodeephash <dataset> --help` to see the available options.

## Citation

```bibtex
@article{song2026biodeephash,
  title   = {BioDeepHash: Generating Consistent Templates for Secure Biometric Recognition},
  author  = {Song, Baogang and Zhao, Dongdong and Yan, Jiang and Li, Huanhuan and Jiang, Hao},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  year    = {2026},
  volume  = {23},
  number  = {3},
  pages   = {6019--6036},
  doi     = {10.1109/TDSC.2026.3660294}
}
```

## License

This repository includes the MIT license and its copyright notice. [License](LICENSE)
