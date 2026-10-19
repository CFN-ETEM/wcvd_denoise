# wcvd-denoise

Command-line tool for denoising HRTEM videos with the Weight Centric Video Denoising (WCVD) model from:

Lee, B., Li, M., Yang, J.C. et al. Machine learning pipeline for denoising low signal-to-noise ratio and out-of-distribution transmission electron microscopy datasets. npj Comput Mater (2026). https://doi.org/10.1038/s41524-026-02193-9

The training and test data used in the paper are in the original repository: https://github.com/CFN-ETEM/HRTEM_denoising_model

Questions about this work can be directed to: blee2@bnl.gov

It supports:

- Denoising with the bundled pretrained weights.
- Self-supervised retraining on a user's own noisy video (no clean data needed).
- `.npy` arrays, multi-frame TIFF stacks, and folders or patterns of PNG/TIFF frames.
- Automatic use of an NVIDIA CUDA GPU when one is available, with CPU fallback.

## Installation

Python 3.9 or newer is required. Creating a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install the package from GitHub:

```bash
python -m pip install --upgrade pip
python -m pip install git+https://github.com/CFN-ETEM/wcvd_denoise.git
```

This installs the command-line tool:

```text
denoise
```

If a particular CUDA build of PyTorch is required, install it first by following https://pytorch.org/get-started/locally/, then install this package.

## Quick Start

Download the example video [`examples/au.npy`](examples/au.npy) (three frames of an HRTEM video of gold) and run:

```bash
denoise --input au.npy --output au_denoised.npy
```

This saves two results, `lowpass_ml_au_denoised.npy` and `ml_au_denoised.npy` (see [Two Outputs](#two-outputs-with-and-without-the-low-pass-filter)).

## Usage

```bash
denoise --input <input> --output <output> [options]
```

### Input

`--input` (`-i`) accepts:

| Input | Example |
|---|---|
| NumPy array with shape `(Nt, Nx, Ny)` | `-i video.npy` |
| Multi-frame TIFF stack | `-i video.tif` |
| Folder of PNG or TIFF frames | `-i frames/` |
| Pattern matching PNG or TIFF frames | `-i "frames/name_*.png"` |

Frames are ordered by the numbers in their file names, so `name_2.png` comes before `name_10.png`. Color images are converted to grayscale by averaging the color channels. 8-bit, 16-bit, and floating-point images are supported.

The model uses three consecutive frames to predict their center frame, so an input with `Nt` frames produces `Nt - 2` output frames. The first and last input frames are not included in the output.

The original model is intended to be used for time-series videos. If user supplies single frame image, the model replicates the image 3 times to preprocess it into a video to make the final output. For example, an input of shape `(512, 512)` or `(1, 512, 512)` gives an output of shape `(1, 512, 512)`. Retraining requires a video with at least three frames. Inputs with exactly two frames are not supported.

### Output

Every run saves two results, with file names starting with `lowpass_ml_` and `ml_` (see below). The type of `--output` (`-o`) is set by its extension:

| Output | Result |
|---|---|
| `out.npy` | `lowpass_ml_out.npy` and `ml_out.npy`, NumPy arrays with shape `(Nt - 2, Nx, Ny)` |
| `out.tif` | `lowpass_ml_out.tif` and `ml_out.tif`, multi-frame float32 TIFF stacks |
| `out_folder/` (no extension) | One image per frame for each result |

When writing to a folder, each output frame is named after its matching input frame. For example, `name_2.png` becomes `out_folder/lowpass_ml_name_2.tif` and `out_folder/ml_name_2.tif`. Frames are saved as float32 TIFF by default. Add `--format png` to save 16-bit PNG instead; PNG frames are rescaled to 0-65535 using the minimum and maximum of the whole video, so absolute intensities are not preserved.

### Two Outputs: With and Without the Low-Pass Filter

After the ML model, a circular low-pass filter in Fourier space removes spatial frequencies above a fixed cutoff (detail finer than about 6 pixels), as in the paper. Both versions are saved so you can compare them:

- `lowpass_ml_*`: ML denoising followed by the low-pass filter. This is the result reported in the paper and usually the cleanest.
- `ml_*`: ML denoising only.

The low-pass filter can hurt when real structure, such as a small lattice spacing, is close to or finer than the cutoff. In that case the `ml_*` result keeps that detail. However, the model was trained with the low-pass filter applied to its output, so frequencies above the cutoff were not constrained during training. The `ml_*` result can therefore contain some high-frequency noise or artifacts; this does not by itself mean the denoising failed. Check both results, and their Fourier transforms, before deciding which to use.

### Options

| Option | Default | Description |
|---|---|---|
| `-w`, `--weights` | bundled weights | Weight file (`.pt`) to use, e.g. from retraining |
| `--batch-size` | `4` | Three-frame windows per batch. Lower it if you run out of memory |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, ... |
| `--format` | `tif` | `tif` or `png`, used when `--output` is a folder |
| `--retrain` | off | Fine-tune the weights on the input video before denoising |
| `--epochs` | `300` | Retraining epochs |
| `--lr` | `0.001` | Retraining learning rate (Adam with `ReduceLROnPlateau`) |
| `--save-weights` | none | Where to save the retrained weights |

Run `denoise --help` for the full list.

## Retraining

For data that differ from what the pretrained model was trained on, retraining fine-tunes the weights on your own noisy video. Clean target data are not required: the blind-spot model learns to predict each center frame from neighboring spatial and temporal information.

```bash
denoise -i "frames/name_*.tif" -o denoised/ --retrain --save-weights my_weights.pt
```

This retrains for 300 epochs, saves the weights to `my_weights.pt`, and denoises the video. The saved weights can be reused on other videos:

```bash
denoise -i "other/name_*.tif" -o other_denoised/ --weights my_weights.pt
```

Retraining is much faster on a GPU.

## Intensity Scaling

The network has no bias terms, so multiplying the input by a constant multiplies the output by the same constant. Images exported as 8-bit or 16-bit PNG/TIFF can therefore be used without matching the intensity units of the training data. A constant offset (for example, background subtraction or contrast stretching applied on export) is not removed by this property and can change the result. When possible, use raw or linearly scaled frames.

## Python API

The same functions can be used from Python:

```python
import numpy as np
from wcvd_denoise import denoise, load_model, retrain, read_video

video, names = read_video(["frames/name_*.png"])   # shape (Nt, Nx, Ny)
lowpass_ml, ml = denoise(video)                   # each (Nt - 2, Nx, Ny)

model = retrain(video, load_model(), number_of_epochs=300)
lowpass_ml, ml = denoise(video, model)
```
