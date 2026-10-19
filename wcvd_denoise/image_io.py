import glob
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

IMAGE_SUFFIXES = (".png", ".tif", ".tiff")
TIFF_SUFFIXES = (".tif", ".tiff")


def natural_sort_key(path):
    """Sort name_2 before name_10 instead of plain alphabetical order."""
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Average the color channels of RGB(A) or LA images; drop alpha."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (2, 3, 4):
        if image.shape[-1] in (2, 4):
            image = image[..., :-1]
        return image.mean(axis=-1)
    raise ValueError(f"Unsupported image shape {image.shape}.")


def read_image(path: Path) -> np.ndarray:
    """Read one PNG or TIFF image as a single 2D grayscale frame."""
    if path.suffix.lower() in TIFF_SUFFIXES:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            image = series.asarray()
            if "S" not in series.axes and image.ndim == 3:
                raise ValueError(
                    f"{path} contains {image.shape[0]} frames. Pass a "
                    "multi-frame TIFF on its own instead of in a list of images."
                )
    else:
        with Image.open(path) as img:
            if img.mode == "P":
                img = img.convert("RGB")
            image = np.asarray(img)
    return to_grayscale(image)


def read_tiff_stack(path: Path) -> np.ndarray:
    """Read a multi-frame TIFF with shape (Nt, Nx, Ny)."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        video = series.asarray()
        if "S" in series.axes:
            # Move color samples to the last axis and convert them to gray.
            video = np.moveaxis(video, series.axes.index("S"), -1)
            if video.shape[-1] in (2, 4):
                video = video[..., :-1]
            video = video.mean(axis=-1)
    return video


def expand_inputs(inputs):
    """Turn --input values (files, folders, or glob patterns) into paths.

    Glob patterns are expanded here so both quoted ("name_*.png") and
    shell-expanded (name_*.png) patterns work.
    """
    paths = []
    for item in inputs:
        if glob.has_magic(item):
            matches = [Path(match) for match in glob.glob(item)]
            if not matches:
                raise FileNotFoundError(f"No files match {item}.")
            paths.extend(matches)
        else:
            path = Path(item)
            if path.is_dir():
                matches = [
                    child for child in path.iterdir()
                    if child.suffix.lower() in IMAGE_SUFFIXES
                ]
                if not matches:
                    raise FileNotFoundError(f"No PNG or TIFF images found in {path}.")
                paths.extend(matches)
            elif path.exists():
                paths.append(path)
            else:
                raise FileNotFoundError(f"{path} does not exist.")
    return sorted(set(paths), key=natural_sort_key)


def read_video(inputs):
    """Read the input as a video with shape (Nt, Nx, Ny).

    Returns the video and a list of frame names (one per input frame) that is
    used to name per-frame output files.
    """
    paths = expand_inputs(inputs)

    if len(paths) == 1:
        path = paths[0]
        suffix = path.suffix.lower()
        if suffix == ".npy":
            video = np.load(path)
        elif suffix in TIFF_SUFFIXES:
            video = read_tiff_stack(path)
        else:
            video = read_image(path)
        if video.ndim == 2 or len(video) == 1:
            return video, [path.stem]
        names = [f"{path.stem}_{index:05d}" for index in range(len(video))]
        return video, names

    unsupported = [path for path in paths if path.suffix.lower() not in IMAGE_SUFFIXES]
    if unsupported:
        raise ValueError(
            f"Only PNG and TIFF images can be combined into a video: {unsupported[0]}"
        )

    frames = [read_image(path) for path in paths]
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"All frames must have the same size, but found {shapes}.")
    return np.stack(frames), [path.stem for path in paths]


def to_uint16(video: np.ndarray) -> np.ndarray:
    """Rescale the whole video to 0-65535 with one min/max for all frames."""
    low, high = float(video.min()), float(video.max())
    scale = 65535.0 / (high - low) if high > low else 0.0
    return np.round((video - low) * scale).astype(np.uint16)


def write_video(video: np.ndarray, output, frame_names, prefix, image_format="tif"):
    """Save the denoised video with prefix added to every file name.

    output ending in .npy      -> NumPy array, e.g. out/<prefix>name.npy
    output ending in .tif/.tiff -> one float32 multi-frame TIFF
    anything else              -> folder with one image per frame,
                                  e.g. out/<prefix><frame name>.tif
    """
    output = Path(output)
    suffix = output.suffix.lower()
    output.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".npy":
        path = output.with_name(prefix + output.name)
        np.save(path, video)
        return [path]
    if suffix in TIFF_SUFFIXES:
        path = output.with_name(prefix + output.name)
        tifffile.imwrite(path, video.astype(np.float32))
        return [path]

    output.mkdir(parents=True, exist_ok=True)
    saved = []
    if image_format == "png":
        video = to_uint16(video)
    for frame, name in zip(video, frame_names):
        if image_format == "png":
            path = output / f"{prefix}{name}.png"
            Image.fromarray(frame).save(path)
        else:
            path = output / f"{prefix}{name}.tif"
            tifffile.imwrite(path, frame.astype(np.float32))
        saved.append(path)
    return saved
