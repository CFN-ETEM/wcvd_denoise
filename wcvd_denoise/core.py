from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .wcvd import FFTCircularLowPass, WCVD


def default_weights_path() -> Path:
    """Return the path of the pretrained weights shipped with the package."""
    return Path(str(files("wcvd_denoise") / "weights" / "denoising_model_weights.pt"))


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(weights_path=None, device="auto") -> WCVD:
    """Build the WCVD model and load a state dictionary into it."""
    device = get_device(device) if isinstance(device, str) else device
    if weights_path is None:
        weights_path = default_weights_path()
    model = WCVD(num_slayers=4, mid_channels=3).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def is_single_frame(video: np.ndarray) -> bool:
    return video.ndim == 2 or (video.ndim == 3 and video.shape[0] == 1)


def repeat_single_frame(video: np.ndarray) -> np.ndarray:
    """Turn a single image (Nx, Ny) or (1, Nx, Ny) into a three-frame video.

    The model is intended for time-series videos. A single frame is replicated
    three times so the model can still produce one denoised frame.
    """
    video = np.asarray(video)
    if is_single_frame(video):
        video = np.repeat(video.reshape(1, *video.shape[-2:]), 3, axis=0)
    return video


def check_video(video: np.ndarray) -> torch.Tensor:
    if video.ndim != 3:
        raise ValueError(
            f"Input data must have shape (Nt, Nx, Ny), but received {video.shape}."
        )
    if video.shape[0] < 3:
        raise ValueError("Input data must contain at least three frames.")
    # Input intensities should use scaling similar to the model's training data.
    return torch.from_numpy(np.asarray(video, dtype=np.float32))


def denoise(video, model=None, batch_size=4, device="auto"):
    """Denoise a video with shape (Nt, Nx, Ny).

    Returns two arrays: (lowpass_ml, ml). lowpass_ml is the model output after
    the low-pass filter, as used in the paper; ml is the model output alone.

    Three input frames produce one denoised center frame. The outputs therefore
    contain Nt - 2 frames and exclude the first and last input frames.
    A single frame is replicated three times and gives one output frame.
    """
    device = get_device(device) if isinstance(device, str) else device
    video = check_video(repeat_single_frame(video))
    if model is None:
        model = load_model(device=device)
    model = model.to(device)
    model.eval()

    image_size = (video.shape[1], video.shape[2])
    lowpass = FFTCircularLowPass(Ns=3, img_size=image_size).to(device)
    number_of_windows = video.shape[0] - 2
    lowpass_batches = []
    ml_batches = []

    with torch.inference_mode():
        for start in range(0, number_of_windows, batch_size):
            stop = min(start + batch_size, number_of_windows)
            windows = torch.stack(
                [video[index:index + 3] for index in range(start, stop)]
            )
            windows = windows.unsqueeze(1).to(device)
            predictions = model(windows)
            lowpass_batches.append(lowpass(predictions)[:, 0, 0].cpu())
            ml_batches.append(predictions[:, 0, 0].cpu())

    lowpass_ml = torch.cat(lowpass_batches, dim=0).numpy()
    ml = torch.cat(ml_batches, dim=0).numpy()
    return lowpass_ml, ml


def retrain(
    video,
    model=None,
    number_of_epochs=300,
    learning_rate=0.001,
    batch_size=4,
    device="auto",
    log=print,
) -> WCVD:
    """Fine-tune the model on a noisy video without clean targets.

    The blind-spot model is trained to predict each center frame from
    neighboring spatial and temporal information.
    """
    device = get_device(device) if isinstance(device, str) else device
    if is_single_frame(np.asarray(video)):
        # Replicated neighbors would contain the center frame's own noise, so
        # the blind-spot training would learn to copy it instead of denoising.
        raise ValueError("Retraining requires a video with at least three frames.")
    video = check_video(video)
    if model is None:
        model = load_model(device=device)
    model = model.to(device)
    model.train()

    number_of_windows = video.shape[0] - 2
    image_size = (video.shape[1], video.shape[2])
    lowpass = FFTCircularLowPass(Ns=3, img_size=image_size).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=10)
    loss_function = nn.MSELoss()

    for epoch in range(1, number_of_epochs + 1):
        augmented_video = video

        # Transposition is used only for square images so the image size is unchanged.
        if video.shape[1] == video.shape[2] and np.random.random() > 0.5:
            augmented_video = torch.transpose(augmented_video, -2, -1)
        if np.random.random() > 0.5:
            augmented_video = torch.flip(augmented_video, (-1,))
        if np.random.random() > 0.5:
            augmented_video = torch.flip(augmented_video, (-2,))

        shuffled_indices = torch.randperm(number_of_windows)
        total_loss = 0.0

        for start in range(0, number_of_windows, batch_size):
            batch_indices = shuffled_indices[start:start + batch_size]
            index_list = batch_indices.tolist()
            windows = torch.stack(
                [augmented_video[index:index + 3] for index in index_list]
            )
            targets = augmented_video[batch_indices + 1]
            windows = windows.unsqueeze(1).to(device)
            targets = targets.unsqueeze(1).unsqueeze(2).to(device)

            optimizer.zero_grad()
            predictions = lowpass(model(windows))
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(index_list)

        mean_loss = total_loss / number_of_windows
        lr_scheduler.step(mean_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == number_of_epochs:
            current_learning_rate = optimizer.param_groups[0]["lr"]
            log(
                f"Epoch {epoch:3d}/{number_of_epochs}, "
                f"loss: {mean_loss:.6g}, learning rate: {current_learning_rate:.3g}"
            )

    model.eval()
    return model
