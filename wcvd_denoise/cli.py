import argparse
from pathlib import Path

import torch

from .core import denoise, get_device, is_single_frame, load_model, retrain
from .image_io import read_video, write_video

EXAMPLES = """
examples:
  denoise -i video.npy -o video_denoised.npy
  denoise -i "frames/name_*.png" -o denoised_frames/
  denoise -i frames/ -o denoised_stack.tif
  denoise -i stack.tif -o denoised/ --format png
  denoise -i video.npy -o out.npy --retrain --save-weights my_weights.pt
  denoise -i other_video.npy -o out2.npy --weights my_weights.pt
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="denoise",
        description=(
            "Weight Centric Video Denoising (WCVD) for HRTEM videos. Three "
            "consecutive frames produce one denoised center frame, so Nt input "
            "frames give Nt - 2 output frames."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, nargs="+",
        help="A .npy array (Nt, Nx, Ny), a multi-frame TIFF, a folder of "
             "PNG/TIFF frames, or a pattern such as 'name_*.png'. Frames are "
             "ordered by the numbers in their file names. A single image is "
             "replicated 3 times to form a video.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output .npy file, .tif/.tiff stack, or a folder for one image "
             "per frame. Two results are saved, with file names starting with "
             "lowpass_ml_ (ML + low-pass filter) and ml_ (ML only).",
    )
    parser.add_argument(
        "--format", choices=("tif", "png"), default="tif",
        help="Image format when --output is a folder (default: tif). TIFF keeps "
             "float32 values; PNG is rescaled to 16-bit using the min/max of "
             "the whole video.",
    )
    parser.add_argument(
        "-w", "--weights", default=None,
        help="Weight file (.pt) to use instead of the bundled pretrained "
             "weights. With --retrain these are the starting weights.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of three-frame windows per batch (default: 4). Lower it "
             "if you run out of memory.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'auto' (CUDA if available, otherwise CPU), 'cpu', 'cuda', "
             "'cuda:1', ... (default: auto)",
    )

    training = parser.add_argument_group("retraining")
    training.add_argument(
        "--retrain", action="store_true",
        help="Fine-tune the weights on the input video (self-supervised, no "
             "clean data needed) before denoising it.",
    )
    training.add_argument(
        "--epochs", type=int, default=300,
        help="Number of retraining epochs (default: 300).",
    )
    training.add_argument(
        "--lr", type=float, default=0.001,
        help="Retraining learning rate (default: 0.001).",
    )
    training.add_argument(
        "--save-weights", default=None,
        help="Where to save the retrained weights (.pt), so they can be reused "
             "with --weights.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.save_weights and not args.retrain:
        parser.error("--save-weights requires --retrain.")

    device = get_device(args.device)
    try:
        video, frame_names = read_video(args.input)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"denoise: error: {error}\n")
    single_frame = is_single_frame(video)
    if single_frame and args.retrain:
        parser.exit(
            1, "denoise: error: --retrain requires a video with at least three frames.\n"
        )
    if not single_frame and (video.ndim != 3 or video.shape[0] < 3):
        parser.exit(
            1,
            "denoise: error: input must be a single image or a video of at least "
            f"three frames with shape (Nt, Nx, Ny), but received {video.shape}.\n",
        )
    print(f"Device: {device}")
    print(f"Input shape: {video.shape}")
    if single_frame:
        print("Single frame: replicating it 3 times to form a video.")

    model = load_model(args.weights, device=device)

    if args.retrain:
        model = retrain(
            video,
            model,
            number_of_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=device,
        )
        if args.save_weights:
            weights_path = Path(args.save_weights)
            if weights_path.suffix != ".pt":
                weights_path = weights_path.with_name(weights_path.name + ".pt")
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            print(f"Saved retrained weights to: {weights_path}")

    lowpass_ml, ml = denoise(video, model, batch_size=args.batch_size, device=device)

    if single_frame:
        print(f"Output shape: {lowpass_ml.shape}")
    else:
        print(f"Output shape: {lowpass_ml.shape} (first and last frames are excluded)")

    # Output frame k is the center of input frames k, k+1, k+2.
    output_names = frame_names if single_frame else frame_names[1:-1]
    outputs = (
        ("lowpass_ml_", lowpass_ml, "ML denoising + low-pass filter"),
        ("ml_", ml, "ML denoising only"),
    )
    for prefix, denoised, label in outputs:
        saved = write_video(denoised, args.output, output_names, prefix, args.format)
        if len(saved) == 1:
            print(f"Saved {label} to: {saved[0]}")
        else:
            print(
                f"Saved {label} ({len(saved)} frames) to: "
                f"{Path(args.output) / prefix}*"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
