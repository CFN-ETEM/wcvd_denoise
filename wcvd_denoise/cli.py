import argparse
from pathlib import Path

import torch

from .core import denoise, get_device, load_model, retrain
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
             "ordered by the numbers in their file names.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output .npy file, .tif/.tiff stack, or a folder for one image "
             "per frame.",
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
    if video.ndim != 3 or video.shape[0] < 3:
        parser.exit(
            1,
            "denoise: error: input must be a video of at least three frames "
            f"with shape (Nt, Nx, Ny), but received {video.shape}.\n",
        )
    print(f"Device: {device}")
    print(f"Input shape: {video.shape}")

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

    denoised = denoise(video, model, batch_size=args.batch_size, device=device)

    # Output frame k is the center of input frames k, k+1, k+2.
    saved = write_video(denoised, args.output, frame_names[1:-1], args.format)
    print(f"Output shape: {denoised.shape} (first and last frames are excluded)")
    if len(saved) == 1:
        print(f"Saved denoised data to: {saved[0]}")
    else:
        print(f"Saved {len(saved)} denoised frames to: {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
