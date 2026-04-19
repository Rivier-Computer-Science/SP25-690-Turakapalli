import argparse
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from PIL import Image
import torchvision.transforms as T

from models.residual_cnn import ResidualCNN
from models.baseline_cnn import BaselineCNN


def load_image(path: str, image_size: int = 128):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # [1, C, H, W]


def save_image(tensor: torch.Tensor, path: str):
    tensor = tensor.clamp(0, 1).squeeze(0)
    img = T.ToPILImage()(tensor.cpu())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def get_model(name: str, checkpoint: str = None, device: str = "cpu"):
    if name == "residual":
        model = ResidualCNN()
    else:
        model = BaselineCNN()
    model.to(device)

    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        print(f"Loaded weights from {checkpoint}")
    else:
        print("Warning: No checkpoint loaded, using random weights.")

    model.eval()
    return model


def main(args):
    print(f"Input image: {args.input}")
    print(f"Output path: {args.output}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.image_size}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided, using random weights")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = get_model(args.model, args.checkpoint, device)

    noisy = load_image(args.input, args.image_size).to(device)
    print(f"Loaded noisy image shape: {noisy.shape}")

    with torch.no_grad():
        denoised = model(noisy)
    print(f"Denoised image shape: {denoised.shape}")

    save_image(denoised, args.output)
    print(f"Denoised image saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="residual", choices=["baseline", "residual"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=128)

    args = parser.parse_args()
    main(args)