import argparse
import yaml
import torch

from pathlib import Path
from tqdm.auto import tqdm
from torchvision.utils import save_image

from utils.seed import set_seed
from src.model import build_model


def main():

    parser = argparse.ArgumentParser(
        description="Generate synthetic images"
    )

    parser.add_argument("--spore-class", choices=["Peronospora_manshurica", "Microsphaera_diffusa", "Phakopsora_pachyrhizi"], required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, choices=[100, 300, 500, 700, 1000, 1300], required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", type=Path, default=Path("configs/ddpm.yaml"))

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    checkpoint_path = args.checkpoint_path

    output_folder = Path("outputs") / args.spore_class / f"{args.dataset_size}"
    output_folder.mkdir(parents=True, exist_ok=True)

    diffusion = build_model(config=config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    ema_state = checkpoint["ema"]

    prefix = "ema_model."

    ema_weights = {
        key[len(prefix):]: value 
        for key, value in ema_state.items()
        if key.startswith(prefix)
    }

    if not ema_weights:
        raise ValueError("EMA weights not found")

    diffusion.load_state_dict(ema_weights, strict=True)
    diffusion = diffusion.to(device)
    diffusion.eval()

    start = args.start_index
    end = args.start_index + args.num_samples
    
    with torch.inference_mode():
        with tqdm(
            total=args.num_samples,
            desc="Generating images",
            unit="img"
        ) as pbar:
            
            for i in range(start, end, args.batch_size):
                n = min(args.batch_size, end - i)

                images = diffusion.sample(batch_size=n)

                for j, img in enumerate(images):
                    index = i + j
                    
                    subfolder = (index - 1) // 5000
                    output_subfolder = output_folder / f"{subfolder:03d}"
                    output_subfolder.mkdir(parents=True, exist_ok=True)

                    path = output_subfolder / f"{args.spore_class}_{(i + j):05d}.png"
                    save_image(img.clamp(0, 1), path)

                pbar.update(n)


if __name__ == "__main__":
    main()
