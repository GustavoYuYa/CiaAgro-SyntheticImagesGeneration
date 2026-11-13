from utils.seed import set_seed

from pathlib import Path

import os
import argparse
import random

set_seed(42)

def main():

    parser = argparse.ArgumentParser(
        description="Prepare dataset"
    )

    # parser.add_argument("--class", choices=["mildio", "oidio", "phakopsora"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-path", type=Path, required=True)
    args = parser.parse_args()

    os.makedirs(args.splits_path, exist_ok=True)

    classes = [
        "Peronospora_manshurica",
        "Microsphaera_diffusa",
        "Phakopsora_pachyrhizi"
    ]

    nums = [100, 300, 500, 700, 1000, 1300]

    # Garantir que pega apenas imagens
    extensions = (".png", ".jpg", ".jpeg")

    SEED = 42

    for classe in classes:

        dataset_folder = os.path.join(
            args.dataset_root,
            classe
        )

        split_folder = os.path.join(
            args.splits_path,
            classe
        )

        os.makedirs(split_folder, exist_ok=True)

        all_images = []

        for subclass in os.listdir(dataset_folder):

            dataset_subfolder = os.path.join(dataset_folder, subclass)

            for file in os.listdir(dataset_subfolder):

                all_images.append(os.path.join(dataset_subfolder, file))

        all_images.sort()

        rng = random.Random(SEED)
        rng.shuffle(all_images)

        print(f"\nClass: {classe}")
        print(f"Available images: {len(all_images)}")

        max_num = max(nums)

        if len(all_images) < max_num:
            print(
                f"WARNING: {classe} has only {len(all_images)} images, "
                f"but {max_num} were requested."
            )

        for num in nums:

            # Ele pega as primeiras imagens até o limitador (num)
            selected = all_images[:min(num, len(all_images))]

            output_file = os.path.join(
                split_folder,
                f"{num}.txt"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                for filename in selected:
                    f.write(filename + "\n")

            print(
                f"  {num:4d} images -> {len(selected):4d} saved -> "
                f"{output_file}"
            )

if __name__ == "__main__":
    main()
    