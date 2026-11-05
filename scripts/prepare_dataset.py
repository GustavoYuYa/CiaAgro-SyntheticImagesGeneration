from utils.seed import set_seed

from pathlib import Path

import argparse

set_seed(42)

def main():

    parser = argparse.ArgumentParser(
        description="Prepare dataset"
    )

    parser.add_argument("--class", choices=["mildio", "oidio", "phakopsora"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-path", type=Path, default=Path("C:/Analise_Esporos_2026/Difusao_ModeloAnterior/Dataset_Splits"))
    args = parser.parse_args()

    base_dataset = r"C:\Analise_Esporos_2026\Dataset_Recortado_4_classes"

    base_splits = r"C:\Analise_Esporos_2026\Difusao_ModeloAnterior\Dataset_Splits"
    os.makedirs(base_splits, exist_ok=True)

    classes = [
        "mildio",
        "oidio",
        "puccinias",
        "phakopsora"
    ]

    nums = [100, 300, 500, 700, 1000, 1300]

    # Garantir que pega apenas imagens
    extensions = (".png", ".jpg", ".jpeg")

    SEED = 42

    for classe in classes:

        dataset_folder = os.path.join(
            base_dataset,
            classe
        )

        split_folder = os.path.join(
            base_splits,
            classe
        )

        os.makedirs(split_folder, exist_ok=True)

        all_images = [
            filename
            for filename in os.listdir(dataset_folder)
            if filename.lower().endswith(extensions)
        ]

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