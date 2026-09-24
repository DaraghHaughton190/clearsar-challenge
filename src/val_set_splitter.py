import os
from pathlib import Path
import shutil
import random
from utils.seed_everything import seed_everything

data_dir = Path("/home/daragh/clearsar-challenge/ClearSAR_Dataset/data")
images_dir = data_dir / "images/train"
labels_dir = data_dir / "labels/train"

new_train_images = data_dir / "images/local_train"
new_val_images = data_dir / "images/local_val"
new_train_labels = data_dir / "labels/local_train"
new_val_labels = data_dir / "labels/local_val"

def validation_splitter(split_ratio = 0.9):
    """
    New function to produce a very small validation set for use with optuna
    """

    print("Splitting dataset for hparam optimisation")
    seed_everything(seed = 42, deterministic = True)


    for p in [new_train_images, new_val_images, new_train_labels, new_val_labels]:
        p.mkdir(parents=True, exist_ok=True)

    all_images = [f for f in images_dir.glob("*.png") if (labels_dir / f"{f.stem}.txt").exists()]

    split_index = int(len(all_images) * split_ratio)
    train_files = all_images[:split_index]
    val_files = all_images[split_index:]

    print(f"Split successful: {len(train_files)} training | {len(val_files)} validation")

    for img_path in train_files:
        shutil.copy(img_path, new_train_images / img_path.name)
        shutil.copy(labels_dir / f"{img_path.stem}.txt", new_train_labels / f"{img_path.stem}.txt")

    for img_path in val_files:
        shutil.copy(img_path, new_val_images / img_path.name)
        shutil.copy(labels_dir / f"{img_path.stem}.txt", new_val_labels / f"{img_path.stem}.txt")

if __name__ == "__main__":
    validation_splitter()