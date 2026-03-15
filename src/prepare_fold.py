"""
Generates a specific fold from folds.json into val images/labels folders
using symlinks (to not duplicate files)
"""

import json
import argparse
from pathlib import Path


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-folds-path", required=True,
                          help="Path to folds.json")
      parser.add_argument("-images-train-dir", required=True,
                          help="Path to images/train folder")
      parser.add_argument("-labels-train-dir", required=True,
                          help="Path to labels/train folder")
      parser.add_argument("-images-val-dir", required=True,
                          help="Path to labels/val folder (will be created)")
      parser.add_argument("-labels-val-dir", required=True,
                          help="Path to labels/val folder (will be created)")
      parser.add_argument("-fold", type=int, default=0,
                          help="Fold index to generate (defaultO 0)")
      args = parser.parse_args()
      
      # use resolve() as we need to convert every path to absolute paths
      # otherwise, seamlinks will break
      images_train_dir = Path(args.images_train_dir).resolve()
      labels_train_dir = Path(args.labels_train_dir).resolve()
      images_val_dir = Path(args.images_val_dir).resolve()
      labels_val_dir = Path(args.labels_val_dir).resolve()
      
      images_val_dir.mkdir(parents=True, exist_ok=True)
      labels_val_dir.mkdir(parents=True, exist_ok=True)
      
      with open(args.folds_path) as f:
            folds = json.load(f)
            
      val_ids = folds[str(args.fold)]["val"]
      
      # create symlinks
      img_count = 0
      lbl_count = 0
      for img_id in val_ids:
            # image symlink
            src_img = images_train_dir / f"{img_id}.png"
            dst_img = images_val_dir / f"{img_id}.png"
            if not dst_img.exists():
                  dst_img.symlink_to(src_img)
                  img_count += 1
            
            # label symlink
            src_lbl = labels_train_dir / f"{img_id}.txt"
            dst_lbl = labels_val_dir / f"{img_id}.txt"
            if not dst_lbl.exists():
                  dst_lbl.symlink_to(src_lbl)
                  lbl_count += 1
      
      # summary
      print(f"Fold {args.fold} created:")
      print(f"    Image symlinks created: {img_count}")
      print(f"    Label symlinks created: {lbl_count}")
      print(f"    Val images dir: {images_val_dir}")
      print(f"    Val labels dir: {labels_val_dir}")
      

if __name__ == "__main__":
      main()