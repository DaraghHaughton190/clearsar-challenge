"""
Splits the training set into k stratified folds and saves the result to a JSON file.
Stratification is based on RFI count per image, binned into 4 groups.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


def bin_count(n):
      if n == 0:
            return 0
      elif n <= 3:
            return 1
      elif n <= 10:
            return 2
      else:
            return 3


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-annotations-path", required=True,
                              help="Path to instances_train.json")
      parser.add_argument("-output-path", required=True,
                              help="Path to save folds.json")
      parser.add_argument("-n-splits", type=int, default=5,
                              help="Number of folds (default: 5)")
      args = parser.parse_args()

      ann_path = Path(args.annotations_path)
      output_path = Path(args.output_path)
      output_path.parent.mkdir(parents=True, exist_ok=True)

      with open(ann_path) as f:
            data = json.load(f)
            
      # count annotations per image
      ann_counts = defaultdict(int)
      for ann in data["annotations"]:
            ann_counts[ann["image_id"]] += 1

      # build ordered lists 
      # StratifiedKFold needs arrays, order must be consistent
      image_ids = [img["id"] for img in data["images"]]
      counts = [ann_counts[img_id] for img_id in image_ids]
      
      #  bin counts for stratification
      bins = [bin_count(c) for c in counts]
      
      # run the split process and save output
      skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
      folds = {}
      for fold_idx, (train_indices, val_indices) in enumerate(skf.split(image_ids, bins)):
            folds[fold_idx] = {
                  "train": [image_ids[i] for i in train_indices],
                  "val": [image_ids[i] for i in val_indices]
            }

      with open(output_path, "w") as f:
            json.dump(folds, f, indent=2)
            
      # summary
      print(f"Saved {args.n_splits} folds to {output_path}")
      for fold_idx, fold in folds.items():
            val_bins = [bin_count(ann_counts[img_id]) for img_id in fold["val"]]
            print(f"  Fold {fold_idx}: {len(fold['train'])} train, {len(fold['val'])} val "
                  f"| val bins -> 0:{val_bins.count(0)} 1-3:{val_bins.count(1)} "
                  f"4-10:{val_bins.count(2)} 10+:{val_bins.count(3)}")


if __name__ == "__main__":
      main()