"""
Handles the conversion of our annotations from COCO JSON to the format Ultralytics
can consume (one txt file per image, nornalized coords)
"""

import json
import argparse

from pathlib import Path
from collections import defaultdict


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-annotations-path", required=True,
                          help="Path the instances_train.json file")
      parser.add_argument("-data-output-dir", required=True,
                          help="Directory where to created the txt files for YOLO")
      args = parser.parse_args()
      
      # extract paths
      ann_path = Path(args.annotations_path)
      labels_out_dir = Path(args.data_output_dir)
      labels_out_dir.mkdir(parents=True, exist_ok=True)
      
      # load json file and build a lookup dict
      with open(ann_path) as f:
            annotations = json.load(f)
            
      images_lookup = {
            img["id"]: (Path(img['file_name']).stem, img['width'], img['height'])
                        for img in annotations['images']
      }
      
      anns_by_image = defaultdict(list)
      for ann in annotations['annotations']:
            anns_by_image[ann['image_id']].append(ann)
            

      # write labels
      for i in images_lookup:
            stem, img_w, img_h = images_lookup[i]
            # default dict handles empty list for negative images
            img_anns = anns_by_image[i]
            
            with open(labels_out_dir / f"{stem}.txt", "w") as out_f:
                  for ann in img_anns:
                        x, y, w, h = ann["bbox"]
                        cx = (x + w / 2) / img_w
                        cy = (y + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h
                  
                        out_f.write(f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")
      
      # summary
      print(f"Number of label files written: {len(list(labels_out_dir.iterdir()))}")
      print(f"Number of empty (negative) files:\
            {sum(1 for f in labels_out_dir.iterdir() if f.stat().st_size == 0)}")
            

if __name__ == "__main__":
      main()