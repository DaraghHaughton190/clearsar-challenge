"""
Runs Weighted Boxes Fusion (WBF) to generate a model ensemble.
Can run it by passing the predictions from different folds of the same model,
or combining predictions from different architectures (e.g. yolo11s and rt-detr)
"""

import json
import argparse
import numpy as np

from pathlib import Path
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion


def load_predictions(pred_files: list) -> dict:
      """Loads predictions from multiple JSON files and groups
      them by image_id. 
      Returns a dict: {image_id: [list of detections per model]}.
      Each element of this list corresponds to one model's detection for
      that image.
      """
      all_preds = {}
      for pred_file in pred_files:
            with open(pred_file) as f:
                  preds = json.load(f)
            for det in preds:
                  img_id = det["image_id"]
                  if img_id not in all_preds:
                        all_preds[img_id] = [[] for _ in pred_files]
                  all_preds[img_id][pred_files.index(pred_file)].append(det)
      
      return all_preds


def get_image_dimensions(images_dir: Path) -> dict:
      """Read width and height for every image in the direcotry.
      """
      dimensions = {}
      for img_path in images_dir.iterdir():
            if img_path.suffix == '.png':
                  img_id = int(img_path.stem)
                  with Image.open(img_path) as img:
                        # get (width, height)
                        dimensions[img_id] = img.size
      
      return dimensions


def run_wbf(model_preds: list, width: int, height:int,
            iou_threshold: float, skip_box_threshold: float) -> list:
      """Run WBF for a single image across all models.
      """
      boxes_list = []
      scores_list = []
      labels_list = []
      
      for model_dets in model_preds:
            boxes = []
            scores = []
            labels = []
            for det in model_dets:
                  x, y, w, h = det["bbox"]
                  
                  # normalize to [0, 1] as required by WBF
                  x1 = x / width
                  y1 = y / height
                  x2 = (x + w) / width
                  y2 = (y + h) / height
                  
                  # clamp to [0, 1] to avoid out-of-bounds boxes
                  boxes.append([
                        max(0, min(1, x1)),
                        max(0, min(1, y1)),
                        max(0, min(1, x2)),
                        max(0, min(1, y2))
                  ])
                  scores.append(det["score"])
                  labels.append(1)
            
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
      
      if not any(boxes_list):
            return []
      
      boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_threshold
      )
      
      # convert back to COCO format
      detections = []
      for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            detections.append({
                  "bbox": [
                        x1 * width,
                        y1 * height,
                        (x2 - x1) * width,
                        (y2 - y1) * height
                  ],
                  "score": float(score)
            })
      
      return detections


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-pred_files", nargs="+", required=True,
                          help="List of prediction JSON files")
      parser.add_argument("-images_dir", required=True,
                          help="Path to test images directory")
      parser.add_argument("-output", required=True,
                          help="Path for output submission JSON")
      parser.add_argument("-iou_threshold", type=float, default=0.55,
                          help="WBF IoU threshold (default: 0.55)")
      parser.add_argument("-skip_box_threshold", type=float, default=0.0001,
                          help="WBF skip box threshold (default: 0.0001)")
      args = parser.parse_args()
      
      images_dir = Path(args.images_dir)
      output_path = Path(args.output)
      output_path.parent.mkdir(parents=True, exist_ok=True)
      
      print("Loading predictions...")
      all_preds = load_predictions(args.pred_files)
      
      print("Reading image dimensions...")
      dimensions = get_image_dimensions(images_dir)
      
      print("Running WBF...")
      final_detections = []
      for img_id, model_preds in all_preds.items():
            width, height = dimensions[img_id]
            merged = run_wbf(model_preds, width, height,
                             args.iou_threshold, args.skip_box_threshold)
            
            for det in merged:
                  final_detections.append({
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": det["bbox"],
                        "score": det["score"]
                  })

      with open(output_path, "w") as f:
            json.dump(final_detections, f)
            
      print(f"Saved {len(final_detections)} detections to {output_path}")
      

if __name__ == "__main__":
      main()