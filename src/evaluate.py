"""
Wraps pycocotools COCOeval to evaluate model predictions against ground truth annotations.
Returns a full mAP breakdown: mAP50:95, mAP50, mAP75 and mAP for small/medium/large
objects separately.
"""

import json
import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(ann_path: str, pred_path: str, img_ids: list = None) -> dict:
      coco_gt = COCO(ann_path)
      
      with open(pred_path) as f:
            preds = json.load(f)
            
      # return 0s if no predictions
      if len(preds) == 0:
            print("Warning: predictions file is empty")
            return {k: 0.0 for k in
                    ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large"]}
      
      coco_dt = coco_gt.loadRes(preds)
      
      evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
      
      # restrict evaluation to specific image IDs (if provided)
      # otherwise use all images present in the predictions file
      if img_ids is not None:
            evaluator.params.imgIds = img_ids
      else:
            evaluator.params.imgIds = list({p["image_id"] for p in preds})
      
      evaluator.evaluate()
      evaluator.accumulate()
      evaluator.summarize()
      
      return {
            "mAP":        float(evaluator.stats[0]),
            "mAP50":      float(evaluator.stats[1]),
            "mAP75":      float(evaluator.stats[2]),
            "mAP_small":  float(evaluator.stats[3]),
            "mAP_medium": float(evaluator.stats[4]),
            "mAP_large":  float(evaluator.stats[5]),
      }
      

def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-ann-path", required=True,
                          help="Path to instances_train.json")
      parser.add_argument("-pred-path", required=True,
                          help="Path to prdictions JSON (COCO format)")
      parser.add_argument("-img-ids", nargs="+", type=int, default=None,
                          help="Optional list of image IDs to run evaluation on")
      args = parser.parse_args()
      
      metrics = evaluate(args.ann_path, args.pred_path, args.img_ids)
      
      print("\nEVALUATION RESULTS")
      for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
      main()