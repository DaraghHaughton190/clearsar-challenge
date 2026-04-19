"""
Visualizes predictions vs ground truth with a difference panel.
Left: predictions (red boxes)
Middle: ground truth (blue boxes)  
Right: TP pred=green GT=cyan, FP=red, FN=blue
"""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def load_preds(pred_path: str) -> dict:
      with open(pred_path) as f:
            preds = json.load(f)
      result = {}
      for p in preds:
            img_id = p["image_id"]
            if img_id not in result:
                  result[img_id] = []
            result[img_id].append(p["bbox"])
      return result


def load_gt(ann_path: str) -> dict:
      with open(ann_path) as f:
            ann = json.load(f)
      result = {}
      for a in ann["annotations"]:
            img_id = a["image_id"]
            if img_id not in result:
                  result[img_id] = []
            result[img_id].append(a["bbox"])
      return result


def draw_boxes(img: Image.Image, boxes: list, color: str) -> None:
      draw = ImageDraw.Draw(img)
      for x, y, w, h in boxes:
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)


def iou(box1, box2):
      x1, y1, w1, h1 = box1
      x2, y2, w2, h2 = box2
      xi1 = max(x1, x2)
      yi1 = max(y1, y2)
      xi2 = min(x1+w1, x2+w2)
      yi2 = min(y1+h1, y2+h2)
      inter = max(0, xi2-xi1) * max(0, yi2-yi1)
      union = w1*h1 + w2*h2 - inter
      return inter/union if union > 0 else 0


def classify_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
      tp_pairs = []  # (pred_box, gt_box)
      fp_pred = []
      fn_gt = []
      matched_gt = set()
      for pb in pred_boxes:
            matched = False
            for i, gb in enumerate(gt_boxes):
                  if i not in matched_gt and iou(pb, gb) >= iou_threshold:
                        tp_pairs.append((pb, gb))
                        matched_gt.add(i)
                        matched = True
                        break
            if not matched:
                  fp_pred.append(pb)
      for i, gb in enumerate(gt_boxes):
            if i not in matched_gt:
                  fn_gt.append(gb)
      return tp_pairs, fp_pred, fn_gt


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-preds", required=True, help="Predictions JSON")
      parser.add_argument("-images_dir", required=True, help="Val images directory")
      parser.add_argument("-ann_path", required=True, help="Path to instances_train.json")
      parser.add_argument("-output_dir", required=True, help="Output directory")
      parser.add_argument("-iou_threshold", type=float, default=0.5,
                              help="IoU threshold for TP/FP/FN classification (default: 0.5)")
      args = parser.parse_args()

      output_dir = Path(args.output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)

      preds = load_preds(args.preds)
      gt = load_gt(args.ann_path)

      all_ids = set(preds.keys()) | set(gt.keys())

      for img_id in sorted(all_ids):
            img_path = Path(args.images_dir) / f"{img_id}.png"
            if not img_path.exists():
                  continue

            img_pred = Image.open(img_path).convert("RGB")
            img_gt = img_pred.copy()
            img_diff = img_pred.copy()

            pred_boxes = preds.get(img_id, [])
            gt_boxes = gt.get(img_id, [])

            # panel 1: predictions
            draw_boxes(img_pred, pred_boxes, color="red")

            # panel 2: ground truth
            draw_boxes(img_gt, gt_boxes, color="blue")

            # panel 3: difference
            tp_pairs, fp, fn = classify_boxes(pred_boxes, gt_boxes, args.iou_threshold)
            for pred_box, gt_box in tp_pairs:
                  draw_boxes(img_diff, [pred_box], color="green")
                  draw_boxes(img_diff, [gt_box], color="cyan")
            draw_boxes(img_diff, fp, color="red")
            draw_boxes(img_diff, fn, color="blue")

            ImageDraw.Draw(img_pred).text((5, 5), "Predictions", fill="red")
            ImageDraw.Draw(img_gt).text((5, 5), "Ground Truth", fill="blue")
            ImageDraw.Draw(img_diff).text((5, 5), "TP pred=green GT=cyan FP=red FN=blue", 
                                          fill="white")

            w, h = img_pred.size
            combined = Image.new("RGB", (w * 3, h))
            combined.paste(img_pred, (0, 0))
            combined.paste(img_gt, (w, 0))
            combined.paste(img_diff, (w*2, 0))
            combined.save(output_dir / f"{img_id}.png")

      print(f"Saved {len(all_ids)} images to {output_dir}")


if __name__ == "__main__":
      main()