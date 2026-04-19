"""
Runs inference on the test set using a trained Ultralytics model checkpoint
and outputs a valid submission.json in COCO detection format (as expected for
the challenge.)
"""

import json
import argparse

from pathlib import Path
from ultralytics import YOLO


def run_predict(checkpoint: str, images_dir: str, 
                output_path: str, conf: float = 0.25,
                imgsz: int = 640) -> list:
      model = YOLO(str(checkpoint))
      results = model.predict(
            source=str(images_dir),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
      )
      
      detections = []
      for result in results:
            image_id = int(Path(result.path).stem)
            if result.boxes is None or len(result.boxes) == 0:
                  continue
            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            for box, score in zip(boxes_xyxy, scores):
                  x_min, y_min, x_max, y_max = box
                  w = x_max - x_min
                  h = y_max - y_min
                  detections.append({
                  "image_id": image_id,
                  "category_id": 1,
                  "bbox": [x_min, y_min, w, h],
                  "score": score
                  })
                  
      output_path = Path(output_path)
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, 'w') as f:
            json.dump(detections, f)
            
      return detections


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-checkpoint", required=True,
                          help="Path to model checkpoint (.pt file)")
      parser.add_argument("-images-dir", required=True,
                          help="Path to test images folder")
      parser.add_argument("-output", required=True,
                          help="Path for output JSON")
      parser.add_argument("-conf", type=float, default=0.25,
                          help="Confidence threshold for detections (default: 0.25)")
      parser.add_argument("-imgsz", type=int, default=640,
                    help="Inference image size (default: 640)")
      args = parser.parse_args()
      
      detections = run_predict(checkpoint=args.checkpoint,
                               images_dir=args.images_dir,
                               output_path=args.output,
                               conf=args.conf,
                               imgsz=args.imgsz)
      
      print(f"Saved {len(detections)} detections to {args.output}")
      
      return detections


if __name__ == "__main__":
      main()