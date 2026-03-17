"""
Runs inference on the test set using a trained Ultralytics model checkpoint
and outputs a valid submission.json in COCO detection format (as expected for
the challenge.)
"""

import json
import argparse

from pathlib import Path
from ultralytics import YOLO


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
      args = parser.parse_args()
      
      checkpoint = Path(args.checkpoint)
      images_dir = Path(args.images_dir)
      output_path = Path(args.output)
      output_path.parent.mkdir(parents=True, exist_ok=True)
      
      # load model and run inference
      model = YOLO(str(checkpoint))
      
      results = model.predict(
            source=str(images_dir),
            conf=args.conf,
            verbose=False,
      )
      
      # convert inference results to COCO format
      detections = []
      
      for result in results:
            # extract the filename without file extenstion
            image_id = int(Path(result.path).stem)
            
            # handle negative cases with 0 RFI predicted (skip them)
            if result.boxes is None or len(result.boxes) == 0:
                  continue
            
            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            scores  = result.boxes.conf.cpu().tolist()
            
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
      
      # save and summarise
      with open(output_path, 'w') as f:
            json.dump(detections, f)
      
      print(f"Saved {len(detections)} detections for "
            f"{len(results)} images to {output_path}")
      print(f"Confidence threshold used: {args.conf}")
      print(f"Images with 0 detections: "
            f"{sum(1 for r in results if r.boxes is None or len(r.boxes) == 0)}")
      

if __name__ == "__main__":
      main()