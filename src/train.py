"""
Handles the training of models from Ultralytics.
"""

import argparse

from pathlib import Path
from ultralytics import YOLO


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument('-data_config', type=str, required=True,
                          help="Path to dataset.yaml")
      parser.add_argument("-model", type=str, default="yolo11n.pt",
                          help="YOLO architecture to use (defautl: YOLO 11 nano)")
      parser.add_argument("-epochs", required=True, type=int,
                          help="Number of training epochs")
      parser.add_argument('-imgsz', type=int, default=640,
                          help="Resolution that Ultralytics will use to resize the images")
      parser.add_argument("-batch", type=int, default=16,
                          help="Batch size (default: 16)")
      parser.add_argument("-project", default="experiments", type=str,
                          help="Project name")
      parser.add_argument("-run_name", required=True, type=str,
                          help="Name of the run")
      args = parser.parse_args()

      model = YOLO(args.model)

      model.train(
            data=Path(args.data_config),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.run_name
      )


if __name__ == "__main__":
      main()