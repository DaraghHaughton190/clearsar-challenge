"""
Handles the training of models from Ultralytics.
"""

import argparse
import mlflow
import pandas as pd

from pathlib import Path
from utils.seed_everything import seed_everything
from ultralytics import YOLO

def main():
      parser = argparse.ArgumentParser()
      parser.add_argument('-data_config', type=str, required=True,
                          help="Path to dataset.yaml")
      parser.add_argument("-model", type=str, default="yolo11n.pt",
                          help="YOLO architecture to use (defautl: YOLO 11 nano)")
      parser.add_argument("-train_config", type=str, required=True,
                    help="Path to training config YAML")
      parser.add_argument("-project", default="experiments", type=str,
                          help="Project name")
      parser.add_argument("-run_name", required=True, type=str,
                          help="Name of the run")
      parser.add_argument("-seed", type=int, default=96,
                          help="Random seed for reproducibility (default: 96)")
      args = parser.parse_args()

      seed_everything(args.seed)

      # set up mlflow
      mlflow.set_tracking_uri("runs/mlflow")
      mlflow.set_experiment(args.project)
            
      with mlflow.start_run(run_name=args.run_name):
            mlflow.log_params({
                  "model": args.model,
                  "train_config": args.train_config,
                  "data_config": args.data_config,
                  "seed": args.seed
            })
            
            model = YOLO(args.model)
            model.train(
                  cfg=args.train_config,
                  data=Path(args.data_config),
                  project=args.project,
                  name=args.run_name
            )
            
            run_dir = Path("runs/detect") / args.project / args.run_name
            results_csv = run_dir / "results.csv"
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            # find the best epoch by mAP50-95 and log its metrics
            best_row = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
            mlflow.log_metrics({
                  "best_mAP50":    best_row["metrics/mAP50(B)"],
                  "best_mAP50-95": best_row["metrics/mAP50-95(B)"],
                  "best_precision": best_row["metrics/precision(B)"],
                  "best_recall":   best_row["metrics/recall(B)"],
                  "best_epoch":    int(best_row["epoch"]),
            })
      
      # log best checkpoint as artifact
      best_pt = run_dir / "weights" / "best.pt"
      if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
      else:
            print(f"Warning: best.pt not found at {best_pt}")


if __name__ == "__main__":
      main()