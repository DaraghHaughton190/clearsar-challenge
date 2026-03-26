"""
Handles the training of models from Ultralytics.
"""

import argparse
import mlflow
import pandas as pd

from pathlib import Path
from utils.seed_everything import seed_everything
from ultralytics import YOLO


def make_epoch_callback(run_id: str):
      """Return a callback that logs per-epoch metrics to mlflow
      """
      def on_fit_epoch_end(trainer):
            if not trainer.metrics:
                  return
            epoch = trainer.epoch
            metrics = {
                  "train/box_loss":   float(trainer.loss_items[0]) if trainer.loss_items is not None else 0.0,
                  "train/cls_loss":   float(trainer.loss_items[1]) if trainer.loss_items is not None else 0.0,
                  "train/dfl_loss":   float(trainer.loss_items[2]) if trainer.loss_items is not None else 0.0,
                  "val/precision":    float(trainer.metrics.get("metrics/precision(B)", 0.0)),
                  "val/recall":       float(trainer.metrics.get("metrics/recall(B)", 0.0)),
                  "val/mAP50":        float(trainer.metrics.get("metrics/mAP50(B)", 0.0)),
                  "val/mAP50-95":     float(trainer.metrics.get("metrics/mAP50-95(B)", 0.0)),
            }
            with mlflow.start_run(run_id=run_id, nested=True):
                  mlflow.log_metrics(metrics, step=epoch)
            
      return on_fit_epoch_end


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument('-data_config', type=str, required=True,
                          help="Path to dataset.yaml")
      parser.add_argument("-model", type=str, default="yolo11n.pt",
                          help="YOLO architecture to use (defautl: YOLO 11 nano)")
      parser.add_argument("-weights", type=str, default=None,
                          help="Pretrained weights to load (optional)")
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
            
      with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.log_params({
                  "model": args.model,
                  "train_config": args.train_config,
                  "data_config": args.data_config,
                  "seed": args.seed
            })
            
            if args.weights:
                  model = YOLO(args.model).load(args.weights)
            else:
                  model = YOLO(args.model)
                  
            model.add_callback("on_fit_epoch_end",
                       make_epoch_callback(run.info.run_id))
            
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