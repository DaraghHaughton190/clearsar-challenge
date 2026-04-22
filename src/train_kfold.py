"""
Trains a model on all k folds sequentially and evaluates each fold
with pycocotools. Logs per-fold metrics as nested MLflow runs under
a parent run, and logs the average mAP at the end.
"""

import json
import argparse
import mlflow
import torch
import gc

from pathlib import Path
from ultralytics import YOLO

from utils.seed_everything import seed_everything
from predict import run_predict
from evaluate import evaluate


def clear_val_dirs(images_val_dir: Path, labels_val_dir: Path):
      """Remove all symlinks from val directories.
      """
      for d in [images_val_dir, labels_val_dir]:
            if d.exists():
                  for f in d.iterdir():
                        if f.is_symlink():
                              f.unlink()


def prepare_fold(folds: dict, fold: int,
                 images_train_dir: Path, labels_train_dir: Path,
                 images_val_dir: Path, labels_val_dir: Path):
      """Create symlinks for the given fold
      """
      images_val_dir.mkdir(parents=True, exist_ok=True)
      labels_val_dir.mkdir(parents=True, exist_ok=True)

      val_ids = folds[str(fold)]["val"]
      for img_id in val_ids:
            src_img = images_train_dir / f"{img_id}.png"
            dst_img = images_val_dir / f"{img_id}.png"
            if not dst_img.exists():
                  dst_img.symlink_to(src_img)

            src_lbl = labels_train_dir / f"{img_id}.txt"
            dst_lbl = labels_val_dir / f"{img_id}.txt"
            if not dst_lbl.exists():
                  dst_lbl.symlink_to(src_lbl)


def write_fold_train_txt(folds: dict, fold: int,
                         images_train_dir: Path, output_dir: Path) -> Path:
      """Write a txt file disting paths of train split images
      for this fold.
      """
      output_dir.mkdir(parents=True, exist_ok=True)
      txt_path = output_dir / f"fold{fold}_train.txt"
      train_ids = folds[str(fold)]["train"]
      
      with open(txt_path, "w") as f:
            for img_id in train_ids:
                  f.write(str(images_train_dir / f"{img_id}.png") + "\n")
      
      return txt_path
 

def write_fold_dataset_yaml(data_config: str, fold: int, train_txt_path: Path) -> str:
      """Write a dataset yaml for the given fold with train pointing to the fold txt file.
      """
      with open(data_config) as f:
            content = f.read()
      
      content = content.replace(
            [l for l in content.splitlines() if l.startswith("train:")][0],
            f"train: {train_txt_path.resolve()}"
      )
      fold_yaml_path = Path(data_config).parent / f"dataset_fold{fold}.yaml"
      
      with open(fold_yaml_path, "w") as f:
            f.write(content)
            
      return str(fold_yaml_path)

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


def patch_albumentations():
      """Method to monkey-patch parameters for custom transforms
      """
      from ultralytics.data.augment import Albumentations
      
      def custom_init(self, p=1.0, **kwargs):
            self.p = p
            self.transform = None
            try:
                  import albumentations as A
                  T = [
                        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                                   contrast_limit=0.2, 
                                                   p=0.3),
                  ]
                  self.transform = A.Compose(T)
                  self.contains_spatial = False
            except Exception as e:
                  print(f"Albumentations patch failed: {e}")
      
      Albumentations.__init__ = custom_init


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("-model", type=str, default="yolo11s.pt",
                              help="Model to train")
      parser.add_argument("-weights", type=str, default=None,
                          help="Pretrained weights to load (optional)")
      parser.add_argument("-data_config", type=str, required=True,
                              help="Path to base dataset.yaml")
      parser.add_argument("-train_config", type=str, required=True,
                              help="Path to training config YAML")
      parser.add_argument("-folds_path", type=str, required=True,
                              help="Path to folds.json")
      parser.add_argument("-images_train_dir", type=str, required=True,
                              help="Path to images/train")
      parser.add_argument("-labels_train_dir", type=str, required=True,
                              help="Path to labels/train")
      parser.add_argument("-images_val_dir", type=str, required=True,
                              help="Path to images/val")
      parser.add_argument("-labels_val_dir", type=str, required=True,
                              help="Path to labels/val")
      parser.add_argument("-ann_path", type=str, required=True,
                              help="Path to instances_train.json")
      parser.add_argument("-project", type=str, default="experiments",
                              help="Project name")
      parser.add_argument("-run_name", type=str, required=True,
                              help="Parent run name")
      parser.add_argument("-seed", type=int, default=96,
                              help="Random seed")
      parser.add_argument("--no-deterministic", dest="deterministic", 
                    action="store_false",
                    help="Disable cudnn deterministic mode (required for RT-DETR)")
      parser.add_argument("-imgsz", type=int, default=640,
                    help="Inference image size for validation (default: 640)")
      parser.add_argument("-num_folds", type=int, default=None,
                    help="Number of folds to run (default: all folds)")
      parser.add_argument("-start_fold", type=int, default=0,
                    help="Fold to start from (default: 0)")
      parser.set_defaults(deterministic=True)
      args = parser.parse_args()

      seed_everything(args.seed, deterministic=args.deterministic)

      images_train_dir = Path(args.images_train_dir).resolve()
      labels_train_dir = Path(args.labels_train_dir).resolve()
      images_val_dir = Path(args.images_val_dir).resolve()
      labels_val_dir = Path(args.labels_val_dir).resolve()

      with open(args.folds_path) as f:
            folds = json.load(f)

      mlflow.set_tracking_uri("runs/mlflow")
      mlflow.set_experiment(args.project)

      fold_maps = []

      with mlflow.start_run(run_name=args.run_name) as parent_run:
            mlflow.log_params({
                  "model": args.model,
                  "train_config": args.train_config,
                  "data_config": args.data_config,
                  "seed": args.seed,
                  "n_folds": len(folds),
                  "imgsz": args.imgsz
            })
            
            n = args.start_fold + (args.num_folds if args.num_folds is not None else len(folds))
            for fold in range(args.start_fold, n):
                  fold_run_name = f"{args.run_name}_fold{fold}"
                  print(f"\n{'='*50}")
                  print(f"Starting fold {fold}")
                  print(f"{'='*50}\n")

                  # prepare val symlinks for this fold
                  clear_val_dirs(images_val_dir, labels_val_dir)
                  prepare_fold(folds, fold,
                              images_train_dir, labels_train_dir,
                              images_val_dir, labels_val_dir)
                  
                  train_txt = write_fold_train_txt(folds, fold, images_train_dir,
                                  Path("data/splits"))

                  # write fold-specific dataset yaml
                  fold_yaml = write_fold_dataset_yaml(args.data_config, fold, train_txt)

                  # train
                  if args.weights:
                        model = YOLO(args.model).load(args.weights)
                  else:
                        model = YOLO(args.model)
                  
                  with mlflow.start_run(run_name=fold_run_name, nested=True) as fold_run:
                        model.add_callback("on_fit_epoch_end", 
                                           make_epoch_callback(fold_run.info.run_id))
                        mlflow.log_params({
                              "fold": fold,
                              "model": args.model,
                              "train_config": args.train_config,
                        })

                        # call the monkey-patch method for data aug
                        patch_albumentations()
                        
                        model.train(
                              cfg=args.train_config,
                              data=fold_yaml,
                              project=args.project,
                              name=fold_run_name,
                              exist_ok=True
                        )

                        # predict on val set
                        best_pt = Path("runs/detect") / args.project / fold_run_name / "weights" / "best.pt"
                        pred_path = f"validation/{fold_run_name}.json"
                        run_predict(str(best_pt), str(images_val_dir),
                                    pred_path, conf=0.25, imgsz=args.imgsz)

                        # evaluate with pycocotools
                        val_ids = folds[str(fold)]["val"]
                        metrics = evaluate(args.ann_path, pred_path, img_ids=val_ids)
                        fold_maps.append(metrics["mAP"])

                        mlflow.log_metrics({
                              "mAP":        metrics["mAP"],
                              "mAP50":      metrics["mAP50"],
                              "mAP75":      metrics["mAP75"],
                              "mAP_small":  metrics["mAP_small"],
                              "mAP_medium": metrics["mAP_medium"],
                              "mAP_large":  metrics["mAP_large"],
                        })

                        print(f"Fold {fold} pycocotools mAP: {metrics['mAP']:.4f}")
                        
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()

            # log average across folds to parent run
            avg_map = sum(fold_maps) / len(fold_maps)
            mlflow.log_metric("avg_mAP", avg_map)
            print(f"\nK-fold complete. Average mAP: {avg_map:.4f}")
            print(f"Per-fold mAPs: {[round(m, 4) for m in fold_maps]}")


if __name__ == "__main__":
      main()