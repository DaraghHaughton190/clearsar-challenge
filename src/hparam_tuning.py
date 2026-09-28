import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO, RTDETR
from utils.seed_everything import seed_everything
from optuna.integration.mlflow import MLflowCallback

def model_loader(model_name: str) -> nn.Module:
    # NOTE: THESE MODELS ARE PRETRAINED ON THE COCO DATASET!
    # Good for RT-DETR
    if model_name == "RT-DETR":
        seed_everything(seed=42, deterministic=False)
        model = RTDETR("rtdetr-l.pt")

        return model

    # Not sure how that will effect the YOLO model
    if model_name == "YOLO":
        seed_everything(seed=42, deterministic=True)
        model = YOLO("yolo11n.pt")
        return model

    else:
        raise ValueError("Unsupported model type")
    
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    model_choice = trial.suggest_categorical("model_name", ["RT-DETR", "YOLO"])

    box_weight = trial.suggest_float("box", 1, 10, log = True) #bboc loss
    cls_weight = trial.suggest_float("cls", 0.2, 4, log = True) # classification loss
    dfl_weight = trial.suggest_float("dfl", 0.2, 4, log = True) # dist-focal loss

    if model_choice  == "RT-DETR":
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log = True)

    else:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log = True)

    model = model_loader(model_choice)

    def pruner(trainer):
        current_map = trainer.fitness
        epoch = trainer.epoch

        trial.report(current_map, step = epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    model.add_callback("pruner", pruner)

    model.train(data = "configs/dataset.yaml",
                lr0 = lr,
                imgsz = 512,
                rect = True,
                mosaic = 0.0,
                batch = batch_size,
                box = box_weight,
                cls = cls_weight,
                dfl = dfl_weight,
                optimizer = "AdamW", 
                epochs = 20, 
                seed=42)

    metrics = model.val()
    val_mAP = metrics.box.map

    trial.report(val_mAP, step= 2)

    return val_mAP

if __name__ == "__main__":

    mlflc = MLflowCallback(
        tracking_uri="file:./mlruns",
        metric_name="val_mAP",
        create_experiment=True
    )


    study = optuna.create_study(study_name = "SAR_Architecture_Sweep_V2",
                                direction = "maximize",
                                 pruner = optuna.pruners.MedianPruner(
                                    n_startup_trials = 3,
                                    n_warmup_steps = 5,
                                    interval_steps =1
                                 )
                                 )

    study.optimize(objective, n_trials = 5, callbacks=[mlflc])

    print("Best trial:", study.best_trial.value)
    print("Best params:", study.best_trial.params)

    # view ML flow dash: mlflow ui --host 0.0.0.0 --port 5000