import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO, RTDETR
from utils.seed_everything import seed_everything


def get_data_loaders(batch_size):
    return [None], [None]

def model_loader(model_name: str) -> nn.Module:
    #if model_name == "faster_rcnn":
    #    model = fasterrcnn_resnet50_fpn(pretrained=True)
    #    return model
    # for eventaully integrating FRCNN:
    # https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    # NOTE: THESE MODELS ARE PRETRAINED ON THE COCO DATASET!!!
    if model_name == "RT-DETR":
        seed_everything(seed=42, deterministic=False)
        model = RTDETR("rtdetr-l.pt")

        return model

    if model_name == "YOLO":
        seed_everything(seed=42, deterministic=True)
        model = YOLO("yolov8n.pt")
        return model

    else:
        raise ValueError("Unsupported model type")
    
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    model_choice = trial.suggest_categorical("model_name", ["RT-DETR", "YOLO"])

    box_weight = trial.suggest_float("box", 1, 10, log = True) #bboc loss
    cls_weight = trial.suggest_float("cls", 0.2, 4, log = True) # classification loss
    dfl_weight = trial.suggest_float("dlf", 0.2, 4, log = True) # dist-focal loss

    if model_choice  == "RT-DETR":
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log = True)

    else:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log = True)

    model = model_loader(model_choice)

    model.train(data = "configs/dataset.yaml",
                lr0 = lr,
                batch = batch_size,
                box = box_weight,
                cls = cls_weight,
                dfl = dfl_weight,
                optimizer = "AdamW", 
                epochs = 2, 
                seed=42)

    metrics = model.val()
    val_mAP = metrics.box.map

    trial.report(val_mAP)

    return val_mAP

if __name__ == "__main__":
    study = optuna.create_study(direction = "maximize",
                                 #pruner = optuna.pruner.MedianPruner()
                                 )

    study.optimize(objective, n_trials = 20)

    print("Best trial:", study.best_trial.value)
    print("Best params:", study.best_trial.params)