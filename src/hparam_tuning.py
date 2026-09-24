import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
from utils.seed_everything import seed_everything


def get_data_loaders(batch_size):
    return [None], [None]

def model_loader(model_name: str, num_classes: int) -> nn.Module:
    seed_everything(seed=42, deterministic=True)
    #if model_name == "faster_rcnn":
    #    model = fasterrcnn_resnet50_fpn(pretrained=True)
    #    return model
    # for eventaully integrating FRCNN:
    # https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    if model_name == "YOLO":
        model = YOLO("yolov8n.pt")
        return model

    else:
        raise ValueError("Unsupported model type")
    
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    model_choice = trial.suggest_categorical("model_name", ["YOLO"])

    #train_loader, val_loader = get_data_loaders(batch_size = batch_size)
    model = model_loader(model_choice, num_classes= 2)

    #optimiser = optim.AdamW(model.parameters(), lr=lr)

    #yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01

    model.train(data = "/home/daragh/clearsar-challenge/configs/dataset.yaml",
                 lr0 = lr, batch = batch_size, optimizer = "AdamW", epochs = 2, seed=42, 
                 deterministic = True)

    #for epoch in range(epochs):
    #    model.train()

    #    for images, targets in train_loader:
    #        optimiser.zero_grad()
    #        loss = model(images, targets)
    #        loss.backward()
    #        optimiser.step()

    metrics = model.val()
    val_mAP = metrics.box.map

    trial.report(val_mAP)

    #if trial.should_prune():
    #    raise optuna.TrialPruned()

    return val_mAP

if __name__ == "__main__":
    study = optuna.create_study(direction = "maximize",
                                 #pruner = optuna.pruner.MedianPruner()
                                 )

    study.optimize(objective, n_trials = 20)

    print("Best trial:", study.best_trial.value)
    print("Best params:", study.best_trial.params)