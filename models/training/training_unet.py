print("Importing misc libraries")
import sys
import os
import torch
#import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

print("Updating sys.path")
project_root = Path.cwd().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
models_path = Path.cwd().parent
if str(models_path) not in sys.path:
    sys.path.append(str(models_path))
    
print("Importing torch libraries")
from torch.utils.data import DataLoader
import multiprocessing
from torchvision.transforms import ToTensor
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
    
print("Importing unet lib")
import importlib
import unet.Unet as u
importlib.reload(u)

print("Importing dataset lib")
from dataset.IntersectionDataset import IntersectionDataset, IntersectionDataset2, IntersectionDatasetClasses, custom_collate_fn
import loss.loss_lib as ll
importlib.reload(ll)

dataset_dir = "../../dataset/dataset/train"
img_transform = ToTensor()
path_transform = ToTensor()
dataset_train = IntersectionDatasetClasses(root_dir=dataset_dir, 
                                    transform=img_transform,
                                    path_transform=path_transform)

dataset_dir = "../../dataset/dataset/test"
img_transform = ToTensor()
path_transform = ToTensor()
dataset_test = IntersectionDatasetClasses(root_dir=dataset_dir,
                                   transform=img_transform,
                                   path_transform=path_transform)
print(len(dataset_train))
print(len(dataset_test))

num_workers = multiprocessing.cpu_count()
b = 4

train_dataloader = DataLoader(dataset_train, batch_size=b, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(dataset_test, batch_size=b, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_fn)

#os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
def init_weights(m):
    if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
        nn.init.normal_(m.weight, mean=0, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        pass
    
try: 
    del(model)
except NameError:
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = u.UNet(n_channels=3, n_classes=5).to(device) # background, left, right, ahead, stacked
model.apply(init_weights)

try:
    del(optimizer)
except NameError:
    pass
optimizer = Adam(model.parameters(), lr=1e-4)

try:
    del(scheduler)
except NameError:
    pass
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)

try:
    del(scaler)
except NameError:
    pass
scaler = GradScaler(device=device)

class_counts = torch.tensor([152000, 2000, 2000, 2000, 2000], dtype=torch.float)
weights = 1.0 / class_counts
weights = weights / weights.sum()
lb = torch.nn.CrossEntropyLoss(weight=weights.to(device))
lb = lb.to(device)

from tqdm import tqdm

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

n_epochs = 500
alpha = 0.5
epochs = tqdm(range(n_epochs), desc="Training", unit=" epoch")


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

for epoch in epochs:
    model.train()
    running_train_loss = 0.0
    running_train_correct = 0
    running_train_total = 0
    
    batches = tqdm(train_dataloader, desc="Batches", unit=" batch", leave=False)
    
    for batch in batches:
        satellite = batch["satellite"].to(device, non_blocking=True)
        class_labels = batch["class_labels"].to(device, non_blocking=True)
        #path_line = batch["paths"]
        
        class_labels = class_labels.squeeze(1)
        
        optimizer.zero_grad()
        
        with autocast("cuda"):
            output = model(satellite)
            loss = lb(output, class_labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_train_loss += loss.item()
        
        p = torch.argmax(output, dim=1)
        running_train_correct += (p == class_labels).sum().item()
        running_train_total += class_labels.size(0)
        
        batches.set_postfix({"Loss": loss.item()})
        
        
    avg_train_loss = running_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_accuracy = running_train_correct / running_train_total
    train_accuracies.append(train_accuracy)
    
    model.eval()
    running_test_loss = 0.0
    running_test_correct = 0
    running_test_total = 0
    
    test_batches = tqdm(test_dataloader, desc="Batches", unit=" batch", leave=False)
    with torch.no_grad():
        for batch in test_batches:
            satellite = batch["satellite"].to(device, non_blocking=True)
            class_labels = batch["class_labels"].to(device, non_blocking=True)
            #path_line = batch["paths"]
            
            class_labels = class_labels.squeeze(1)
            
            with autocast("cuda"):
                output = model(satellite)
                loss = lb(output, class_labels)
            
            running_test_loss += loss.item()
            
            p = torch.argmax(output, dim=1)
            running_test_correct += (p == class_labels).sum().item()
            running_test_total += class_labels.size(0)
            
            test_batches.set_postfix({"Loss": loss.item()})
            
    test_batches.close()
            
    avg_test_loss = running_test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    
    test_accuracy = running_test_correct / running_test_total
    test_accuracies.append(test_accuracy)
    
    scheduler.step()  
    
    if (epoch + 1) % 50 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
        }
        os.makedirs('./ckpt', exist_ok=True)
        torch.save(checkpoint, f'./ckpt/bce_checkpoint_epoch_{epoch + 1}_5classes.pth')
        
    epochs.set_postfix({"Train Loss": avg_train_loss, "Test Loss": avg_test_loss, "Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy})
    batches.close()
    
epochs.close()

    
#torch.save(model.state_dict(), "model_200e_ce_new_dataset_3class.pth")