import sys
import os
import torch
from pathlib import Path
import datetime

project_root = Path.cwd().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
models_path = Path.cwd().parent
if str(models_path) not in sys.path:
    sys.path.append(str(models_path))
    
from torch.utils.data import DataLoader
import multiprocessing
from torchvision.transforms import ToTensor
import torch
from torch.optim import Adam, AdamW
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
    
file_name = os.path.basename(__file__).lower()
from dataset.IntersectionDataset import IntersectionDatasetClasses, custom_collate_fn
import loss.loss_lib as ll
import loss.topo_lib as tl

# ================= Datasets =====================

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
#print(len(dataset_train))
#print(len(dataset_test))

num_workers = multiprocessing.cpu_count()
b = 4

train_dataloader = DataLoader(dataset_train, batch_size=b, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(dataset_test, batch_size=b, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_fn)

# ====================== Names ======================

if "deeplab" in file_name:
    import deeplabv3.Deeplabv3 as d
    model_name = "deeplab"
elif "unet" in file_name:
    import unet.Unet as u
    model_name = "unet"
elif "vit" in file_name:
    import vit.ViT as v
    model_name = "vit"
elif "swin" in file_name:
    import swin.Swin as s
    model_name = "swin"
else:
    raise ValueError("Model type not recognized in file name.")

loss_fn = "ce-topo"
optimizer_name = "adamw"
scheduler_name = "cos"

# ====================== Model ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
match model_name:
    case "deeplab":
        model = d.DeepLabV3().to(device) 
    case "unet":
        model = u.Unet().to(device)
    case "vit":
        model = v.ViT().to(device)
    case "swin":
        model = s.Swin().to(device)
    case _:
        raise ValueError("Model type not recognized.")
    
# ====================== alpha ======================

alpha_high = 0.99
alpha_low = 0.5
T_warm = 30

# ====================== optimizer ======================

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
#optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


# ====================== Scheduler ======================

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-5)
#scheduler = ExponentialLR(optimizer, gamma=0.99)

scaler = GradScaler(device=device)

# ====================== Loss ======================

class_counts = torch.tensor([152000, 2000, 2000, 2000, 2000], dtype=torch.float)
weights = 1.0 / class_counts
weights = torch.log1p(class_counts.sum() / class_counts)

lce = nn.CrossEntropyLoss(weight=weights.to(device)).to(device)
ltopo = tl.TopologyLoss().to(device)


# ====================== Training ======================

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

n_epochs = 100

save_epochs = [10, 20, 50, 100]

time_of_start = datetime.datetime.now()

for epoch in range(n_epochs):
    
    if (epoch + 1) % 20 == 0:
        time_of_epoch = datetime.datetime.now()
        time_diff = time_of_epoch - time_of_start
        time_diff = time_diff.total_seconds()
        print(f"Time elapsed (s) {time_diff:.0f} | Epoch {epoch + 1}/{n_epochs} | Train Loss: {running_train_loss / len(train_dataloader):.4f} | Train Accuracy: {running_train_correct / running_train_total:.4f} | Test Loss: {running_test_loss / len(test_dataloader):.4f} | Test Accuracy: {running_test_correct / running_test_total:.4f}")

    model.train()
    running_train_loss = 0.0
    running_train_correct = 0
    running_train_total = 0
    
    alpha = tl.alpha(epoch = epoch, alpha_hi = alpha_high, alpha_lo = alpha_low, T_warm = T_warm, N_epochs = n_epochs)
        
    for i, batch in enumerate(train_dataloader):
        satellite = batch["satellite"].to(device, non_blocking=True)
        class_labels = batch["class_labels"].to(device, non_blocking=True)
        
        class_labels = class_labels.squeeze(1)
        
        optimizer.zero_grad()
        
        with autocast("cuda"):
            output = model(satellite)
            L_ce = lce(output, class_labels)
            L_topo = ltopo(output)
            
            loss = alpha * L_ce + (1 - alpha) * L_topo
            
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_train_loss += loss.item()
        
        p = torch.argmax(output, dim=1)
        running_train_correct += (p == class_labels).sum().item()
        running_train_total += class_labels.size(0)
                
        
    avg_train_loss = running_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_accuracy = running_train_correct / running_train_total
    train_accuracies.append(train_accuracy)
    
    model.eval()
    running_test_loss = 0.0
    running_test_correct = 0
    running_test_total = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            satellite = batch["satellite"].to(device, non_blocking=True)
            class_labels = batch["class_labels"].to(device, non_blocking=True)
            
            class_labels = class_labels.squeeze(1)
            
            with autocast("cuda"):
                output = model(satellite)
                L_ce = lce(output, class_labels)
                L_topo = ltopo(output)
                
                loss = alpha * L_ce + (1 - alpha) * L_topo
            
            running_test_loss += loss.item()
            
            p = torch.argmax(output, dim=1)
            running_test_correct += (p == class_labels).sum().item()
            running_test_total += class_labels.size(0)
                        
    avg_test_loss = running_test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    
    test_accuracy = running_test_correct / running_test_total
    test_accuracies.append(test_accuracy)
    
    scheduler.step()  
    
    if (epoch + 1) in save_epochs:
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
        save_dir = "/mnt/e/msc_checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        
        # f"c_{model}_{M}_{D}_{epoch}_{loss_fn}_{optimizer}_{scheduler}_{[opt]}"
        date = datetime.datetime.now().strftime("%m_%d")
        torch.save(checkpoint, f'{save_dir}/c_{model_name}_{date}_{epoch + 1}_{loss_fn}_{optimizer_name}_{scheduler_name}.pth')
        print(f"Checkpoint saved at epoch {epoch + 1}")

    
#torch.save(model.state_dict(), "model_200e_ce_new_dataset_3class.pth")