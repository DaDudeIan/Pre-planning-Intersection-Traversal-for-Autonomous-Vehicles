import os
import json
import numpy as np
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class IntersectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, path_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.path_transform = path_transform
        
        self.path_dirs = glob.glob(f'{root_dir}/*/paths/*')
        
    def __len__(self):
        return len(self.path_dirs)
    
    def __getitem__(self, idx):
        path_dir = self.path_dirs[idx]
        
        # Load satellite image (../../satellite.png)
        satellite_path = os.path.join(os.path.dirname(os.path.dirname(path_dir)), 'satellite.png')
        satellite_img = Image.open(satellite_path).convert('RGB')
        
        if self.transform:
            satellite_img = self.transform(satellite_img)
            
        # load path line image (./path_line.png)
        path_line_path = os.path.join(path_dir, 'path_line.png')
        path_line_img = Image.open(path_line_path).convert('L')
        
        if self.path_transform:
            path_line_img = self.path_transform(path_line_img)
            
        # load E/E json file (./path_line_ee.json)
        json_path = os.path.join(path_dir, 'path_line_ee.json')
        with open(json_path) as f:
            ee_data = json.load(f)
            
        # load cold map npy (./cold_map.npy)
        cold_map_path = os.path.join(path_dir, 'cold_map.npy')
        cold_map = np.load(cold_map_path)
        
        # return sample
        sample = {
            'satellite': satellite_img,
            'path_line': path_line_img,
            'ee_data': ee_data,
            'cold_map': cold_map
        }
        return sample

class IntersectionDataset2(Dataset):
    def __init__(self, root_dir, transform=None, path_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.path_transform = path_transform
        
        self.intersections = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f))
        ]
        
    def __len__(self):
        return len(self.intersections)
    
    def __getitem__(self, idx):
        intersection_dir = self.intersections[idx]
        
        # Load satellite image
        satellite_path = os.path.join(intersection_dir, 'satellite.png')
        satellite_img = Image.open(satellite_path).convert('RGB')
        
        if self.transform:
            satellite_img = self.transform(satellite_img)
            
        # Store paths for each satellite image
        paths_data = []
        paths_dir = os.path.join(intersection_dir, 'paths')
        if os.path.exists(paths_dir):
            path_folders = [
                os.path.join(paths_dir, f) 
                for f in os.listdir(paths_dir) 
                if os.path.isdir(os.path.join(paths_dir, f))
            ]
            
            for path_folder in path_folders:
                # Path line image
                path_line_path = os.path.join(path_folder, 'path_line.png')
                path_line_img = Image.open(path_line_path).convert('RGB')
                
                if self.path_transform:
                    path_line_img = self.path_transform(path_line_img)
                    
                # E/E json file
                json_path = os.path.join(path_folder, 'path_line_ee.json')
                with open(json_path) as f:
                    ee_data = json.load(f)
                    
                # Load cold map npy
                cold_map_path = os.path.join(path_folder, 'cold_map.npy')
                cold_map = np.load(cold_map_path)
                
                # save data
                paths_data.append({
                    'path_line': path_line_img,
                    'ee_data': ee_data,
                    'cold_map': cold_map
                })
               
        # return sample 
        sample = {
            'satellite': satellite_img,
            'paths': paths_data
        }
        return sample
    
class IntersectionDatasetClasses(Dataset):
    def __init__(self, root_dir, transform=None, path_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.path_transform = path_transform
        
        self.intersections = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f))
        ]
        
    def __len__(self):
        return len(self.intersections)
    
    def __getitem__(self, idx):
        intersection_dir = self.intersections[idx]
        
        # Load satellite image
        satellite_path = os.path.join(intersection_dir, 'satellite.png')
        satellite_img = Image.open(satellite_path).convert('RGB')
        
        if self.transform:
            satellite_img = self.transform(satellite_img)
            
        # Load class labels image
        class_labels_path = os.path.join(intersection_dir, 'class_labels.npy')
        class_labels = np.load(class_labels_path)
        
        if self.path_transform:
            class_labels = self.path_transform(class_labels)
            
        # Store paths for each satellite image
        paths_data = []
        paths_dir = os.path.join(intersection_dir, 'paths')
        if os.path.exists(paths_dir):
            path_folders = [
                os.path.join(paths_dir, f) 
                for f in os.listdir(paths_dir) 
                if os.path.isdir(os.path.join(paths_dir, f))
            ]
            
            for path_folder in path_folders:
                # Path line image
                path_line_path = os.path.join(path_folder, 'path_line.png')
                path_line_img = Image.open(path_line_path).convert('RGB')
                
                if self.path_transform:
                    path_line_img = self.path_transform(path_line_img)
                    
                # E/E json file
                json_path = os.path.join(path_folder, 'path_line_ee.json')
                with open(json_path) as f:
                    ee_data = json.load(f)
                    
                # Load cold map npy
                cold_map_path = os.path.join(path_folder, 'cold_map.npy')
                cold_map = np.load(cold_map_path)
                
                # save data
                paths_data.append({
                    'path_line': path_line_img,
                    'ee_data': ee_data,
                    'cold_map': cold_map
                })
               
        # return sample 
        sample = {
            'satellite': satellite_img,
            'class_labels': class_labels,
            'paths': paths_data
        }
        return sample
                    
                
def custom_collate_fn(batch):
    """
    Custom collate function that handles the variable-length 'paths' list.
    For the satellite images, we stack them into a tensor.
    For the 'paths' field, we simply collect them into a list.
    """
    satellite_batch = torch.stack([item["satellite"] for item in batch])
    class_labels_batch = torch.stack([item["class_labels"] for item in batch])
    # Keep 'paths' as a list of lists (variable-length) without stacking.
    paths_batch = [item["paths"] for item in batch]
    return {"satellite": satellite_batch, "class_labels": class_labels_batch, "paths": paths_batch}


def main():
    from torchvision.transforms import ToTensor
    import multiprocessing
    b = 4
    num_workers = multiprocessing.cpu_count()
    
    dataset_dir = "dataset"
    img_transform = ToTensor()
    path_transform = ToTensor()
    dataset = IntersectionDataset(root_dir=dataset_dir,
                                transform=img_transform,
                                path_transform=path_transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=b, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)