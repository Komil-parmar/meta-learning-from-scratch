import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm



class OmniglotDataset(Dataset):
    """Dataset class for Omniglot characters"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.character_paths = []
        
        print("Loading character paths...")
        # Get all character folders
        alphabets = os.listdir(data_path)
        for alphabet in tqdm(alphabets, desc="Processing alphabets"):
            alphabet_path = os.path.join(data_path, alphabet)
            if os.path.isdir(alphabet_path):
                characters = os.listdir(alphabet_path)
                for char in characters:
                    char_path = os.path.join(alphabet_path, char)
                    if os.path.isdir(char_path):
                        self.character_paths.append(char_path)
        
        print(f"Found {len(self.character_paths)} character classes")
    
    def __len__(self):
        return len(self.character_paths)
    
    def __getitem__(self, idx):
        char_path = self.character_paths[idx]
        images = [f for f in os.listdir(char_path) if f.endswith('.png')]
        
        # Load all images for this character
        image_tensors = []
        for img_name in images:
            img_path = os.path.join(char_path, img_name)
            img = Image.open(img_path).convert('L')
            img = img.resize((105, 105))
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
            image_tensors.append(img_tensor)
        
        return torch.stack(image_tensors), idx

class OmniglotTaskDataset(Dataset):
    """Dataset for generating N-way K-shot tasks"""
    def __init__(self, omniglot_dataset, n_way=5, k_shot=1, k_query=15, num_tasks=1000):
        self.omniglot_dataset = omniglot_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_tasks = num_tasks
        
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        # Sample N random character classes
        selected_chars = random.sample(range(len(self.omniglot_dataset)), self.n_way)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for class_idx, char_idx in enumerate(selected_chars):
            char_images, _ = self.omniglot_dataset[char_idx]
            
            # Randomly select images for support and query
            available_images = len(char_images)
            if available_images < self.k_shot + self.k_query:
                # If not enough images, repeat some
                indices = torch.randperm(available_images).repeat(
                    (self.k_shot + self.k_query + available_images - 1) // available_images
                )[:self.k_shot + self.k_query]
            else:
                indices = torch.randperm(available_images)[:self.k_shot + self.k_query]
            
            # Split into support and query
            support_indices = indices[:self.k_shot]
            query_indices = indices[self.k_shot:self.k_shot + self.k_query]
            
            # Add support images
            for idx in support_indices:
                support_data.append(char_images[idx])
                support_labels.append(class_idx)
            
            # Add query images
            for idx in query_indices:
                query_data.append(char_images[idx])
                query_labels.append(class_idx)
        
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        return support_data, support_labels, query_data, query_labels