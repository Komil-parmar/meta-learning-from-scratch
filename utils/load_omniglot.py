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
        """
        Initialize the Omniglot dataset loader.
        
        Scans the data directory to find all character classes across different
        alphabets and stores their paths for later access.
        
        Args:
            data_path (str): Path to the root directory of the Omniglot dataset.
            transform (callable, optional): Optional transform to be applied to images.
                Defaults to None.
        
        Attributes:
            character_paths (list): List of paths to all character directories found
                in the dataset.
        """
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
        """
        Return the total number of character classes in the dataset.
        
        Returns:
            int: Number of character classes available.
        """
        return len(self.character_paths)
    
    def __getitem__(self, idx):
        """
        Retrieve all images for a specific character class.
        
        Loads all available samples for the character at the given index,
        preprocesses them (grayscale conversion, resize to 105x105, normalization),
        and returns them as a tensor stack.
        
        Args:
            idx (int): Index of the character class to retrieve.
        
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Stack of image tensors with shape (N, 1, 105, 105),
                  where N is the number of samples for this character.
                - int: The index of the character class.
        """
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


class PrefetchedOmniglotDataset(Dataset):
    """
    Prefetched Omniglot Dataset - loads entire dataset into RAM for faster access.
    
    This class loads all character images into memory during initialization,
    significantly speeding up data access during training. Ideal for Omniglot
    since the entire dataset is relatively small (~1.5GB uncompressed).
    
    Args:
        data_path (str): Path to Omniglot data directory
        transform (callable, optional): Optional transform to be applied
        
    Memory usage: Approximately 200-300 MB for background set, 100-150 MB for evaluation set
    """
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.character_data = []  # List of tensors, one per character class
        self.character_paths = []
        
        print("🚀 Prefetching Omniglot dataset into RAM...")
        print("   This may take 20-30 seconds but will speed up training significantly!")
        
        # Get all character folders
        alphabets = sorted(os.listdir(data_path))
        total_chars = 0
        
        # Count total characters for progress bar
        for alphabet in alphabets:
            alphabet_path = os.path.join(data_path, alphabet)
            if os.path.isdir(alphabet_path):
                characters = os.listdir(alphabet_path)
                total_chars += len([c for c in characters if os.path.isdir(os.path.join(alphabet_path, c))])
        
        # Load all character data into memory
        with tqdm(total=total_chars, desc="Loading characters into RAM") as pbar:
            for alphabet in alphabets:
                alphabet_path = os.path.join(data_path, alphabet)
                if not os.path.isdir(alphabet_path):
                    continue
                    
                characters = sorted(os.listdir(alphabet_path))
                for char in characters:
                    char_path = os.path.join(alphabet_path, char)
                    if not os.path.isdir(char_path):
                        continue
                    
                    # Load all images for this character
                    image_files = sorted([f for f in os.listdir(char_path) if f.endswith('.png')])
                    image_tensors = []
                    
                    for img_name in image_files:
                        img_path = os.path.join(char_path, img_name)
                        img = Image.open(img_path).convert('L')
                        img = img.resize((105, 105))
                        img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
                        image_tensors.append(img_tensor)
                    
                    # Stack all images for this character
                    if image_tensors:
                        self.character_data.append(torch.stack(image_tensors))
                        self.character_paths.append(char_path)
                    
                    pbar.update(1)
        
        # Calculate memory usage
        total_memory = sum(char_data.element_size() * char_data.nelement() 
                          for char_data in self.character_data)
        memory_mb = total_memory / (1024 * 1024)
        
        print(f"✅ Prefetching complete!")
        print(f"   📊 Loaded {len(self.character_data)} character classes")
        print(f"   💾 Memory usage: {memory_mb:.1f} MB")
        print(f"   ⚡ Data access will now be ~10-50x faster!")
    
    def __len__(self):
        return len(self.character_data)
    
    def __getitem__(self, idx):
        """
        Returns all images for a character class.
        
        Returns:
            tuple: (images, idx) where images is a tensor of shape [num_images, 1, 105, 105]
        """
        return self.character_data[idx], idx

class OmniglotTaskDataset(Dataset):
    """Dataset for generating N-way K-shot tasks"""
    
    def __init__(self, omniglot_dataset, n_way=5, k_shot=1, k_query=15, num_tasks=1000):
        """
        Initialize the task generator for few-shot learning.
        
        Creates a dataset that generates episodic tasks for meta-learning,
        where each task consists of N classes with K support examples and
        query examples per class.
        
        Args:
            omniglot_dataset (OmniglotDataset): The underlying Omniglot dataset
                to sample characters from.
            n_way (int, optional): Number of classes per task. Defaults to 5.
            k_shot (int, optional): Number of support examples per class. Defaults to 1.
            k_query (int, optional): Number of query examples per class. Defaults to 15.
            num_tasks (int, optional): Total number of tasks to generate. Defaults to 1000.
        """
        self.omniglot_dataset = omniglot_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_tasks = num_tasks
        
    def __len__(self):
        """
        Return the total number of tasks in the dataset.
        
        Returns:
            int: Number of tasks that will be generated.
        """
        return self.num_tasks
    
    def __getitem__(self, idx):
        """
        Generate a single N-way K-shot task.
        
        Randomly samples N character classes and creates support and query sets
        for a few-shot learning task. If a character class doesn't have enough
        images, some images are repeated to meet the requirements.
        
        Args:
            idx (int): Index of the task (note: tasks are randomly generated,
                so the same index may produce different tasks).
        
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Support set images with shape (N*K, 1, 105, 105).
                - torch.Tensor: Support set labels with shape (N*K,).
                - torch.Tensor: Query set images with shape (N*Q, 1, 105, 105),
                  where Q is k_query.
                - torch.Tensor: Query set labels with shape (N*Q,).
        """
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