"""
Latent Embedding Optimization (LEO) Implementation.

This module provides a PyTorch implementation of LEO, a meta-learning algorithm
that learns to adapt to new tasks by optimizing in a learned low-dimensional latent space.

Classes:
    LatentEmbeddingOptimization: Core LEO algorithm implementation
    LEOEncoder: Encoder network that maps task data to latent codes
    LEORelationNetwork: Relation network that processes pairs of examples
    LEODecoder: Decoder network that maps latent codes to model parameters
    
Functions:
    train_leo: High-level training function with progress tracking

Reference:
    Rusu, A. A., et al. (2019). Meta-Learning with Latent Embedding Optimization.
    ICLR 2019.
    https://arxiv.org/abs/1807.05960
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, List, Tuple


class LEOEncoder(nn.Module):
    """
    Encoder network for LEO that maps CNN features to latent representations.
    
    The encoder takes pre-extracted features from the shared CNN classifier
    and projects them to a low-dimensional latent space. This ensures that
    both encoding and classification use the same feature extractor.
    
    Architecture:
    - Takes CNN features (64*6*6 = 2304 dimensional)
    - Projects to latent space via single linear layer
    - Output: latent_dim dimensional vectors
    
    Args:
        latent_dim: Dimension of the latent space (default: 64)
        feature_dim: Dimension of CNN features (default: 2304)
    """
    
    def __init__(self, latent_dim: int = 64, feature_dim: int = 2304):
        super(LEOEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Project CNN features to latent space
        self.fc = nn.Linear(feature_dim, latent_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            features: CNN features [batch_size, feature_dim]
            
        Returns:
            Latent representations [batch_size, latent_dim]
        """
        # Project to latent space
        latent = self.fc(features)
        
        return latent


class LEORelationNetwork(nn.Module):
    """
    Relation network for LEO that processes pairs of encoded examples.
    
    This network computes relationships between examples to generate
    class-specific latent codes that capture task structure.
    
    Args:
        latent_dim: Dimension of the latent space (default: 64)
    """
    
    def __init__(self, latent_dim: int = 64):
        super(LEORelationNetwork, self).__init__()
        self.latent_dim = latent_dim
        
        # Relation network layers
        self.fc1 = nn.Linear(latent_dim * 2, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute relation between two latent codes.
        
        Args:
            z1: First latent code [batch_size, latent_dim]
            z2: Second latent code [batch_size, latent_dim]
            
        Returns:
            Relation output [batch_size, latent_dim]
        """
        # Concatenate and process
        x = torch.cat([z1, z2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class LEODecoder(nn.Module):
    """
    Decoder network for LEO that maps latent codes to class prototypes.
    
    This network generates prototype vectors (one per class) from latent codes.
    Each prototype is a vector in the feature space that represents a class.
    Classification is done by computing similarity between query features and prototypes.
    
    Architecture:
        Latent code [latent_dim] → MLP → Prototype [feature_dim]
        For N classes: [N, latent_dim] → Decoder → [N, feature_dim]
    
    This is the correct LEO approach: learning prototypes in a metric space
    rather than generating full FC layer parameters.
    
    Args:
        latent_dim: Dimension of the latent space (default: 64)
        feature_dim: Dimension of prototype vectors (default: 2304 = 64*6*6)
    """
    
    def __init__(self, latent_dim: int = 64, feature_dim: int = 2304):
        super(LEODecoder, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Decoder layers - generates prototype vector from latent code
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, feature_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to prototype vectors.
        
        Args:
            z: Latent code [N, latent_dim] or [latent_dim]
               For N classes, expects [N, latent_dim]
            
        Returns:
            Prototype vectors [N, feature_dim] or [feature_dim]
            Each row is a prototype vector for one class
        """
        # Decode latent code to prototype vector
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        prototypes = self.fc3(x)  # [N, feature_dim] or [feature_dim]
        
        return prototypes


class LEOClassifier(nn.Module):
    """
    CNN classifier for LEO with shared feature extractor.
    
    The CNN feature extractor (conv layers) is shared and fixed across all tasks.
    Only the final FC layer parameters are generated by the decoder for each task.
    
    Args:
        num_classes: Number of output classes (default: 5)
    """
    
    def __init__(self, num_classes: int = 5):
        super(LEOClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Shared feature extractor (fixed across tasks)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Task-specific FC layer (parameters generated by decoder)
        self.fc = nn.Linear(64 * 6 * 6, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using shared CNN backbone.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Features [batch_size, feature_dim]
        """
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x: torch.Tensor, prototypes: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional prototype vectors for classification.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            prototypes: Optional prototype vectors [num_classes, feature_dim]
                       If None, uses default self.fc parameters
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract features using shared CNN
        features = self.extract_features(x)  # [batch_size, feature_dim]
        
        # Classify using prototypes or default FC
        if prototypes is None:
            # Use default FC parameters
            logits = self.fc(features)
        else:
            # Use provided prototypes as classifier weights
            # features: [batch, 2304], prototypes: [N, 2304]
            # Result: [batch, N] - similarity scores
            logits = features @ prototypes.T  # Matrix multiplication
        
        return logits


class LatentEmbeddingOptimization:
    """
    Latent Embedding Optimization (LEO) algorithm for few-shot learning.
    
    LEO learns to adapt to new tasks by optimizing in a learned low-dimensional
    latent space rather than directly in the high-dimensional parameter space.
    The algorithm consists of four main components:
    
    1. **Shared CNN Classifier**: Feature extractor shared across all tasks
    2. **Encoder**: Maps CNN features (64*6*6) to latent space (e.g., 64D)
    3. **Relation Network**: Processes relationships between examples
    4. **Decoder**: Generates class prototype vectors from latent codes
    
    Architecture Flow:
        ```
        Input Images [N*K, 1, 105, 105]
                ↓
        Shared CNN (conv1-4) → Features [N*K, 2304]
                ↓                      ↓
            Encoder                Classifier
                ↓                      ↓
        Latent [N*K, 64]      Prototypes → Logits
                ↓
        Per-class aggregation → [N, 64]
                ↓
        Relation Network (refine via cross-class relations)
                ↓
        Refined Latent [N, 64]
                ↓
            Decoder
                ↓
        Prototypes [N, 2304]
        ```
    
    Algorithm Overview:
        ```
        Initialize encoder, decoder, relation network, shared CNN
        while not converged:
            Sample batch of tasks τ ~ p(τ)
            for each task τᵢ in batch:
                # Extract features and encode to latent space
                features = shared_CNN(support_set)
                z = encoder(features)
                
                # Aggregate per class
                z_class = aggregate_by_class(z, labels)
                
                # Refine using relation network
                z_refined = relation_network(z_class)
                
                # Generate initial prototypes
                prototypes = decoder(z_refined)
                
                # Inner loop: Optimize in latent space
                for k steps:
                    z = z - α∇_z L_τᵢ(shared_CNN, decoder(z))
                
                # Decode to adapted prototypes
                prototypes' = decoder(z)
                
                # Outer loop: Evaluate on query set
                Compute L_τᵢ(shared_CNN, θ_fc') on query set
            
            # Meta-update all components
            Update encoder, decoder, relation network, shared CNN
        ```
    
    Attributes:
        encoder (LEOEncoder): 
            Encoder network that maps inputs to latent space
            
        decoder (LEODecoder): 
            Decoder network that maps latent codes to parameters
            
        relation_net (LEORelationNetwork):
            Relation network for processing example pairs
            
        classifier (LEOClassifier):
            Shared CNN feature extractor + task-specific FC layer
            
        inner_lr (float): 
            Learning rate for latent code optimization (inner loop)
            
        outer_lr (float): 
            Meta-learning rate for encoder/decoder updates (outer loop)
            
        inner_steps (int): 
            Number of gradient steps in latent space (inner loop)
            
        latent_dim (int):
            Dimension of the latent space
            
        meta_optimizer (torch.optim.Optimizer): 
            Optimizer for meta-parameter updates
    
    Methods:
        encode_task: Encode support set to latent codes
        decode_to_params: Decode latent codes to model parameters
        inner_update: Optimize latent codes for task adaptation
        meta_train_step: Perform one meta-training step on a batch of tasks
    
    Example:
        >>> # Basic usage
        >>> leo = LatentEmbeddingOptimization(
        ...     num_classes=5,
        ...     latent_dim=64,
        ...     inner_lr=0.01,
        ...     outer_lr=0.001,
        ...     inner_steps=5
        ... )
        >>> 
        >>> # Training loop
        >>> for task_batch in task_dataloader:
        ...     loss = leo.meta_train_step(task_batch)
        >>> 
        >>> # Adapt to new task
        >>> latent_codes = leo.encode_task(support_data, support_labels)
        >>> adapted_codes = leo.inner_update(support_data, support_labels, latent_codes)
        >>> params = leo.decode_to_params(adapted_codes)
        >>> predictions = leo.classifier(query_data, params)
    
    Notes:
        - LEO operates in a low-dimensional latent space (e.g., 64D)
        - This makes optimization more efficient than MAML's parameter space
        - The decoder generates all parameters from compact latent codes
        - Supports any number of classes and latent dimensions
    
    Reference:
        Rusu, A. A., et al. (2019). Meta-Learning with Latent Embedding Optimization.
        ICLR 2019.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        latent_dim: int = 64,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        optimizer_cls: type = optim.Adam,
        optimizer_kwargs: dict = None
    ):
        """
        Initialize LEO algorithm.
        
        Args:
            num_classes: Number of classes per task
            latent_dim: Dimension of latent space
            inner_lr: Learning rate for latent code optimization
            outer_lr: Meta-learning rate for encoder/decoder
            inner_steps: Number of inner loop optimization steps
            optimizer_cls: Optimizer class for meta-learning
            optimizer_kwargs: Additional optimizer arguments
        """
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        # Initialize networks
        self.encoder = LEOEncoder(latent_dim=latent_dim)
        self.decoder = LEODecoder(latent_dim=latent_dim, feature_dim=2304)
        self.relation_net = LEORelationNetwork(latent_dim=latent_dim)
        self.classifier = LEOClassifier(num_classes=num_classes)
        
        # Combine all meta-learnable parameters
        # Include classifier CNN backbone + encoder + decoder + relation network
        meta_params = list(self.encoder.parameters()) + \
                     list(self.decoder.parameters()) + \
                     list(self.relation_net.parameters()) + \
                     list(self.classifier.parameters())  # Shared CNN + default FC
        
        # Initialize meta-optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        self.meta_optimizer = optimizer_cls(
            meta_params,
            lr=outer_lr,
            **optimizer_kwargs
        )
        
    def to(self, device: torch.device):
        """Move all networks to device."""
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.relation_net = self.relation_net.to(device)
        self.classifier = self.classifier.to(device)
        return self
    
    def train(self):
        """Set all networks to training mode."""
        self.encoder.train()
        self.decoder.train()
        self.relation_net.train()
        self.classifier.train()
    
    def eval(self):
        """Set all networks to evaluation mode."""
        self.encoder.eval()
        self.decoder.eval()
        self.relation_net.eval()
        self.classifier.eval()
    
    def apply_relation_network(self, class_codes: torch.Tensor) -> torch.Tensor:
        """
        Apply relation network to refine class latent codes.
        
        For each class, compute its relation with all classes (including itself),
        then average the relation outputs to get refined latent codes.
        
        Args:
            class_codes: Initial class latent codes [N, latent_dim]
            
        Returns:
            Refined latent codes [N, latent_dim]
        """
        num_classes = class_codes.size(0)
        refined_codes = []
        
        for i in range(num_classes):
            # Get latent code for class i
            class_i = class_codes[i]  # [latent_dim]
            
            # Compute relation with all classes (including itself)
            relations = []
            for j in range(num_classes):
                class_j = class_codes[j]  # [latent_dim]
                
                # Compute relation between class i and class j
                relation_output = self.relation_net(
                    class_i.unsqueeze(0),  # [1, latent_dim]
                    class_j.unsqueeze(0)   # [1, latent_dim]
                )  # [1, latent_dim]
                
                relations.append(relation_output.squeeze(0))
            
            # Stack all relations: [N, latent_dim]
            relations = torch.stack(relations, dim=0)
            
            # Average to get refined code for class i
            refined_code = relations.mean(dim=0)  # [latent_dim]
            refined_codes.append(refined_code)
        
        # Stack to [N, latent_dim]
        refined_codes = torch.stack(refined_codes, dim=0)
        
        return refined_codes
    
    def encode_task(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        Encode support set to latent codes using shared CNN features.
        
        This method first extracts features using the classifier's CNN backbone,
        then encodes them to latent space, aggregates per-class, and refines
        using the relation network.
        
        Args:
            support_data: Support set images [N*K, C, H, W]
            support_labels: Support set labels [N*K]
            
        Returns:
            Latent codes [N, latent_dim] (one per class, refined by relation net)
        """
        # Extract features using shared CNN (WITH gradients for inner_update)
        features = self.classifier.extract_features(support_data)  # [N*K, 2304]
        
        # Encode features to latent space
        latent_codes = self.encoder(features)  # [N*K, latent_dim]
        
        # Aggregate per class
        class_codes = []
        for class_idx in range(self.num_classes):
            class_mask = support_labels == class_idx
            class_examples = latent_codes[class_mask]
            
            if len(class_examples) > 0:
                # Average pooling over examples of same class
                class_code = class_examples.mean(dim=0)
            else:
                # Handle case where class is missing (shouldn't happen in practice)
                class_code = torch.zeros(self.latent_dim, device=latent_codes.device)
            
            class_codes.append(class_code)
        
        # Stack to [N, latent_dim]
        class_codes = torch.stack(class_codes, dim=0)
        
        # Refine using relation network
        refined_codes = self.apply_relation_network(class_codes)
        
        return refined_codes
    
    def decode_to_prototypes(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to prototype vectors.
        
        Args:
            latent_codes: Latent codes [N, latent_dim] or [latent_dim]
            
        Returns:
            Prototype vectors [N, feature_dim] or [feature_dim]
        """
        # Decode to prototypes
        prototypes = self.decoder(latent_codes)
        
        return prototypes
    
    def inner_update(
        self, 
        support_data: torch.Tensor, 
        support_labels: torch.Tensor,
        initial_codes: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimize latent codes for task adaptation (inner loop).
        
        This method performs gradient descent in the latent space to adapt
        to the current task using the support set.
        
        Args:
            support_data: Support set images [N*K, C, H, W]
            support_labels: Support set labels [N*K]
            initial_codes: Initial latent codes [N, latent_dim]
            
        Returns:
            Optimized latent codes [N, latent_dim]
        """
        # Clone initial codes and enable gradients
        latent_codes = initial_codes.clone().detach().requires_grad_(True)
        
        # Inner loop optimization
        for step in range(self.inner_steps):
            # Decode to prototypes
            prototypes = self.decode_to_prototypes(latent_codes)
            
            # Forward pass with decoded prototypes
            logits = self.classifier(support_data, prototypes)
            
            # Compute loss
            loss = F.cross_entropy(logits, support_labels)
            
            # Compute gradient w.r.t. latent codes
            grad = torch.autograd.grad(
                loss,
                latent_codes,
                create_graph=True  # Enable second-order gradients for meta-learning
            )[0]
            
            # Update latent codes
            latent_codes = latent_codes - self.inner_lr * grad
        
        return latent_codes
    
    def meta_train_step(
        self,
        support_data_batch: torch.Tensor,
        support_labels_batch: torch.Tensor,
        query_data_batch: torch.Tensor,
        query_labels_batch: torch.Tensor
    ) -> float:
        """
        Perform one meta-training step across a batch of tasks.
        
        This method implements the outer loop of LEO, which updates the encoder,
        decoder, and relation network based on query set performance after
        latent code optimization.
        
        Algorithm:
            ```
            Zero gradients
            meta_loss = 0
            
            for each task in batch:
                # Encode to latent space
                z = encode_task(support_set)
                
                # Optimize in latent space
                z' = inner_update(support_set, z)
                
                # Decode to parameters
                θ' = decode_to_params(z')
                
                # Evaluate on query set
                L_query = loss(θ', query_set)
                meta_loss += L_query
            
            # Meta-update
            Update encoder, decoder, relation network
            return average_meta_loss
            ```
        
        Args:
            support_data_batch: Support data for all tasks [batch_size, N*K, C, H, W]
            support_labels_batch: Support labels for all tasks [batch_size, N*K]
            query_data_batch: Query data for all tasks [batch_size, N*Q, C, H, W]
            query_labels_batch: Query labels for all tasks [batch_size, N*Q]
            
        Returns:
            Average meta-loss across all tasks in the batch
        
        Example:
            >>> support_data_batch = torch.randn(4, 5, 1, 105, 105)
            >>> support_labels_batch = torch.randint(0, 5, (4, 5))
            >>> query_data_batch = torch.randn(4, 75, 1, 105, 105)
            >>> query_labels_batch = torch.randint(0, 5, (4, 75))
            >>> 
            >>> loss = leo.meta_train_step(
            ...     support_data_batch, support_labels_batch,
            ...     query_data_batch, query_labels_batch
            ... )
        """
        self.train()
        self.meta_optimizer.zero_grad()
        
        # Initialize meta loss
        device = next(self.encoder.parameters()).device
        meta_loss_sum = torch.tensor(0.0, device=device)
        
        batch_size = support_data_batch.size(0)
        
        for i in range(batch_size):
            support_data = support_data_batch[i]
            support_labels = support_labels_batch[i]
            query_data = query_data_batch[i]
            query_labels = query_labels_batch[i]
            
            # Encode support set to latent codes
            initial_codes = self.encode_task(support_data, support_labels)
            
            # Inner loop: Optimize latent codes
            adapted_codes = self.inner_update(support_data, support_labels, initial_codes)
            
            # Decode to adapted prototypes
            adapted_prototypes = self.decode_to_prototypes(adapted_codes)
            
            # Evaluate on query set
            query_logits = self.classifier(query_data, adapted_prototypes)
            query_loss = F.cross_entropy(query_logits, query_labels)
            
            # Accumulate meta loss
            meta_loss_sum = meta_loss_sum + query_loss
        
        # Average over batch
        meta_loss = meta_loss_sum / batch_size
        
        # Backpropagate through meta-parameters
        meta_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.relation_net.parameters()),
            max_norm=1.0
        )
        
        # Meta-update
        self.meta_optimizer.step()
        
        return meta_loss.item()


# Convenience alias
LEO = LatentEmbeddingOptimization


def train_leo(
    num_classes: int,
    task_dataloader: torch.utils.data.DataLoader,
    latent_dim: int = 64,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    optimizer_cls: type = optim.Adam,
    optimizer_kwargs: dict = None
):
    """
    Train using Latent Embedding Optimization (LEO).
    
    This function implements the complete LEO training pipeline, which learns
    to adapt to new tasks by optimizing in a learned low-dimensional latent space.
    
    Algorithm:
        For each batch of tasks:
            1. Encode support set to latent space
            2. Inner Loop (Latent Optimization):
               - Optimize latent codes on support set
               - Decode to task-specific parameters
            3. Outer Loop (Meta-Learning):
               - Evaluate on query set
               - Update encoder, decoder, relation network
    
    Args:
        num_classes: Number of classes per task (e.g., 5 for 5-way)
        
        task_dataloader: DataLoader that yields batches of tasks
            Each task: (support_data, support_labels, query_data, query_labels)
            
        latent_dim: Dimension of latent space (default: 64)
            - Smaller (32): More compression, faster but may lose information
            - Larger (128): More capacity, slower but better performance
            
        inner_lr: Learning rate for latent code optimization (default: 0.01)
            Controls adaptation speed in latent space
            
        outer_lr: Meta-learning rate for encoder/decoder (default: 0.001)
            Controls meta-parameter update speed
            
        inner_steps: Number of latent optimization steps (default: 5)
            More steps = better adaptation but slower training
            
        optimizer_cls: Optimizer for meta-learning (default: Adam)
            
        optimizer_kwargs: Additional optimizer arguments (default: None)
    
    Returns:
        tuple: (leo, losses)
            - leo: Trained LEO object
            - losses: Training loss history
    
    Example:
        >>> # Basic usage
        >>> task_dataset = OmniglotTaskDataset(dataset, n_way=5, k_shot=1, num_tasks=2000)
        >>> task_loader = DataLoader(task_dataset, batch_size=4, shuffle=True)
        >>> leo, losses = train_leo(
        ...     num_classes=5,
        ...     task_dataloader=task_loader,
        ...     latent_dim=64,
        ...     inner_lr=0.01,
        ...     outer_lr=0.001
        ... )
        >>> 
        >>> # Custom hyperparameters
        >>> leo, losses = train_leo(
        ...     num_classes=5,
        ...     task_dataloader=task_loader,
        ...     latent_dim=128,  # Larger latent space
        ...     inner_lr=0.05,   # Faster adaptation
        ...     inner_steps=3    # Fewer steps
        ... )
    
    Notes:
        - LEO is particularly effective for very few-shot scenarios (1-shot)
        - The latent space makes optimization more efficient than MAML
        - Training time: ~15-25 minutes for 2000 tasks on GPU
        - Memory efficient due to low-dimensional latent space
    
    Performance Tips:
        - Use GPU for 5-10x speedup
        - Increase batch_size (e.g., 8-16) if memory allows
        - Tune latent_dim based on task complexity
        - More inner_steps improves adaptation but increases compute
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize LEO
    leo = LatentEmbeddingOptimization(
        num_classes=num_classes,
        latent_dim=latent_dim,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs
    )
    
    # Move to device
    leo = leo.to(device)
    
    print(f"Starting LEO training...")
    print(f"Hyperparameters: latent_dim={latent_dim}, inner_lr={inner_lr}, "
          f"outer_lr={outer_lr}, inner_steps={inner_steps}")
    print(f"Optimizer: {optimizer_cls.__name__}")
    
    losses = []
    best_loss = float('inf')
    
    progress_bar = tqdm(task_dataloader, desc="Training", dynamic_ncols=True)
    
    for batch_idx, task_batch in enumerate(progress_bar):
        try:
            # Unpack task batch
            support_data_batch, support_labels_batch, query_data_batch, query_labels_batch = task_batch
            
            # Move to device
            support_data_batch = support_data_batch.to(device)
            support_labels_batch = support_labels_batch.to(device)
            query_data_batch = query_data_batch.to(device)
            query_labels_batch = query_labels_batch.to(device)
            
            # Meta-training step
            loss = leo.meta_train_step(
                support_data_batch,
                support_labels_batch,
                query_data_batch,
                query_labels_batch
            )
            
            losses.append(loss)
            
            # Update best loss
            if loss < best_loss:
                best_loss = loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'best': f'{best_loss:.4f}',
                'avg': f'{np.mean(losses[-100:]):.4f}'
            })
            
            # Periodic logging
            if (batch_idx + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"\nStep {batch_idx + 1}: Loss = {loss:.4f}, "
                      f"Avg Loss (last 100) = {avg_loss:.4f}, Best = {best_loss:.4f}")
        
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            continue
    
    print(f"\nTraining completed! Final loss: {np.mean(losses[-100:]):.4f}")
    
    return leo, losses
