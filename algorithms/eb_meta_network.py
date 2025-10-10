"""
Embedding-based Meta Networks for Few-Shot Learning.

This module implements Meta Networks, which learn to generate task-specific parameters
(fast weights) through a meta-learner network. Unlike MAML which optimizes for good
initialization, Meta Networks directly produce classifier parameters from support set
embeddings.

This implementation uses Meta Dropout to maintain consistent dropout masks across
support and query sets within the same task, ensuring the meta-learner learns from
consistent regularization patterns.

Classes:
    - EmbeddingNetwork: Base CNN that extracts feature embeddings from images
    - MetaLearner: Generates fast weights (U, V, e) for task-specific classification
    - MetaNetwork: Complete system combining embedding network and meta-learner

Functions:
    - train_meta_network: Training pipeline for Meta Networks
    - evaluate_meta_network: Evaluation function compatible with utils.evaluate

Reference:
    Munkhdalai & Yu, "Meta Networks", ICML 2017
    https://arxiv.org/abs/1703.00837
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from algorithms.meta_dropout import MetaDropout


class EmbeddingNetwork(nn.Module):
    """
    Base CNN that extracts feature embeddings from images.
    
    Architecture: 4 convolutional blocks with max pooling, followed by a fully
    connected layer. Each block consists of Conv2d -> BatchNorm -> ReLU -> MaxPool.
    
    Uses Meta Dropout for consistent regularization across support and query sets
    within the same task.
    
    Input: 105×105 grayscale images
    Output: Fixed-dimensional feature embeddings
    """
    
    def __init__(self, embedding_dim: int = 64, dropout_rates: list = None):
        """
        Initialize the Embedding Network.
        
        Args:
            embedding_dim (int): Dimension of output embeddings. Default: 64
            dropout_rates (list, optional): Dropout rates for each layer [p1, p2, p3].
                                           Default: [0.05, 0.10, 0.15]
        """
        super(EmbeddingNetwork, self).__init__()
        
        if dropout_rates is None:
            dropout_rates = [0.05, 0.10, 0.15]
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # Output embedding: 64 channels * 6 * 6 spatial dimensions after 4 pooling layers
        self.fc = nn.Linear(64 * 6 * 6, embedding_dim)

        # Use Meta Dropout for consistent masks across support and query
        self.dropout1 = MetaDropout(p=dropout_rates[0])
        self.dropout2 = MetaDropout(p=dropout_rates[1])
        self.dropout3 = MetaDropout(p=dropout_rates[2])
        
        # Track if masks need reset
        self._masks_initialized = False

        self.force_eval = False  # If True, remove dropout even in train mode

    def reset_dropout_masks(self, input_shape: torch.Size, device: torch.device):
        """
        Reset dropout masks for a new task based on input shape.
        
        Args:
            input_shape: Shape of the input tensor [batch_size, C, H, W]
            device: Device to create masks on
        """
        # Calculate shapes at dropout application points (BEFORE pooling)
        # After conv1 (before pool): [B, 64, 105, 105]
        shape1 = (1, 64, 105, 105)
        self.dropout1.reset_mask(shape1, device)
        
        # After conv2 (before pool): [B, 64, 52, 52]  (52 = 105//2)
        shape2 = (1, 64, 52, 52)
        self.dropout2.reset_mask(shape2, device)
        
        # After conv3 (before pool): [B, 64, 26, 26]  (26 = 52//2)
        shape3 = (1, 64, 26, 26)
        self.dropout3.reset_mask(shape3, device)
        
        self._masks_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding network.
        
        Args:
            x (torch.Tensor): Input images [batch_size, 1, 105, 105]
        
        Returns:
            torch.Tensor: Feature embeddings [batch_size, embedding_dim]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x) if not self.training or self.force_eval else x
        x = self.pool(x)  # 52x52
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x) if not self.training or self.force_eval else x
        x = self.pool(x)  # 26x26
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x) if not self.training or self.force_eval else x
        x = self.pool(x)  # 13x13

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 6x6
        
        x = self.flatten(x)
        
        x = self.fc(x)
        
        return x


class MetaLearner(nn.Module):
    """
    Meta-learner that generates fast weights for task-specific adaptation.
    
    The meta-learner learns to produce classifier parameters from support set examples.
    It uses three learnable components:
    - U: Matrix to project support embeddings (embedding_dim x hidden_dim)
    - V: Matrix to project query embeddings (embedding_dim x hidden_dim)
    - e: Base embedding vector (hidden_dim)
    
    Algorithm:
        For each support example (x_i, y_i):
            1. Compute embedding: h_i = embedding_network(x_i)
            2. Project: r_i = U @ h_i
            3. Combine with base: w_i = r_i + e
        
        Fast weights for class c:
            W_c = mean(w_i for all i where y_i = c)
        
        Classification for query x:
            h = embedding_network(x)
            logits_c = (V @ h)^T @ W_c
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int):
        """
        Initialize the Meta-Learner.
        
        Args:
            embedding_dim (int): Dimension of embeddings from base network
            hidden_dim (int): Hidden dimension for meta-learner
            num_classes (int): Number of classes (N-way)
        """
        super(MetaLearner, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Learnable parameters: U, V, and e
        self.U = nn.Parameter(torch.randn(hidden_dim, embedding_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_dim, embedding_dim) * 0.01)
        self.e = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
    def forward(self, support_embeddings: torch.Tensor, support_labels: torch.Tensor,
                query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate fast weights and compute predictions.
        
        Args:
            support_embeddings (torch.Tensor): Embeddings of support set [N*K, embedding_dim]
            support_labels (torch.Tensor): Labels of support set [N*K]
            query_embeddings (torch.Tensor): Embeddings of query set [N*Q, embedding_dim]
        
        Returns:
            torch.Tensor: Logits for query set [N*Q, num_classes]
        """
        # Generate fast weights from support set
        # r_i = U @ h_i for each support embedding
        r = torch.matmul(support_embeddings, self.U.t())  # [N*K, hidden_dim]
        
        # w_i = r_i + e
        w = r + self.e.unsqueeze(0)  # [N*K, hidden_dim]
        
        # Compute class prototypes (average fast weights per class)
        class_weights = []
        for c in range(self.num_classes):
            mask = (support_labels == c)  # 0, 1, 2...
            if mask.sum() > 0:
                class_weight = w[mask].mean(dim=0)  # [hidden_dim]
            else:
                # If no examples for this class, use base embedding
                class_weight = self.e
            class_weights.append(class_weight)
        
        class_weights = torch.stack(class_weights)  # [num_classes, hidden_dim]
        
        # Compute query projections: V @ h for each query
        query_proj = torch.matmul(query_embeddings, self.V.t())  # [N*Q, hidden_dim]
        
        # Compute logits: (V @ h)^T @ W_c for each class c
        logits = torch.matmul(query_proj, class_weights.t())  # [N*Q, num_classes]
        
        return logits


class MetaNetwork(nn.Module):
    """
    Complete Meta Network combining embedding network and meta-learner.
    
    Meta Networks learn to generate task-specific parameters through a meta-learner
    that processes support set examples. The system consists of:
    1. Embedding Network: Extracts features from raw images
    2. Meta-Learner: Generates fast weights (U, V, e) for classification
    
    Unlike MAML which optimizes initialization, Meta Networks directly learn
    to produce task-specific parameters from the support set.
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, num_classes: int = 5, dropout_rates: list = None):
        """
        Initialize Meta Network.
        
        Args:
            embedding_dim (int): Dimension of feature embeddings. Default: 64
            hidden_dim (int): Hidden dimension for meta-learner. Default: 128
            num_classes (int): Number of classes per task (N-way). Default: 5
            dropout_rates (list, optional): Dropout rates for embedding network.
                                           Default: [0.05, 0.10, 0.15]
        """
        super(MetaNetwork, self).__init__()
        self.embedding_network = EmbeddingNetwork(embedding_dim, dropout_rates)
        self.meta_learner = MetaLearner(embedding_dim, hidden_dim, num_classes)
        
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embed data and generate predictions using fast weights.
        
        Args:
            support_data (torch.Tensor): Support set images [N*K, C, H, W]
            support_labels (torch.Tensor): Support set labels [N*K]
            query_data (torch.Tensor): Query set images [N*Q, C, H, W]
        
        Returns:
            torch.Tensor: Logits for query set [N*Q, num_classes]
        """
        # Reset dropout masks for this task (same masks for support and query)
        self.embedding_network.reset_dropout_masks(support_data.shape, support_data.device)
        
        # Extract embeddings with shared dropout masks
        support_embeddings = self.embedding_network(support_data)
        query_embeddings = self.embedding_network(query_data)
        
        # Generate fast weights and predict
        logits = self.meta_learner(support_embeddings, support_labels, query_embeddings)
        
        return logits


def train_meta_network(
    model: MetaNetwork,
    task_dataloader: torch.utils.data.DataLoader,
    learning_rate: float = 0.001,
    optimizer_cls: type = optim.Adam,
    optimizer_kwargs: dict = None
):
    """
    Train a Meta Network for few-shot learning.
    
    Meta Networks learn to generate task-specific parameters through a meta-learner
    that processes support set examples. The training optimizes the embedding network
    and meta-learner (U, V, e) jointly across multiple tasks.
    
    Uses Meta Dropout: dropout masks are automatically reset per task and shared
    across support and query sets within the same task, ensuring consistent
    regularization when generating fast weights.
    
    Algorithm:
        For each batch of tasks:
            1. Reset dropout masks (automatic in forward pass)
            2. Extract embeddings from support and query sets (same masks)
            3. Meta-learner generates fast weights from support embeddings
            4. Compute predictions on query set using fast weights
            5. Backpropagate loss to update embedding network and U, V, e
    
    Args:
        model (MetaNetwork):
            The Meta Network model containing embedding network and meta-learner.
            Will be moved to appropriate device (CPU/GPU) automatically.
            
        task_dataloader (torch.utils.data.DataLoader):
            DataLoader yielding batches of tasks. Each task should be:
            (support_data, support_labels, query_data, query_labels)
            - support_data: [batch_size, N*K, C, H, W]
            - support_labels: [batch_size, N*K]
            - query_data: [batch_size, N*Q, C, H, W]
            - query_labels: [batch_size, N*Q]
        
        learning_rate (float, optional):
            Learning rate for meta-learning updates. Controls how quickly the
            meta-learner parameters (U, V, e) and embedding network are updated.
            Default: 0.001
            Typical range: 0.0001-0.01
        
        optimizer_cls (type, optional):
            Optimizer class for training. Default: torch.optim.Adam
            Common alternatives: torch.optim.SGD, torch.optim.AdamW
        
        optimizer_kwargs (dict, optional):
            Additional optimizer arguments.
            Default: None
            Example: {'weight_decay': 1e-5}
    
    Returns:
        tuple: (model, optimizer, losses)
            - model (MetaNetwork): Trained model
            - optimizer: Optimizer with final state
            - losses (list[float]): Training loss history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, **optimizer_kwargs)
    
    print(f"Starting Meta Network training...")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: {optimizer_cls.__name__}")
    print(f"Meta-learner parameters: U ({model.meta_learner.U.shape}), V ({model.meta_learner.V.shape}), e ({model.meta_learner.e.shape})")
    
    losses = []
    best_loss = float('inf')
    
    model.train()
    progress_bar = tqdm(task_dataloader, desc="Training", dynamic_ncols=True)
    
    for batch_idx, task_batch in enumerate(progress_bar):
        try:
            batch_loss = 0.0
            optimizer.zero_grad()
            
            # Process each task in batch
            for support_data, support_labels, query_data, query_labels in zip(*task_batch):
                support_data = support_data.to(device, non_blocking=True)
                support_labels = support_labels.to(device, non_blocking=True)
                query_data = query_data.to(device, non_blocking=True)
                query_labels = query_labels.to(device, non_blocking=True)
                
                # Forward pass
                logits = model(support_data, support_labels, query_data)
                loss = F.cross_entropy(logits, query_labels)
                
                # Backward pass (accumulate gradients)
                (loss / len(task_batch[0])).backward()
                
                batch_loss += loss.item()
            
            # Update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            avg_loss = batch_loss / len(task_batch[0])
            losses.append(avg_loss)
            
            # Update progress bar
            if len(losses) >= 20:
                recent_loss = np.mean(losses[-20:])
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Avg': f'{recent_loss:.4f}',
                    'Best': f'{best_loss:.4f}'
                })
                
                if recent_loss < best_loss:
                    best_loss = recent_loss
            
            # Log periodically
            if (batch_idx + 1) % 100 == 0:
                avg_loss_100 = np.mean(losses[-100:])
                tqdm.write(f"Step {batch_idx + 1}: Loss={avg_loss_100:.4f}")
                
        except Exception as e:
            tqdm.write(f"Error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTraining completed! Final loss: {np.mean(losses[-100:]):.4f}")
    return model, optimizer, losses


def evaluate_meta_network(
    model: MetaNetwork,
    eval_dataloader: torch.utils.data.DataLoader,
    num_classes: int = None,
    verbose: bool = True
):
    """
    Evaluate Meta Network performance on new unseen tasks.
    
    This function measures the model's ability to generate effective fast weights
    for new tasks. Unlike MAML, there's no adaptation step - the meta-learner
    directly generates task-specific parameters from the support set.
    
    Uses Meta Dropout: dropout masks are automatically reset per task in the
    forward pass, ensuring consistent evaluation behavior matching training.
    
    Args:
        model (MetaNetwork):
            Trained Meta Network model. Should be on appropriate device.
            
        eval_dataloader (torch.utils.data.DataLoader):
            DataLoader yielding evaluation tasks with batch size 1.
            Each task: (support_data, support_labels, query_data, query_labels)
            
        num_classes (int, optional):
            Number of classes per task. If None, inferred from labels.
            Default: None
            
        verbose (bool, optional):
            Whether to print detailed results.
            Default: True
    
    Returns:
        dict: Evaluation metrics including accuracy, loss, and per-task statistics.
              Compatible with utils.evaluate.plot_evaluation_results()
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_accuracies = []
    all_losses = []
    
    if num_classes is None:
        num_classes = None
    
    for support_data, support_labels, query_data, query_labels in tqdm(
        eval_dataloader, desc="Evaluating", disable=not verbose
    ):
        support_data = support_data.squeeze(0).to(device)
        support_labels = support_labels.squeeze(0).to(device)
        query_data = query_data.squeeze(0).to(device)
        query_labels = query_labels.squeeze(0).to(device)
        
        if num_classes is None:
            num_classes = len(torch.unique(query_labels))
        
        # Forward pass (no adaptation needed!)
        with torch.no_grad():
            logits = model(support_data, support_labels, query_data)
            loss = F.cross_entropy(logits, query_labels)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
            
            all_accuracies.append(accuracy)
            all_losses.append(loss.item())
    
    if len(all_accuracies) == 0:
        raise ValueError("Evaluation dataloader is empty")
    
    # Calculate metrics
    avg_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    avg_loss = np.mean(all_losses)
    random_baseline = 1.0 / num_classes if num_classes else 0.2
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"META NETWORK EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Tasks Evaluated: {len(all_accuracies)}")
        print(f"Task Structure: {num_classes}-way classification")
        print(f"")
        print(f"Performance:")
        print(f"   Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Random Baseline: ~{random_baseline:.4f}")
        print(f"")
        print(f"Task Distribution:")
        print(f"   Tasks with >50% accuracy: {sum(1 for acc in all_accuracies if acc > 0.5)}/{len(all_accuracies)} ({sum(1 for acc in all_accuracies if acc > 0.5)/len(all_accuracies)*100:.1f}%)")
        print(f"   Tasks with >80% accuracy: {sum(1 for acc in all_accuracies if acc > 0.8)}/{len(all_accuracies)} ({sum(1 for acc in all_accuracies if acc > 0.8)/len(all_accuracies)*100:.1f}%)")
        print(f"   Tasks with >90% accuracy: {sum(1 for acc in all_accuracies if acc > 0.9)}/{len(all_accuracies)} ({sum(1 for acc in all_accuracies if acc > 0.9)/len(all_accuracies)*100:.1f}%)")
        print(f"{'='*70}")
    
    # Return format compatible with utils.evaluate.plot_evaluation_results()
    return {
        'after_adaptation_accuracy': avg_accuracy,
        'after_adaptation_std': std_accuracy,
        'before_adaptation_accuracy': random_baseline,  # Meta Networks don't have "before" - use baseline
        'before_adaptation_std': 0.0,
        'all_accuracies': all_accuracies,
        'all_before_accuracies': [random_baseline] * len(all_accuracies),  # Baseline for all tasks
        'all_losses': all_losses,
        'num_tasks': len(all_accuracies),
        'random_baseline': random_baseline
    }