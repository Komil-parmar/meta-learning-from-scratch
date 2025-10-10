"""
Original Meta Networks for Few-Shot Learning.

This module implements the original Meta Networks algorithm from the paper
"Meta Networks" by Munkhdalai & Yu (2017). Unlike the embedding-based variant,
this implementation follows the true Meta Networks approach where the meta-learner
learns to predict the actual weights and biases of the base network's classifier layer.

The key insight is that one neural network (meta-learner) learns to generate
the parameters of another neural network (base learner) for task-specific classification.

Classes:
    - MetaLearner: Generates FC layer weights and biases from support set
    - OriginalMetaNetwork: Complete system combining embedding network and meta-learner

Functions:
    - train_original_meta_network: Training pipeline for Original Meta Networks
    - evaluate_original_meta_network: Evaluation function compatible with utils.evaluate

Note:
    Uses shared EmbeddingNetwork from EmbeddingNetwork.py - the same base CNN
    architecture used across different meta-learning implementations in this repository.
    This ensures consistency and code reusability across meta-learning algorithms.

Reference:
    Munkhdalai & Yu, "Meta Networks", ICML 2017
    https://arxiv.org/abs/1703.00837

Note:
    This is the original Meta Networks (Model-based Meta Learning) approach,
    different from the embedding-based variant (Metric-based Meta Learning).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from algorithms.embedding_network import EmbeddingNetwork


class MetaLearner(nn.Module):
    """
    Original Meta-learner that predicts FC layer weights and biases.

    This is the true Meta Networks approach where one model learns to generate
    the parameters (weights and biases) of another model's classifier layer.

    The meta-learner uses three learnable components:
    - U: Matrix to project support embeddings (hidden_dim x embedding_dim)
    - V: Matrix to combine information (hidden_dim x embedding_dim)
    - e: Base embedding vector (hidden_dim)

    Algorithm:
        1. For each support example (x_i, y_i):
           - Compute embedding: h_i = embedding_network(x_i)
           - Project: r_i = tanh(U @ h_i + e)

        2. For each class c, aggregate support examples:
           - Compute class representation from all r_i where y_i = c

        3. Generate FC layer weights and biases:
           - Weight matrix W: [embedding_dim × num_classes]
           - Bias vector b: [num_classes]

        4. Classification: logits = query_embeddings @ W + b
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

        # Learnable parameters: U, V, and e (from original paper)
        self.U = nn.Parameter(torch.randn(hidden_dim, embedding_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_dim, embedding_dim) * 0.01)
        self.e = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # Networks to generate weights and biases from class representations
        # These predict the actual FC layer parameters
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)  # Generates one column of W per class
        )

        self.bias_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Generates one bias value per class
        )

    def forward(self, support_embeddings: torch.Tensor, support_labels: torch.Tensor,
                query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate FC layer weights/biases and compute predictions.

        Args:
            support_embeddings (torch.Tensor): Embeddings of support set [N*K, embedding_dim]
            support_labels (torch.Tensor): Labels of support set [N*K]
            query_embeddings (torch.Tensor): Embeddings of query set [N*Q, embedding_dim]

        Returns:
            torch.Tensor: Logits for query set [N*Q, num_classes]
        """
        # Step 1: Process support embeddings through U
        # r_i = tanh(U @ h_i + e) for each support embedding
        r = torch.tanh(torch.matmul(support_embeddings, self.U.t()) + self.e.unsqueeze(0))  # [N*K, hidden_dim]

        # Step 2: Compute class representations by averaging r_i per class
        class_representations = []
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                class_rep = r[mask].mean(dim=0)  # [hidden_dim]
            else:
                # If no examples for this class, use base embedding
                class_rep = self.e
            class_representations.append(class_rep)

        class_representations = torch.stack(class_representations)  # [num_classes, hidden_dim]

        # Step 3: Generate FC layer weights and biases from class representations
        # Weight matrix W: each class gets one column
        W_columns = []
        biases = []

        for c in range(self.num_classes):
            class_rep = class_representations[c]  # [hidden_dim]

            # Generate weight column for this class
            w_c = self.weight_generator(class_rep)  # [embedding_dim]
            W_columns.append(w_c)

            # Generate bias for this class
            b_c = self.bias_generator(class_rep)  # [1]
            biases.append(b_c)

        # Construct weight matrix W: [embedding_dim, num_classes]
        W = torch.stack(W_columns, dim=1)  # [embedding_dim, num_classes]

        # Construct bias vector b: [num_classes]
        b = torch.cat(biases, dim=0)  # [num_classes]

        # Step 4: Use predicted weights for classification
        # logits = query_embeddings @ W + b
        logits = torch.matmul(query_embeddings, W) + b.unsqueeze(0)  # [N*Q, num_classes]

        return logits


class OriginalMetaNetwork(nn.Module):
    """
    Complete Original Meta Network combining shared embedding network and meta-learner.

    Original Meta Networks algorithm where the meta-learner learns to predict
    the weights and biases of the base network's classifier layer.

    Architecture:
    1. Shared Embedding Network: Extracts features from raw images (from EmbeddingNetwork.py)
       - Same CNN architecture used across different meta-learning implementations
       - Includes Meta Dropout for consistent regularization
    2. Meta-Learner: Processes support set and generates:
       - Weight matrix W [embedding_dim × num_classes]
       - Bias vector b [num_classes]
    3. Classification: Uses predicted W and b to classify query examples

    This is fundamentally different from MAML and the embedding-based variant:
    - MAML: Learns good initialization + adapts via gradients
    - Embedding-based: Generates embeddings for metric-based classification
    - Original Meta Networks: Learns to directly generate task-specific parameters

    Note:
    Uses the same EmbeddingNetwork as EB_Meta_Network.py for consistency
    and code reusability across meta-learning algorithms.
    """

    def __init__(self, embedding_dim: int = 64, dropout_rates: list = [0.05, 0.1, 0.15, 0.05], hidden_dim: int = 128,
                 num_classes: int = 5):
        """
        Initialize Original Meta Network.

        Args:
            embedding_dim (int): Dimension of feature embeddings
            dropout_rates (list): Dropout rates for embedding network layers
                4 values for 4 conv blocks, e.g. [0.05, 0.10, 0.15, 0.05]
            hidden_dim (int): Hidden dimension for meta-learner
            num_classes (int): Number of classes per task (N-way)
        """
        super(OriginalMetaNetwork, self).__init__()
        # Use shared embedding network with Meta Dropout (shared across implementations)
        self.embedding_network = EmbeddingNetwork(embedding_dim, dropout_rates=dropout_rates)
        self.meta_learner = MetaLearner(embedding_dim, hidden_dim, num_classes)

    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embed data and generate predictions using predicted FC weights.

        Args:
            support_data (torch.Tensor): Support set images [N*K, C, H, W]
            support_labels (torch.Tensor): Support set labels [N*K]
            query_data (torch.Tensor): Query set images [N*Q, C, H, W]

        Returns:
            torch.Tensor: Logits for query set [N*Q, num_classes]
        """
        # Reset dropout masks for this task (same masks for support and query)
        self.embedding_network.reset_dropout_masks(support_data.shape, support_data.device)

        # Extract embeddings (no classification yet)
        support_embeddings = self.embedding_network(support_data)
        query_embeddings = self.embedding_network(query_data)

        # Meta-learner generates FC weights and biases, then classifies
        logits = self.meta_learner(support_embeddings, support_labels, query_embeddings)

        return logits


def train_original_meta_network(
    model: OriginalMetaNetwork,
    task_dataloader: torch.utils.data.DataLoader,
    learning_rate: float = 0.001,
    optimizer_cls: type = optim.Adam,
    optimizer_kwargs: dict = None
):
    """
    Train an Original Meta Network for few-shot learning.

    Original Meta Networks learn to generate task-specific parameters through a meta-learner
    that processes support set examples. The training optimizes the embedding network
    and meta-learner (U, V, e) jointly across multiple tasks.

    Algorithm:
        For each batch of tasks:
            1. Extract embeddings from support and query sets
            2. Meta-learner generates FC layer weights and biases from support embeddings
            3. Compute predictions on query set using predicted weights
            4. Backpropagate loss to update embedding network and U, V, e

    Args:
        model (OriginalMetaNetwork):
            The Original Meta Network model containing embedding network and meta-learner.
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
            - model (OriginalMetaNetwork): Trained model
            - optimizer: Optimizer with final state
            - losses (list[float]): Training loss history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, **optimizer_kwargs)

    print(f"Starting Original Meta Network training...")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: {optimizer_cls.__name__}")
    print(
        f"Meta-learner parameters: U ({model.meta_learner.U.shape}), V ({model.meta_learner.V.shape}), e ({model.meta_learner.e.shape})")

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


def evaluate_original_meta_network(
    model: OriginalMetaNetwork,
    eval_dataloader: torch.utils.data.DataLoader,
    num_classes: int = None,
    verbose: bool = True
):
    """
    Evaluate Original Meta Network performance on new unseen tasks.

    This function measures the model's ability to generate effective FC layer weights
    for new tasks. Unlike MAML, there's no adaptation step - the meta-learner
    directly generates task-specific parameters from the support set.

    Args:
        model (OriginalMetaNetwork):
            Trained Original Meta Network model. Should be on appropriate device.

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
        print(f"\n{'=' * 70}")
        print(f"ORIGINAL META NETWORK EVALUATION RESULTS")
        print(f"{'=' * 70}")
        print(f"Tasks Evaluated: {len(all_accuracies)}")
        print(f"Task Structure: {num_classes}-way classification")
        print(f"")
        print(f"Performance:")
        print(f"   Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Random Baseline: ~{random_baseline:.4f}")
        print(f"")
        print(f"Task Distribution:")
        print(
            f"   Tasks with >50% accuracy: {sum(1 for acc in all_accuracies if acc > 0.5)}/{len(all_accuracies)} ({sum(1 for acc in all_accuracies if acc > 0.5) / len(all_accuracies) * 100:.1f}%)")
        print(
            f"   Tasks with >80% accuracy: {sum(1 for acc in all_accuracies if acc > 0.8)}/{len(all_accuracies)} ({sum(1 for acc in all_accuracies if acc > 0.8) / len(all_accuracies) * 100:.1f}%)")
        print(
            f"   Tasks with >90% accuracy: {sum(1 for acc in all_accuracies if acc > 0.9)}/{len(all_accuracies)} ({sum(1 for acc in all_accuracies if acc > 0.9) / len(all_accuracies) * 100:.1f}%)")
        print(f"{'=' * 70}")

    # Return format compatible with utils.evaluate.plot_evaluation_results()
    return {
        'after_adaptation_accuracy': avg_accuracy,
        'after_adaptation_std': std_accuracy,
        'before_adaptation_accuracy': random_baseline,  # Original Meta Networks don't have "before" - use baseline
        'before_adaptation_std': 0.0,
        'all_accuracies': all_accuracies,
        'all_before_accuracies': [random_baseline] * len(all_accuracies),  # Baseline for all tasks
        'all_losses': all_losses,
        'num_tasks': len(all_accuracies),
        'random_baseline': random_baseline
    }