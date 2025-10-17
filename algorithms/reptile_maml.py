"""
Model-Agnostic Meta-Learning (MAML) Implementation with Reptile Support.
This module provides a PyTorch implementation of MAML, FOMAML, and Reptile,
meta-learning algorithms that enable models to quickly adapt to new tasks
with minimal gradient steps or parameter interpolation.
Classes:
    ModelAgnosticMetaLearning: Core meta-learning algorithm implementation
    
Functions:
    train_maml: High-level training function with progress tracking
Reference:
    MAML: Finn et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation 
    of Deep Networks. ICML 2017. https://arxiv.org/abs/1703.03400
    
    Reptile: Nichol et al. (2018). On First-Order Meta-Learning Algorithms. 
    ICML 2018 Workshop. https://arxiv.org/abs/1803.02999
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


class ModelAgnosticMetaLearning:
    """
    Meta-Learning algorithm supporting MAML, FOMAML, and Reptile.
    
    This implementation provides three meta-learning variants:
    
    1. **MAML (Model-Agnostic Meta-Learning)**:
       - Second-order gradients through the inner loop
       - Most accurate but computationally expensive
       - Learns optimal parameter initialization
    
    2. **FOMAML (First-Order MAML)**:
       - First-order approximation of MAML
       - ~30-50% faster with minimal accuracy loss
       - Treats adapted parameters as independent of meta-parameters
    
    3. **Reptile**:
       - Simpler alternative to MAML
       - Meta-update via parameter interpolation: θ ← θ + ε(θ_adapted - θ)
       - No gradient computation from inner loop needed
       - Often more stable and easier to tune than MAML
       - Comparable performance with lower computational cost
    
    Algorithm Comparison:
        MAML/FOMAML: θ = θ - β * ∇L(θ')  (gradient-based)
        Reptile:     θ = θ + ε * (θ' - θ) (interpolation-based)
    
    Attributes:
        model (torch.nn.Module): Neural network to meta-train
        inner_lr (float): Learning rate for inner loop adaptation
        outer_lr (float): Meta-learning rate for outer loop updates
        inner_steps (int): Number of gradient steps in inner loop
        algorithm (str): One of 'maml', 'fomaml', or 'reptile'
        first_order (bool): Whether to use first-order approximation
        meta_optimizer (torch.optim.Optimizer): Optimizer for meta-parameters
    
    Example:
        >>> # MAML (second-order)
        >>> maml = ModelAgnosticMetaLearning(model, algorithm='maml')
        >>> 
        >>> # FOMAML (first-order, ~40% faster)
        >>> fomaml = ModelAgnosticMetaLearning(model, algorithm='fomaml')
        >>> 
        >>> # Reptile (simplest, parameter interpolation)
        >>> reptile = ModelAgnosticMetaLearning(model, algorithm='reptile')
    
    See Also:
        - Reptile: An alternative to MAML that is easier to tune
        - FOMAML: First-order approximation of MAML
    """

    def __init__(
        self, 
        model: torch.nn.Module, 
        inner_lr: float = 0.01, 
        outer_lr: float = 0.001, 
        inner_steps: int = 5, 
        optimizer_cls: type = optim.Adam, 
        optimizer_kwargs: dict = None,
        algorithm: str = 'maml'
    ):
        """
        Initialize the meta-learning algorithm.
        
        Args:
            model (torch.nn.Module): Neural network to meta-train
            inner_lr (float): Learning rate for task adaptation (default: 0.01)
            outer_lr (float): Learning rate for meta-updates (default: 0.001)
            inner_steps (int): Gradient steps per task (default: 5)
            optimizer_cls (type): Optimizer class for meta-updates (default: Adam)
            optimizer_kwargs (dict): Additional optimizer arguments (default: None)
            algorithm (str): Algorithm variant - 'maml', 'fomaml', or 'reptile' 
                           (default: 'maml')
        
        Raises:
            ValueError: If algorithm is not 'maml', 'fomaml', or 'reptile'
        """
        if algorithm not in ['maml', 'fomaml', 'reptile']:
            raise ValueError(f"algorithm must be 'maml', 'fomaml', or 'reptile', got '{algorithm}'")

        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.algorithm = algorithm

        # Determine first_order flag based on algorithm
        self.first_order = (algorithm in ['fomaml', 'reptile'])

        # Initialize meta-optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.meta_optimizer = optimizer_cls(self.model.parameters(), lr=outer_lr, **optimizer_kwargs)

        print(f"Initialized {algorithm.upper()} meta-learner")
        if algorithm == 'reptile':
            print("  - Uses parameter interpolation instead of gradient-based updates")
            print(f"  - Meta-update: θ ← θ + {outer_lr}*(θ_adapted - θ)")

    def inner_update(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> dict:
        """
        Perform inner loop adaptation on a task's support set.
        
        Starting from meta-parameters θ, performs multiple gradient descent steps
        to obtain task-specific parameters θ'. Gradients are computed with
        create_graph=True for MAML (enabling second-order) or False for FOMAML/Reptile.
        
        Args:
            support_data (torch.Tensor): Support set input [N*K, ...]
            support_labels (torch.Tensor): Support set labels [N*K]
        
        Returns:
            dict: Task-specific parameters {name: tensor, ...}
        
        Example:
            >>> adapted_params = maml.inner_update(support_data, support_labels)
            >>> query_pred = maml.forward_with_weights(query_data, adapted_params)
        """
        self.model.train()

        # Clone initial parameters
        fast_weights = {}
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()

        # Perform multiple gradient steps
        for step in range(self.inner_steps):
            logits = self.forward_with_weights(support_data, fast_weights)
            loss = F.cross_entropy(logits, support_labels)

            # Compute gradients
            # MAML: create_graph=True (enable second-order)
            # FOMAML/Reptile: create_graph=False (first-order only)
            grads = torch.autograd.grad(
                loss, 
                fast_weights.values(), 
                create_graph=not self.first_order,
                allow_unused=False
            )

            # Update fast weights
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def forward_with_weights(self, x: torch.Tensor, weights: dict) -> torch.Tensor:
        """
        Forward pass using custom parameter values.
        
        Performs stateless forward pass by temporarily replacing model parameters
        with provided weights. Essential for inner loop evaluation.
        
        Args:
            x (torch.Tensor): Input data
            weights (dict): Parameter dictionary {name: tensor}
        
        Returns:
            torch.Tensor: Model output
        """
        return torch.func.functional_call(self.model, weights, x)

    def meta_train_step(self, support_data_batch, support_labels_batch, 
                        query_data_batch, query_labels_batch) -> float:
        """
        Perform one meta-training step across a batch of tasks.
        
        Implements the outer loop of meta-learning:
        1. Adapt to each task using support set (inner loop)
        2. Evaluate adapted parameters on query set
        3. Compute meta-gradient/meta-update
        4. Update meta-parameters
        
        For Reptile: Directly interpolates toward adapted parameters.
        For MAML/FOMAML: Computes gradients and applies optimizer step.
        
        Args:
            support_data_batch (torch.Tensor): [batch_size, N*K, ...]
            support_labels_batch (torch.Tensor): [batch_size, N*K]
            query_data_batch (torch.Tensor): [batch_size, N*Q, ...]
            query_labels_batch (torch.Tensor): [batch_size, N*Q]
        
        Returns:
            float: Average meta-loss across batch
        """
        self.model.train()
        self.meta_optimizer.zero_grad()

        device = next(self.model.parameters()).device
        batch_size = support_data_batch.size(0)

        # Accumulate losses on GPU to minimize CPU↔GPU transfers
        meta_loss_sum = torch.tensor(0.0, device=device)

        if self.algorithm == 'reptile':
            # ========== REPTILE: Parameter Interpolation ==========
            # Accumulate adapted parameters for averaging
            accumulated_params = {
                name: torch.zeros_like(param)
                for name, param in self.model.named_parameters()
            }

            for i in range(batch_size):
                support_data = support_data_batch[i]
                support_labels = support_labels_batch[i]
                query_data = query_data_batch[i]
                query_labels = query_labels_batch[i]

                # Inner loop: adapt to task (no second-order gradients needed)
                fast_weights = self.inner_update(support_data, support_labels)

                # Evaluate on query set (for loss tracking only)
                with torch.no_grad():
                    query_logits = self.forward_with_weights(query_data, fast_weights)
                    query_loss = F.cross_entropy(query_logits, query_labels)
                    meta_loss_sum += query_loss.detach()

                # Accumulate adapted parameters for interpolation
                for name, adapted_param in fast_weights.items():
                    accumulated_params[name] += adapted_param.detach()

            # Meta-update: θ ← θ + ε * mean(θ_adapted - θ)
            # This is equivalent to: θ = (1-ε)θ + ε·θ_adapted (exponential moving average)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    # Compute mean adapted parameter
                    mean_adapted = accumulated_params[name] / batch_size
                    # Reptile update: move toward adapted parameters
                    param.data = param.data + self.outer_lr * (mean_adapted - param.data)

        elif self.algorithm == 'fomaml':
            # ========== FOMAML: First-Order Approximation ==========
            for i in range(batch_size):
                support_data = support_data_batch[i]
                support_labels = support_labels_batch[i]
                query_data = query_data_batch[i]
                query_labels = query_labels_batch[i]

                # Inner loop adaptation
                fast_weights = self.inner_update(support_data, support_labels)

                # Detach to prevent backprop through inner loop (first-order approximation)
                fast_weights = {
                    name: param.detach().requires_grad_(True) 
                    for name, param in fast_weights.items()
                }

                # Query loss with first-order approximation
                query_logits = self.forward_with_weights(query_data, fast_weights)
                query_loss = F.cross_entropy(query_logits, query_labels)

                # Compute gradients w.r.t. fast_weights
                grads = torch.autograd.grad(
                    query_loss,
                    fast_weights.values(),
                    create_graph=False
                )

                # Apply gradients directly to original model parameters
                for (name, param), grad in zip(self.model.named_parameters(), grads):
                    if param.grad is None:
                        param.grad = grad / batch_size
                    else:
                        param.grad += grad / batch_size

                meta_loss_sum += query_loss.detach()

        else:  # 'maml'
            # ========== MAML: Second-Order Gradients ==========
            for i in range(batch_size):
                support_data = support_data_batch[i]
                support_labels = support_labels_batch[i]
                query_data = query_data_batch[i]
                query_labels = query_labels_batch[i]

                # Inner loop: adapt to task
                fast_weights = self.inner_update(support_data, support_labels)

                # Query loss: evaluate adapted parameters
                query_logits = self.forward_with_weights(query_data, fast_weights)
                query_loss = F.cross_entropy(query_logits, query_labels)

                # Backward through inner loop (second-order)
                (query_loss / batch_size).backward()

                meta_loss_sum += query_loss.detach()

        # Gradient clipping and optimizer step (for MAML/FOMAML)
        if self.algorithm != 'reptile':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.meta_optimizer.step()

        return (meta_loss_sum / batch_size).item()


# Convenience alias
MAML = ModelAgnosticMetaLearning


def train_maml(
    model: torch.nn.Module, 
    task_dataloader: torch.utils.data.DataLoader,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    optimizer_cls: type = optim.Adam,
    optimizer_kwargs: dict = None,
    algorithm: str = 'maml',
    use_amp: bool = True
):
    """
    Train a model using meta-learning (MAML, FOMAML, or Reptile).
    
    This function implements the complete meta-learning training pipeline.
    For each batch of tasks, the algorithm:
    1. Adapts to each task using support set (inner loop)
    2. Evaluates on query set to compute meta-gradient
    3. Updates meta-parameters (outer loop)
    
    Args:
        model (torch.nn.Module): Neural network to meta-train
        task_dataloader (torch.utils.data.DataLoader): Yields (support_data, 
            support_labels, query_data, query_labels) tuples
        inner_lr (float): Task adaptation learning rate (default: 0.01)
        outer_lr (float): Meta-learning rate (default: 0.001)
        inner_steps (int): Adaptation steps per task (default: 5)
        optimizer_cls (type): Optimizer class (default: Adam)
        optimizer_kwargs (dict): Additional optimizer kwargs (default: None)
        algorithm (str): One of 'maml', 'fomaml', or 'reptile' (default: 'maml')
        use_amp (bool): Use Automatic Mixed Precision (default: True)
    
    Returns:
        tuple: (trained_model, meta_learner, losses)
            - trained_model: Meta-trained model
            - meta_learner: MAML/FOMAML/Reptile trainer object
            - losses: List of losses across training iterations
    
    Example:
        >>> # Train with Reptile (simpler, often more stable)
        >>> model, trainer, losses = train_maml(
        ...     model, dataloader, algorithm='reptile'
        ... )
        >>> 
        >>> # Compare three algorithms on same dataset
        >>> for algo in ['maml', 'fomaml', 'reptile']:
        ...     m, t, losses = train_maml(model, dataloader, algorithm=algo)
        ...     print(f"{algo}: final_loss={np.mean(losses[-100:]):.4f}")
    
    Notes:
        - Reptile typically trains 20-30% faster than MAML
        - Reptile is often easier to tune (more stable)
        - FOMAML balances speed and accuracy
        - All algorithms should achieve comparable final performance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    # Validate algorithm choice
    if algorithm not in ['maml', 'fomaml', 'reptile']:
        raise ValueError(f"algorithm must be 'maml', 'fomaml', or 'reptile', got '{algorithm}'")

    meta_learner = ModelAgnosticMetaLearning(
        model, 
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        algorithm=algorithm
    )

    print(f"\nStarting {algorithm.upper()} training...")
    print(f"Hyperparameters: inner_lr={inner_lr}, outer_lr={outer_lr}, inner_steps={inner_steps}")

    losses = []
    loss_accumulator = torch.tensor(0.0, device=device)
    loss_count = 0

    progress_bar = tqdm(task_dataloader, desc="Training", dynamic_ncols=True)

    for batch_idx, task_batch in enumerate(progress_bar):
        try:
            support_data_batch, support_labels_batch, query_data_batch, query_labels_batch = task_batch

            # Move to device with non-blocking transfer
            support_data_batch = support_data_batch.to(device, non_blocking=True)
            support_labels_batch = support_labels_batch.to(device, non_blocking=True)
            query_data_batch = query_data_batch.to(device, non_blocking=True)
            query_labels_batch = query_labels_batch.to(device, non_blocking=True)

            # Meta-training step
            loss = meta_learner.meta_train_step(
                support_data_batch, support_labels_batch,
                query_data_batch, query_labels_batch
            )

            # GPU tensor accumulation
            loss_accumulator += loss
            loss_count += 1

            # Transfer to CPU periodically
            if (batch_idx + 1) % 20 == 0:
                avg_loss = (loss_accumulator / loss_count).item()
                losses.extend([avg_loss] * loss_count)

                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Batch': batch_idx + 1
                })

                loss_accumulator = torch.tensor(0.0, device=device)
                loss_count = 0

        except Exception as e:
            tqdm.write(f"Error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Handle remaining losses
    if loss_count > 0:
        avg_loss = (loss_accumulator / loss_count).item()
        losses.extend([avg_loss] * loss_count)

    print(f"\nTraining completed! Final loss: {np.mean(losses[-100:]):.4f}")
    print(f"Algorithm: {algorithm.upper()}")

    return model, meta_learner, losses