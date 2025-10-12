"""
ANIL: Almost No Inner Loop Implementation.

This module provides PyTorch implementations of ANIL, a meta-learning algorithm
that only adapts the head (final layer) during the inner loop while keeping the
body (feature extractor) fixed, making it much faster than MAML.

Classes:
    ANIL: Base ANIL implementation (supports both frozen and trainable body)

Functions:
    train_anil: High-level training function with progress tracking

Reference:
    Raghu, A., Raghu, M., Bengio, S., & Vinyals, O. (2020). 
    Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML.
    ICLR 2020.
    https://arxiv.org/abs/1909.09157
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional, Type


class ANIL:
    """
    ANIL: Almost No Inner Loop.

    ANIL simplifies MAML by only adapting the head (final layer) during inner loop
    adaptation, while keeping the body (feature extractor) fixed. This significantly
    reduces computational cost while maintaining comparable performance to MAML.
    
    This class handles multiple variants through the `freeze_body` parameter:
    - freeze_body=False: Original ANIL (body meta-learned, head adapted)
    - freeze_body=True: Frozen pretrained body (only head is trained)

    Key Insight:
        The ANIL paper shows that most of the adaptation in MAML happens in the head,
        and freezing the body during inner loop has minimal impact on performance while
        being much faster (3-10x speedup).
    
    Algorithm:
        ```
        Initialize θ = {θ_body, θ_head} (meta-parameters)
        while not converged:
            Sample batch of tasks τ ~ p(τ)
            for each task τᵢ in batch:
                # Inner loop: ONLY adapt head
                θ'_head = θ_head - α∇_{θ_head} L_τᵢ(θ_body, θ_head)
                
                # Outer loop: evaluate with adapted head
                Compute L_τᵢ(θ_body, θ'_head) on query set
            
            # Meta-update:
            if freeze_body:
                θ_head = θ_head - β∇_{θ_head} Σᵢ L_τᵢ(θ_body, θ'_head)
            else:
                θ = θ - β∇_θ Σᵢ L_τᵢ(θ_body, θ'_head)  # Update both
        ```
    
    Attributes:
        body (torch.nn.Module): Feature extractor network
        head (torch.nn.Module): Classifier/output network
        inner_lr (float): Learning rate for inner loop head adaptation
        outer_lr (float): Meta-learning rate for outer loop updates
        inner_steps (int): Number of gradient steps for head adaptation
        freeze_body (bool): Whether to freeze body during meta-learning
        first_order (bool): Whether to use first-order approximation
        meta_optimizer (torch.optim.Optimizer): Optimizer for meta-updates
    
    Example:
        >>> # Original ANIL: meta-learn both body and head
        >>> body = nn.Sequential(nn.Conv2d(...), nn.ReLU(), ...)
        >>> head = nn.Linear(128, 5)
        >>> anil = ANIL(body, head, inner_lr=0.01, outer_lr=0.001, freeze_body=False)
        >>>
        >>> # Frozen pretrained body: only meta-learn head
        >>> vgg = models.vgg11(pretrained=True)
        >>> body = vgg.features
        >>> head = nn.Linear(512 * 7 * 7, 5)
        >>> anil = ANIL(body, head, inner_lr=0.01, outer_lr=0.001, freeze_body=True)
        >>>
        >>> # Training loop
        >>> for task_batch in task_dataloader:
        ...     loss = anil.meta_train_step(*task_batch)
    
    Notes:
        - Body is frozen during inner loop (no gradients computed)
        - Head is adapted during inner loop
        - In outer loop: both updated (freeze_body=False) or only head (freeze_body=True)
        - Significantly faster than MAML (3-10x speedup)
        - Performance comparable to MAML on many benchmarks
    """
    
    def __init__(
        self,
        body: torch.nn.Module,
        head: torch.nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        freeze_body: bool = False,
        trainable_body_modules: Optional[list] = None,
        optimizer_cls: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        first_order: bool = False
    ):
        """
        Initialize the ANIL algorithm.
        
        Args:
            body (torch.nn.Module): Feature extractor network
            head (torch.nn.Module): Classifier/output network
            inner_lr (float): Learning rate for inner loop head adaptation (default: 0.01)
            outer_lr (float): Meta-learning rate for outer loop (default: 0.001)
            inner_steps (int): Number of gradient steps for head adaptation (default: 5)
            freeze_body (bool): Whether to freeze body during meta-learning (default: False)
                - False: Original ANIL (body updated in outer loop, frozen in inner loop)
                - True: Frozen pretrained body (never updated)
            trainable_body_modules (list): List of specific submodules to keep trainable
                even when freeze_body=True (e.g., adapter networks). Default: None
            optimizer_cls (type): Optimizer class for meta-learning (default: Adam)
            optimizer_kwargs (dict): Additional optimizer arguments (default: None)
            first_order (bool): Use first-order approximation (default: False)
        """
        self.body = body
        self.head = head
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.freeze_body = freeze_body
        self.first_order = first_order
        self.trainable_body_modules = trainable_body_modules or []
        
        # Configure which parameters to optimize
        if freeze_body:
            # Freeze body parameters EXCEPT BatchNorm and specified trainable modules
            bn_params = []
            trainable_module_params = []
            frozen_params = 0
            
            # Collect trainable module parameters
            trainable_module_set = set(self.trainable_body_modules)
            
            for name, module in self.body.named_modules():
                # Check if this module or any parent is in trainable list
                is_trainable_module = any(tm is module for tm in trainable_module_set)
                
                if is_trainable_module:
                    # Keep this entire module trainable
                    for param in module.parameters():
                        param.requires_grad = True
                        trainable_module_params.append(param)
                elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    # Keep BatchNorm parameters trainable
                    for param in module.parameters():
                        param.requires_grad = True
                        bn_params.append(param)
                else:
                    # Freeze all other parameters
                    for param in module.parameters():
                        if param.requires_grad:  # Only freeze if not already frozen
                            param.requires_grad = False
                            frozen_params += param.numel()

            # Remove duplicates (in case trainable modules contain BatchNorm)
            trainable_module_params_set = set(id(p) for p in trainable_module_params)
            bn_params = [p for p in bn_params if id(p) not in trainable_module_params_set]

            # Optimize head + BatchNorm + trainable modules
            params_to_optimize = list(self.head.parameters()) + bn_params + trainable_module_params

            body_params = sum(p.numel() for p in self.body.parameters())
            head_params = sum(p.numel() for p in self.head.parameters())
            bn_param_count = sum(p.numel() for p in bn_params)
            trainable_module_count = sum(p.numel() for p in trainable_module_params)
            
            print(f"ANIL Configuration: Frozen Body (with exceptions)")
            print(f"  Frozen body parameters: {frozen_params:,}")
            print(f"  Trainable BatchNorm parameters: {bn_param_count:,}")
            if trainable_module_count > 0:
                print(f"  Trainable module parameters: {trainable_module_count:,}")
            print(f"  Trainable head parameters: {head_params:,}")
            print(f"  Total trainable: {head_params + bn_param_count + trainable_module_count:,}")
        else:
            # Optimize both body and head
            params_to_optimize = list(self.body.parameters()) + list(self.head.parameters())

            body_params = sum(p.numel() for p in self.body.parameters())
            head_params = sum(p.numel() for p in self.head.parameters())
            print(f"ANIL Configuration: Trainable Body + Head")
            print(f"  Body parameters: {body_params:,}")
            print(f"  Head parameters: {head_params:,}")

        # Initialize meta-optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.meta_optimizer = optimizer_cls(params_to_optimize, lr=outer_lr, **optimizer_kwargs)
    
    def warmup_batchnorm(self, dataloader: torch.utils.data.DataLoader, num_batches: int = 50):
        if not self.freeze_body:
            print("⚠️ Warning: warmup_batchnorm called but body is not frozen!")
            return
        
        print(f"🔥 Warming up BatchNorm statistics with {num_batches} batches...")
        
        self.body.eval()
        set_batchnorm_training(self.body, training=True)  # Force BatchNorm to training mode
        
        with torch.no_grad():
            for i, task_batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                support_data, _, query_data, _ = task_batch
                device = next(self.body.parameters()).device
                
                all_data = torch.cat([
                    support_data.view(-1, *support_data.shape[2:]),
                    query_data.view(-1, *query_data.shape[2:])
                ], dim=0).to(device)
                
                _ = self.body(all_data)
        
        print("✅ BatchNorm warmup complete!")

    def extract_head_params(self) -> Dict[str, torch.Tensor]:
        """Extract head parameters as a dictionary with requires_grad=True."""
        head_params = {}
        for name, param in self.head.named_parameters():
            # Clone and ensure requires_grad is True for gradient computation
            head_params[name] = param.clone().requires_grad_(True)
        return head_params
    
    def forward_with_head(self, x: torch.Tensor, head_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using custom head parameters while keeping body fixed.
        
        Args:
            x (torch.Tensor): Input data
            head_params (dict): Dictionary of head parameters
        
        Returns:
            torch.Tensor: Model output
        """
        # Extract features using body
        # Body mode is already set by caller (train/eval)
        # During outer loop: body.train() - needs gradients for meta-update
        # During inner loop: body.eval() - frozen, but still compute gradients for meta-learning
        features = self.body(x)

        # If body is frozen, we need to detach features from the frozen conv layers
        # but keep gradients flowing for BatchNorm parameters
        # Check if any BatchNorm parameters are trainable
        # has_trainable_bn = any(
        #     isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) and 
        #     any(p.requires_grad for p in m.parameters())
        #     for m in self.body.modules()
        # )
        
        # if self.freeze_body and not has_trainable_bn:
        #     # Fully frozen body - detach features
        #     features = features.detach().requires_grad_(True)
        # If has_trainable_bn, don't detach - we need gradients to flow back to BatchNorm

        # Apply head with custom parameters
        output = torch.func.functional_call(self.head, head_params, features)
        return output
    
    def inner_update(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop adaptation - ONLY adapting the head.
        
        Args:
            support_data (torch.Tensor): Support set input data
            support_labels (torch.Tensor): Support set labels
        
        Returns:
            dict: Dictionary of adapted head parameters
        """
        self.head.train()

        # Start from current head parameters
        fast_head = self.extract_head_params()
        
        # Perform multiple gradient steps on head only
        for step in range(self.inner_steps):
            logits = self.forward_with_head(support_data, fast_head)
            loss = F.cross_entropy(logits, support_labels)
            
            # CRITICAL FIX: Compute gradients w.r.t. head parameters only
            # BUT maintain computation graph for body (create_graph=True for second-order)
            # This allows gradients to flow back to body during outer loop backward()
            grads = torch.autograd.grad(
                loss,
                fast_head.values(),
                create_graph=not self.first_order,  # True for second-order ANIL
                retain_graph=not self.first_order,  # Keep graph for body gradients
                allow_unused=False
            )
            
            # Update head parameters
            fast_head = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_head.items(), grads)
            }
        
        return fast_head
    
    def meta_train_step(
    self,
    support_data_batch: torch.Tensor,
    support_labels_batch: torch.Tensor,
    query_data_batch: torch.Tensor,
    query_labels_batch: torch.Tensor
    ) -> float:
        self.head.train()
        self.body.eval()  # Body in eval mode since frozen, but BatchNorm will be overridden
        set_batchnorm_training(self.body, training=True)  # Force BatchNorm to training mode
        
        self.meta_optimizer.zero_grad()
        
        device = next(self.head.parameters()).device
        meta_loss_sum = torch.tensor(0.0, device=device)
        batch_size = support_data_batch.size(0)
        
        if self.first_order:
            for i in range(batch_size):
                support_data = support_data_batch[i]
                support_labels = support_labels_batch[i]
                query_data = query_data_batch[i]
                query_labels = query_labels_batch[i]
                
                self.body.eval()
                set_batchnorm_training(self.body, training=True)
                
                # Inner loop: adapt head
                fast_head = {
                    name: param.detach().clone().requires_grad_(True)
                    for name, param in self.head.named_parameters()
                }
                
                for step in range(self.inner_steps):
                    # Get features (no gradients needed during inner loop)
                    with torch.no_grad():
                        features = self.body(support_data)
                    
                    # CRITICAL: Enable gradients on features for head adaptation
                    features = features.requires_grad_(True)
                    
                    logits = torch.func.functional_call(self.head, fast_head, features)
                    loss = F.cross_entropy(logits, support_labels)
                    grads = torch.autograd.grad(loss, fast_head.values(), create_graph=False)
                    
                    fast_head = {
                        name: param - self.inner_lr * grad
                        for (name, param), grad in zip(fast_head.items(), grads)
                    }
                
                # Outer loop: evaluate on query set
                self.body.eval()
                set_batchnorm_training(self.body, training=True)
                
                # CRITICAL FIX: Need different handling based on freeze_body
                if self.freeze_body:
                    # Body is frozen: compute features WITH gradients for BatchNorm
                    # but gradients won't flow to frozen conv layers (requires_grad=False)
                    # We need gradients enabled so BatchNorm params can get their gradients
                    features = self.body(query_data)
                else:
                    # Body is trainable: compute features WITH gradients
                    # Features must maintain connection to body parameters
                    features = self.body(query_data)
                
                # Ensure fast_head parameters have gradients enabled
                for param in fast_head.values():
                    param.requires_grad_(True)
                
                # Compute query loss
                logits = torch.func.functional_call(self.head, fast_head, features)
                query_loss = F.cross_entropy(logits, query_labels)
                
                # CRITICAL FIX: Different gradient computation based on freeze_body
                if self.freeze_body:
                    # Frozen body: Manually compute gradients for head + BatchNorm
                    trainable_body_params = []
                    for module in self.body.modules():
                        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                            trainable_body_params.extend([p for p in module.parameters() if p.requires_grad])
                    
                    # Also include trainable modules if any
                    for tm in self.trainable_body_modules:
                        trainable_body_params.extend([p for p in tm.parameters() if p.requires_grad])
                    
                    # Compute gradients for head parameters
                    grads_head = torch.autograd.grad(
                        query_loss,
                        fast_head.values(),
                        create_graph=False,
                        retain_graph=len(trainable_body_params) > 0,
                        allow_unused=True
                    )
                    
                    # Apply gradients to ORIGINAL head parameters
                    for (name, original_param), grad in zip(self.head.named_parameters(), grads_head):
                        if grad is not None:
                            if original_param.grad is None:
                                original_param.grad = grad.clone() / batch_size
                            else:
                                original_param.grad += grad / batch_size
                    
                    # Compute and apply gradients to trainable body parameters (BatchNorm)
                    if len(trainable_body_params) > 0:
                        grads_body = torch.autograd.grad(
                            query_loss,
                            trainable_body_params,
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=True
                        )
                        
                        for param, grad in zip(trainable_body_params, grads_body):
                            if grad is not None:
                                if param.grad is None:
                                    param.grad = grad.clone() / batch_size
                                else:
                                    param.grad += grad / batch_size
                else:
                    # Trainable body: Use standard backward() for ALL parameters
                    # This is the CORRECT approach for first-order ANIL with trainable body
                    # The gradients flow through: query_loss -> fast_head -> features -> body
                    
                    # First, compute gradients w.r.t. fast_head parameters
                    grads_head = torch.autograd.grad(
                        query_loss,
                        fast_head.values(),
                        create_graph=False,
                        retain_graph=True,  # Keep graph for body gradients
                        allow_unused=True
                    )
                    
                    # Apply gradients to ORIGINAL head parameters
                    for (name, original_param), grad in zip(self.head.named_parameters(), grads_head):
                        if grad is not None:
                            if original_param.grad is None:
                                original_param.grad = grad.clone() / batch_size
                            else:
                                original_param.grad += grad / batch_size
                    
                    # Now compute gradients w.r.t. body parameters
                    # These flow through the features that were computed WITH gradients
                    trainable_body_params = [p for p in self.body.parameters() if p.requires_grad]
                    
                    grads_body = torch.autograd.grad(
                        query_loss,
                        trainable_body_params,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True
                    )
                    
                    for param, grad in zip(trainable_body_params, grads_body):
                        if grad is not None:
                            if param.grad is None:
                                param.grad = grad.clone() / batch_size
                            else:
                                param.grad += grad / batch_size
                
                meta_loss_sum += query_loss.detach()
        else:
            for i in range(batch_size):
                support_data = support_data_batch[i]
                support_labels = support_labels_batch[i]
                query_data = query_data_batch[i]
                query_labels = query_labels_batch[i]
                
                self.body.eval()
                set_batchnorm_training(self.body, training=True)  # Ensure BatchNorm training mode
                fast_head = self.inner_update(support_data, support_labels)
                
                self.body.eval()
                set_batchnorm_training(self.body, training=True)
                query_logits = self.forward_with_head(query_data, fast_head)
                query_loss = F.cross_entropy(query_logits, query_labels)
                (query_loss / batch_size).backward()
                
                meta_loss_sum += query_loss.detach()
        
        # Clip gradients for all trainable parameters
        if self.freeze_body:
            # Clip head + BatchNorm parameters
            trainable_params = [p for p in self.head.parameters()]
            for module in self.body.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    trainable_params.extend(module.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        else:
            # Clip all parameters
            all_params = list(self.head.parameters()) + list(self.body.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return (meta_loss_sum / batch_size).item()


def train_anil(
    body: torch.nn.Module,
    head: torch.nn.Module,
    task_dataloader: torch.utils.data.DataLoader,
    freeze_body: bool = False,
    trainable_body_modules: Optional[list] = None,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    optimizer_cls: Type[optim.Optimizer] = optim.Adam,
    optimizer_kwargs: Optional[Dict] = None,
    first_order: bool = False,
    bn_warmup_batches: int = 0
):
    """
    Train a model using ANIL (Almost No Inner Loop).
    
    Args:
        body (torch.nn.Module): Feature extractor network
        head (torch.nn.Module): Classifier/output network
        task_dataloader (DataLoader): Dataloader yielding task batches
        freeze_body (bool): Freeze body completely
        trainable_body_modules (list): List of specific submodules to keep trainable
            even when freeze_body=True (e.g., adapter networks)
        inner_lr (float): Learning rate for inner loop adaptation
        outer_lr (float): Meta-learning rate for outer loop
        inner_steps (int): Number of gradient steps in inner loop
        optimizer_cls (type): Optimizer class
        optimizer_kwargs (dict): Additional optimizer arguments
        first_order (bool): Use first-order approximation
        bn_warmup_batches (int): Number of batches to warm up BatchNorm stats (for frozen body)
    
    Returns:
        tuple: (body, head, anil_trainer, losses)

    Example:
        >>> # Original ANIL (trainable body)
        >>> body, head, anil, losses = train_anil(body, head, dataloader)
        >>>
        >>> # Frozen pretrained body with BN warmup
        >>> body, head, anil, losses = train_anil(
        ...     body, head, dataloader, 
        ...     freeze_body=True,
        ...     bn_warmup_batches=50
        ... )
        >>>
        >>> # Adapter + frozen body
        >>> adapter_body = AdapterBody(adapter, resnet_body)
        >>> body, head, anil, losses = train_anil(
        ...     adapter_body, head, dataloader,
        ...     freeze_body=True,
        ...     trainable_body_modules=[adapter_body.adapter],  # Keep adapter trainable
        ...     bn_warmup_batches=50
        ... )
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    body = body.to(device)
    head = head.to(device)
    
    anil = ANIL(
        body, head,
        inner_lr=inner_lr, outer_lr=outer_lr, inner_steps=inner_steps,
        freeze_body=freeze_body,
        trainable_body_modules=trainable_body_modules,
        optimizer_cls=optimizer_cls, optimizer_kwargs=optimizer_kwargs,
        first_order=first_order
    )
    if freeze_body:
        if trainable_body_modules:
            variant_name = "ANIL (Frozen Body with Trainable Modules)"
        else:
            variant_name = "ANIL (Frozen Pretrained Body)"
    else:
        variant_name = "ANIL (Original)"
    
    print(f"\nStarting {variant_name} training...")
    print(f"Hyperparameters: inner_lr={inner_lr}, outer_lr={outer_lr}, inner_steps={inner_steps}")
    print(f"Optimizer: {optimizer_cls.__name__}")
    if first_order:
        print("Using First-Order approximation")
    
    # Set BatchNorm to training mode
    set_batchnorm_training(body, training=True)
    
    if freeze_body and bn_warmup_batches > 0:
        anil.warmup_batchnorm(task_dataloader, num_batches=bn_warmup_batches)
    
    losses = []
    best_loss = float('inf')
    
    loss_accumulator = torch.tensor(0.0, device=device)
    loss_count = 0
    
    progress_bar = tqdm(task_dataloader, desc="Training", dynamic_ncols=True)
    
    for batch_idx, task_batch in enumerate(progress_bar):
        try:
            support_data_batch, support_labels_batch, query_data_batch, query_labels_batch = task_batch
            
            support_data_batch = support_data_batch.to(device, non_blocking=True)
            support_labels_batch = support_labels_batch.to(device, non_blocking=True)
            query_data_batch = query_data_batch.to(device, non_blocking=True)
            query_labels_batch = query_labels_batch.to(device, non_blocking=True)
            
            loss = anil.meta_train_step(
                support_data_batch, support_labels_batch,
                query_data_batch, query_labels_batch
            )
            
            loss_accumulator += loss
            loss_count += 1
            
            avg_loss = (loss_accumulator / loss_count).item()
            losses.extend([avg_loss] * loss_count)
            
            progress_info = {
                'Loss': f'{avg_loss:.4f}',
                'Batch': batch_idx + 1
            }
            
            if torch.cuda.is_available():
                progress_info['GPU%'] = f'{torch.cuda.utilization()}'
            
            progress_bar.set_postfix(progress_info)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            loss_accumulator = torch.tensor(0.0, device=device)
            loss_count = 0
                
        except Exception as e:
            tqdm.write(f"Error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if loss_count > 0:
        avg_loss = (loss_accumulator / loss_count).item()
        losses.extend([avg_loss] * loss_count)
    
    print(f"\nTraining completed! Final loss: {np.mean(losses[-100:]):.4f}")
    print(f"Best loss: {best_loss:.4f}")
    
    return body, head, anil, losses


def set_batchnorm_training(body: torch.nn.Module, training: bool = True):
    """
    Set all BatchNorm layers in the body to training mode with track_running_stats=True.
    
    Args:
        body (torch.nn.Module): The feature extractor network (e.g., VGG).
        training (bool): Whether to set BatchNorm layers to training mode (default: True).
    """
    for module in body.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.training = training
            module.track_running_stats = True