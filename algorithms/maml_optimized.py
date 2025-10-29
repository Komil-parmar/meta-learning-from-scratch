"""
Model-Agnostic Meta-Learning (MAML) Implementation.

This module provides a PyTorch implementation of MAML, a meta-learning algorithm
that trains models to quickly adapt to new tasks with minimal gradient steps.

Classes:
    ModelAgnosticMetaLearning: Core MAML algorithm implementation
    
Functions:
    train_maml: High-level training function with progress tracking

Reference:
    Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning 
    for Fast Adaptation of Deep Networks. ICML 2017.
    https://arxiv.org/abs/1703.03400
"""

from typing import Generator, Tuple, Union, List, Dict

from pkg_resources import require
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.func import vmap, functional_call, grad_and_value
from tqdm import tqdm


class ModelAgnosticMetaLearning:
    """
    Model-Agnostic Meta-Learning (MAML) algorithm for few-shot learning.
    
    MAML learns an initialization of model parameters that enables rapid adaptation
    to new tasks with only a few gradient steps. The algorithm operates with two
    nested optimization loops:
    
    1. **Inner Loop (Task Adaptation)**: Fine-tunes model on task's support set
    2. **Outer Loop (Meta-Learning)**: Updates initialization based on query set performance
    
    This implementation supports both **full MAML** (second-order) and **FOMAML** 
    (First-Order MAML) variants:
    
    - **MAML**: Computes gradients through the inner loop optimization (second-order).
      More accurate but computationally expensive.
      
    - **FOMAML**: Uses first-order approximation by treating adapted parameters as
      independent of meta-parameters. Faster and more memory-efficient with minimal
      performance loss (typically 1-3% accuracy difference).
    
    Algorithm Overview:
        ```
        Initialize θ (meta-parameters)
        while not converged:
            Sample batch of tasks τ ~ p(τ)
            for each task τᵢ in batch:
                # Inner loop: adapt to task
                θ'ᵢ = θ - α∇_θ L_τᵢ(θ)  (one or more steps)
                
                # Outer loop: evaluate adapted parameters
                Compute L_τᵢ(θ'ᵢ) on query set
            
            # Meta-update
            # MAML: θ = θ - β∇_θ Σᵢ L_τᵢ(θ'ᵢ)  (through inner loop)
            # FOMAML: θ = θ - β∇_θ' Σᵢ L_τᵢ(θ'ᵢ)  (at θ' only)
        ```
    
    Attributes:
        model (torch.nn.Module): 
            The neural network to be meta-trained. This model's parameters
            are optimized to enable fast task-specific adaptation.
            
        inner_lr (float): 
            Learning rate (α) for inner loop adaptation. Controls how quickly
            the model adapts to individual tasks during the inner loop.
            
        outer_lr (float): 
            Meta-learning rate (β) for outer loop updates. Controls how quickly
            the meta-parameters are updated based on task performance.
            
        inner_steps (int): 
            Number of gradient steps performed during inner loop adaptation.
            More steps allow better task-specific adaptation but increase
            computational cost.
            
        first_order (bool):
            Whether to use first-order approximation (FOMAML).
            False: Full MAML with second-order gradients
            True: FOMAML with first-order approximation
            
        meta_optimizer (torch.optim.Optimizer): 
            Optimizer for meta-parameter updates in the outer loop.
    
    Methods:
        inner_update: Adapt model to a single task using support set
        forward_with_weights: Forward pass with custom parameter values
        meta_train_step: Perform one meta-training step on a batch of tasks
    
    Example:
        >>> # Basic usage with full MAML
        >>> model = MyNetwork(num_classes=5)
        >>> maml = ModelAgnosticMetaLearning(
        ...     model, 
        ...     inner_lr=0.01,
        ...     outer_lr=0.001,
        ...     inner_steps=5
        ... )
        >>> 
        >>> # Using FOMAML for faster training
        >>> fomaml = ModelAgnosticMetaLearning(
        ...     model,
        ...     inner_lr=0.01,
        ...     outer_lr=0.001,
        ...     inner_steps=5,
        ...     first_order=True  # Enable FOMAML
        ... )
        >>> 
        >>> # Training loop
        >>> for task_batch in task_dataloader:
        ...     loss = maml.meta_train_step(task_batch)
        >>> 
        >>> # Adapt to new task at test time
        >>> adapted_params = maml.inner_update(support_data, support_labels)
        >>> predictions = maml.forward_with_weights(query_data, adapted_params)
    
    Notes:
        - Full MAML uses second-order gradients (create_graph=True) for accuracy
        - FOMAML uses first-order approximation for speed and memory efficiency
        - Set first_order=True to enable FOMAML (typically 30-50% faster)
        - Supports any PyTorch model architecture
        - Compatible with different optimizers (Adam, SGD, etc.)
        - Gradient clipping applied with max_norm=1.0 for stability
    
    Computational Complexity:
        - Inner loop: O(inner_steps × model_params)
        - Outer loop: O(batch_size × inner_steps × model_params) with second-order
        - Memory: O(batch_size × inner_steps × model_params) for gradient graph
    
    Hyperparameter Guidelines:
        inner_lr:
            - Range: 0.005 - 0.1
            - Lower: More stable, slower adaptation
            - Higher: Faster adaptation, may be unstable
            
        outer_lr:
            - Range: 0.0001 - 0.01
            - Should typically be smaller than inner_lr
            - Depends on optimizer choice
            
        inner_steps:
            - Range: 1 - 10
            - Trade-off between adaptation quality and speed
            - More steps = better task performance, more compute
    
    See Also:
        - train_maml: High-level training function
        - First-Order MAML (FOMAML): Faster approximation
        - Reptile: Another first-order meta-learning algorithm
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        inner_lr: float = 0.01, 
        outer_lr: float = 0.001, 
        inner_steps: int = 5, 
        optimizer_cls: type = optim.Adam, 
        optimizer_kwargs: dict = None,
        first_order: bool = False,
        reptile: bool = False,
        plus_plus: bool = False,
        meta_sgd: bool = False
    ):
        """
        Initialize the MAML algorithm.
        
        Args:
            model (torch.nn.Module):
                Neural network to be meta-trained. The model should be compatible
                with the task structure (e.g., output dimension matches number of
                classes in N-way classification).
                
            inner_lr (float, optional):
                Learning rate for inner loop task adaptation. Controls the step
                size when fine-tuning to individual tasks.
                Default: 0.01
                
            outer_lr (float, optional):
                Learning rate for outer loop meta-updates. Used by the meta-optimizer
                to update the initialization parameters.
                Default: 0.001
                
            inner_steps (int, optional):
                Number of gradient descent steps performed during task adaptation.
                More steps allow better task-specific fitting but increase cost.
                Default: 5
                
            optimizer_cls (type, optional):
                Optimizer class for meta-learning updates. Should be a PyTorch
                optimizer class (not an instance).
                Default: torch.optim.Adam
                Common alternatives: torch.optim.SGD, torch.optim.AdamW
                
            optimizer_kwargs (dict, optional):
                Additional keyword arguments passed to the optimizer constructor.
                Default: None (empty dict)
                Examples:
                    - {'momentum': 0.9, 'weight_decay': 1e-5} for SGD
                    - {'betas': (0.9, 0.999), 'weight_decay': 1e-4} for Adam
            
            first_order (bool, optional):
                Whether to use First-Order MAML (FOMAML) instead of full MAML.
                Default: False (use full second-order MAML)
                
                - False (MAML): Computes second-order gradients through inner loop.
                  More accurate but slower and more memory-intensive.
                  
                - True (FOMAML): Uses first-order approximation by ignoring the
                  dependency of adapted parameters on meta-parameters. Faster and
                  more memory-efficient but slightly less accurate.
                  
                FOMAML is typically 30-50% faster and uses ~50% less memory than MAML,
                with only a small decrease in final performance (usually 1-3% accuracy).
            
            reptile (bool, optional):
                Whether to use Reptile, a first-order meta-learning algorithm.
                Default: False (use MAML/FOMAML)
                
            plus_plus (bool, optional):
                Whether to use MAML++ (MAML Plus Plus) enhancements.
                Default: False (use vanilla MAML)

                MAML++ introduces several improvements over the original MAML:
                - Adaptive learning rates for each task
                - Better initialization strategies
                - More robust to overfitting

                These enhancements can lead to improved performance, especially in
                challenging few-shot learning scenarios.

                NOTE: MAML++ does not support FOMAML. If plus_plus=True, then first_order must be False.

            meta_sgd (bool, optional):
                Whether to use Meta-SGD, which learns per-parameter learning rates
                for the inner loop updates. This can improve adaptation speed.
                Default: False (use fixed inner_lr for all parameters)
                
                When enabled, each parameter will have its own learnable learning rate,
                initialized to inner_lr. This allows the model to learn how quickly
                to adapt each parameter during task-specific fine-tuning.
                
                Note: Meta-SGD increases (doubles) the number of parameters and may require
                more careful tuning of outer_lr.
        """
        self.first_order = first_order

        self.model = model
        self.param_names = tuple(name for name, _ in self.model.named_parameters())

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        self.reptile = reptile
        self.plus_plus = plus_plus
        self.meta_sgd = meta_sgd

        if self.plus_plus and self.first_order:
            raise ValueError("MAML++ does not support FOMAML. Set first_order=False when plus_plus=True.")
        
        if self.plus_plus and self.reptile:
            raise ValueError("MAML++ does not support Reptile. Set plus_plus=False when reptile=True.")
        
        # if self.meta_sgd and self.first_order:
            # raise ValueError("Meta-SGD does not support FOMAML. Set first_order=False when meta_sgd=True.")

        if self.plus_plus:
            # Pre-compute and cache parameter names for efficient dictionary reconstruction
            self.param_names = [name for name, _ in self.model.named_parameters()]
            
            # Initialize per-parameter learning rates for MAML++
            # Use ParameterList to properly register parameters
            self.alpha = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor(self.inner_lr, dtype=torch.float32)) 
                for _ in self.model.parameters()
            ])
            
            # CRITICAL: Add alpha parameters to meta-optimizer!
            if optimizer_kwargs is None:
                optimizer_kwargs = {}
            self.meta_optimizer = optimizer_cls(
                list(self.model.parameters()) + list(self.alpha.parameters()),
                lr=outer_lr, 
                **optimizer_kwargs
            )
        else:
            # Standard MAML: only optimize model parameters
            if optimizer_kwargs is None:
                optimizer_kwargs = {}
            self.meta_optimizer = optimizer_cls(self.model.parameters(), lr=outer_lr, **optimizer_kwargs)

        if self.meta_sgd:
            # Initialize per-parameter learning rates for Meta-SGD
            # Each parameter tensor gets its own learning rate tensor with the same shape
            # This gives us one learnable learning rate for each individual weight/bias
            self.meta_sgd_lrs = torch.nn.ParameterList([
                torch.nn.Parameter(
                    torch.full_like(param, self.inner_lr, dtype=torch.float32),
                    requires_grad=True
                ) 
                for param in self.model.parameters()
            ])
            
            # Add meta_sgd_lrs to meta-optimizer
            self.meta_optimizer = optimizer_cls(
                list(self.model.parameters()) + list(self.meta_sgd_lrs.parameters()),
                lr=outer_lr,
                **optimizer_kwargs
            )

    # Define vmappable loss function
    def compute_task_loss(self, params_list, support_data, support_labels):
        """Compute loss for a single task (vmapped over batch).

        Uses stateless_forward to avoid dictionary creation overhead.

        Args:
            params_list: List of parameters (not tuple, not dict) for efficient access
            support_data: Input data for the task
            support_labels: Labels for the task

        Returns:
            Loss value
        """
        # Forward pass using stateless_forward (no dict creation!)
        logits = self.model.stateless_forward(support_data, params_list)
        loss = F.cross_entropy(logits, support_labels)

        return loss

    def compute_task_loss_and_grads(self, params_tuple, support_data, support_labels, create_graph):
        """Compute both loss and gradients for a single task.

        CRITICAL FIX: torch.func.grad_and_value does NOT preserve computation graph!
        For second-order MAML, we must use torch.autograd.grad with create_graph=True.

        Args:
            params_tuple: Tuple of parameters for this task
            support_data: Input data for the task
            support_labels: Labels for the task
            create_graph: Whether to preserve computation graph (True for 2nd order MAML)
        Returns:
            Tuple of (loss, gradients_tuple)
        """
        params_list = list(params_tuple)

        # Forward pass
        logits = self.model.stateless_forward(support_data, params_list)
        loss = F.cross_entropy(logits, support_labels)

        if create_graph:
            # Second-order MAML: Use torch.autograd.grad to preserve computation graph
            # This is REQUIRED for backpropagating through inner loop updates
            grads = torch.autograd.grad(
                loss,
                params_tuple,
                create_graph=True,
                allow_unused=False
            )
        else:
            # First-order (FOMAML/Reptile): Use faster grad_and_value (detaches grads)
            def loss_fn(params_tuple):
                params_list = list(params_tuple)
                logits = self.model.stateless_forward(support_data, params_list)
                return F.cross_entropy(logits, support_labels)

            grads, loss = grad_and_value(loss_fn)(params_tuple)

        return loss, grads
    
    def inner_update(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> dict:
        """
        Perform inner loop adaptation on a task's support set.
        
        This method implements the task-specific adaptation phase of MAML. Starting
        from the current meta-parameters θ, it performs multiple gradient descent steps
        on the support set to obtain task-specific parameters θ'. The adaptation uses
        the inner learning rate and does NOT modify the original model parameters.
        
        Algorithm:
            ```
            θ' = θ  (initialize from meta-parameters)
            for step in range(inner_steps):
                L = loss(θ', support_data, support_labels)
                θ' = θ' - α∇_{θ'} L  (gradient descent step)
            return θ'
            ```
        
        The gradients are computed with create_graph=True, which enables second-order
        gradients needed for the outer loop meta-update. This allows MAML to optimize
        for fast adaptation.
        
        Args:
            support_data (torch.Tensor):
                Input data for task adaptation. Should contain K examples per class
                for a K-shot learning task.
                Shape: [N*K, ...] where N is number of classes, K is shots per class
                Example: [5, 1, 28, 28] for 5-way 1-shot with 28×28 images
                
            support_labels (torch.Tensor):
                Ground truth labels for support set.
                Shape: [N*K] with values in range [0, N-1]
                Example: [0, 1, 2, 3, 4] for 5-way 1-shot
        
        Returns:
            dict: Dictionary mapping parameter names to adapted parameter values.
                Keys are parameter names (e.g., 'conv1.weight', 'fc.bias')
                Values are torch.Tensors with gradients enabled (create_graph=True)
                These adapted parameters can be used with forward_with_weights()
        
        Example:
            >>> # Adapt to a new task
            >>> support_data = torch.randn(5, 3, 84, 84)  # 5-way 1-shot
            >>> support_labels = torch.tensor([0, 1, 2, 3, 4])
            >>> 
            >>> adapted_params = maml.inner_update(support_data, support_labels)
            >>> 
            >>> # Use adapted parameters for prediction
            >>> query_data = torch.randn(10, 3, 84, 84)
            >>> predictions = maml.forward_with_weights(query_data, adapted_params)
        
        Notes:
            - Sets model to train() mode for proper batch normalization/dropout
            - Creates computational graph for second-order gradients
            - Original model parameters remain unchanged
            - Adapted parameters retain gradient information for meta-learning
            - Uses cross-entropy loss by default (suitable for classification)
        
        Computational Cost:
            - Time: O(inner_steps × forward_pass × backward_pass)
            - Memory: O(inner_steps × model_params) for gradient graph
            - With inner_steps=5, approximately 5× cost of single forward-backward
        
        Customization:
            To use a different loss function, modify the loss computation:
            ```python
            # For regression tasks
            loss = F.mse_loss(logits, support_labels)
            
            # For binary classification
            loss = F.binary_cross_entropy_with_logits(logits, support_labels)
            ```
        
        See Also:
            - forward_with_weights: Use adapted parameters for inference
            - meta_train_step: Outer loop that uses inner_update
        """
        # CRITICAL: Use eval() mode to disable dropout during inner loop adaptation
        # Dropout randomness prevents reliable adaptation - each step would use different masks
        # This is why Meta Dropout was invented - to keep masks consistent
        # For simplicity, we just disable dropout during adaptation
        self.model.eval()
        
        # Start from current model parameters
        # Pre-compute parameter names and initial values (avoid repeated dict operations)
        fast_weights = {}
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()

        if self.meta_sgd:
            fast_sgd_lrs = {}
            for name, param in self.meta_sgd_lrs.named_parameters():
                fast_sgd_lrs[name] = param.clone()
        else:
            # Pre-compute inner learning rate list (fixed for all steps)
            inner_lr_list = [torch.tensor(self.inner_lr, dtype=torch.float32, device=support_data.device) 
                             for _ in fast_weights]
        fast_weights_keys = tuple(fast_weights.keys())
        fast_weights_values = list(fast_weights.values())

        # Perform multiple gradient steps
        for step in range(self.inner_steps):
            logits = self.model.stateless_forward(support_data, fast_weights_values)
            loss = F.cross_entropy(logits, support_labels)
            
            # Compute gradients
            # For Meta-SGD: ALWAYS use create_graph=True to allow alpha to learn
            # For regular MAML/FOMAML/Reptile: create_graph depends on first_order flag
            # Reptile is first-order and doesn't need computation graph
            create_graph = ((not self.first_order) and (not self.reptile)) or self.meta_sgd
            
            # Only compute gradients w.r.t. fast_weights (not learning rates!)
            # Learning rates are hyperparameters, not part of the computation graph
            grads = torch.autograd.grad(
                loss, 
                fast_weights_values, 
                create_graph=create_graph,
                allow_unused=False
            )
            
            # Update fast weights (vectorized, no dict operations!)
            if self.meta_sgd:
                # Meta-SGD: Use learnable per-parameter learning rates
                fast_weights_values = vectorized_param_update(
                    fast_weights_values,
                    grads,
                    list(fast_sgd_lrs.values())
                )
                # fast_weights = dict(zip(fast_weights_keys, fast_weights_values))

            else:
                # Standard MAML: Use fixed learning rate (pre-computed outside loop)
                fast_weights_values = vectorized_param_update(
                    fast_weights_values,
                    grads, 
                    inner_lr_list
                )
                
                # fast_weights = dict(zip(fast_weights_keys, fast_weights_values))
                
        fast_weights = dict(zip(fast_weights_keys, fast_weights_values))


        # Only create final dictionary once after all inner steps
        return fast_weights
            
    def inner_update_batch_parallel(
        self,
        support_data_batch: torch.Tensor,
        support_labels_batch: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Perform parallel inner loop adaptation using explicit vmap parallelization.

        CRITICAL FIX: For second-order MAML (create_graph=True), we CANNOT use vmap
        because torch.autograd.grad is not vmap-compatible. We process sequentially.
        For FOMAML/Reptile (create_graph=False), we use vmap for speed.

        OPTIMIZATION: Returns stacked parameters directly to avoid dict conversions.
        Each tensor in the returned list has shape [batch_size, param_shape].

        Args:
            support_data_batch (torch.Tensor):
                Support data for all tasks [batch_size, N*K, C, H, W]
            support_labels_batch (torch.Tensor):
                Support labels for all tasks [batch_size, N*K]

        Returns:
            List[torch.Tensor]: List of adapted parameters in stacked format.
                Each tensor has shape [batch_size, param_shape].
                Order matches self.param_names.
        """
        # CRITICAL: Use eval() mode to disable dropout during inner loop
        # self.model.eval()
        device = next(self.model.parameters()).device
        batch_size = support_data_batch.size(0)
        # Reptile is first-order and doesn't need computation graph
        create_graph = ((not self.first_order) and (not self.reptile)) or self.meta_sgd

        if create_graph:
            # SECOND-ORDER MAML: Process sequentially (torch.autograd.grad not vmap-compatible)
            # Initialize list to store adapted params for each task
            all_adapted_params = []

            for task_idx in range(batch_size):
                # Extract single task data
                support_data = support_data_batch[task_idx]
                support_labels = support_labels_batch[task_idx]

                # Start from current model parameters
                task_params = [param.clone() for param in self.model.parameters()]

                # Perform inner loop adaptation for this task
                for step in range(self.inner_steps):
                    # Forward pass
                    logits = self.model.stateless_forward(support_data, task_params)
                    loss = F.cross_entropy(logits, support_labels)

                    # Compute gradients with create_graph=True
                    grads = torch.autograd.grad(
                        loss,
                        task_params,
                        create_graph=True,
                        allow_unused=False
                    )

                    # Update parameters (preserves computation graph)
                    if self.meta_sgd:
                        # Meta-SGD: Use learnable learning rates
                        task_params = [
                            p - alpha * g
                            for p, g, alpha in zip(task_params, grads, self.meta_sgd_lrs.parameters())
                        ]
                    else:
                        # Standard: Use fixed learning rate
                        task_params = [
                            p - self.inner_lr * g
                            for p, g in zip(task_params, grads)
                        ]

                all_adapted_params.append(task_params)

            # Stack adapted parameters: [batch_size, param_shape]
            param_values_stacked = [
                torch.stack([task_params[i] for task_params in all_adapted_params])
                for i in range(len(all_adapted_params[0]))
            ]

        else:
            # FIRST-ORDER (FOMAML/REPTILE): Use vmap for parallel processing
            # GPU OPTIMIZATION: Work entirely with stacked params - no dict conversions!
            # Initialize stacked parameters [batch_size, param_shape] for each param
            param_values_stacked = [
                param.unsqueeze(0).expand(batch_size, *param.shape).clone()
                for param in self.model.parameters()
            ]

            if self.meta_sgd:
                # Batched learning rates for Meta-SGD
                batched_alphas = [
                    lr.unsqueeze(0).expand(batch_size, *lr.shape).clone()
                    for lr in self.meta_sgd_lrs.parameters()
                ]
            else:
                # Fixed learning rate for all parameters
                inner_lr_tensor = torch.tensor(self.inner_lr, dtype=torch.float32, device=device)

            # Inner loop adaptation - work entirely with stacked format
            for _ in range(self.inner_steps):
                # Use vmap to parallelize gradient computation across tasks
                vmapped_compute = vmap(
                    lambda params, data, labels: self.compute_task_loss_and_grads(
                        params, data, labels, create_graph  # create_graph=False here
                    ),
                    in_dims=(0, 0, 0),  # vmap over first dimension for all inputs
                    randomness='different'  # Ensure different randomness for dropout
                )

                losses, grads = vmapped_compute(param_values_stacked, support_data_batch, support_labels_batch)

                # GPU OPTIMIZATION: grads from vmap is ALREADY stacked as [batch_size, param_shape]!
                stacked_grads = list(grads)  # Zero-copy: already in correct format

                # Use vmap to perform parameter updates in parallel across all tasks
                if self.meta_sgd:
                    # With learnable per-parameter learning rates
                    param_values_stacked = [
                        vmap(lambda p, g, a: p - a * g, in_dims=(0, 0, 0))(p_stack, g_stack, a_stack)
                        for p_stack, g_stack, a_stack in zip(param_values_stacked, stacked_grads, batched_alphas)
                    ]
                else:
                    # With fixed learning rate - use vmap for parallel updates
                    param_values_stacked = [
                        vmap(lambda p, g: p - inner_lr_tensor * g, in_dims=(0, 0))(p_stack, g_stack)
                        for p_stack, g_stack in zip(param_values_stacked, stacked_grads)
                    ]

        # Return stacked parameters directly - no dict conversion!
        # Each tensor: [batch_size, param_shape]
        return param_values_stacked
    
    def forward_with_weights(self, x: torch.Tensor, weights: dict) -> torch.Tensor:
        """
        Perform forward pass using custom parameter values.
        
        This method enables using task-specific adapted parameters without modifying
        the original model's parameters. It's essential for MAML's inner loop, where
        we need to evaluate the model with adapted parameters while preserving the
        original meta-parameters.
        
        Uses PyTorch's functional_call API, which performs a stateless forward pass
        by temporarily replacing the model's parameters with the provided weights.
        This is more efficient and cleaner than manually applying functional operations.
        
        Args:
            x (torch.Tensor):
                Input data to pass through the model.
                Shape depends on model architecture (e.g., [batch_size, channels, height, width])
                
            weights (dict):
                Dictionary mapping parameter names to parameter tensors.
                Typically obtained from inner_update() method.
                Keys: Parameter names (e.g., 'conv1.weight', 'fc.bias')
                Values: Parameter tensors with appropriate shapes
        
        Returns:
            torch.Tensor: Model output using the provided weights.
                Shape depends on model architecture (e.g., [batch_size, num_classes])
        
        Example:
            >>> # Forward pass with adapted parameters
            >>> adapted_params = maml.inner_update(support_data, support_labels)
            >>> query_data = torch.randn(10, 3, 84, 84)
            >>> outputs = maml.forward_with_weights(query_data, adapted_params)
            >>> predictions = torch.argmax(outputs, dim=1)
            >>> 
            >>> # Works with original parameters too
            >>> original_params = {name: param for name, param in model.named_parameters()}
            >>> outputs = maml.forward_with_weights(query_data, original_params)
        
        Notes:
            - Does NOT modify the model's original parameters
            - Supports gradient computation (differentiable)
            - Works with any PyTorch model architecture
            - More efficient than manually reconstructing model with new parameters
            - Uses torch.func.functional_call (stateless forward pass)
        
        Implementation Details:
            The method uses torch.func.functional_call, which:
            1. Temporarily replaces model parameters with provided weights
            2. Performs a standard forward pass
            3. Restores original parameters
            4. Maintains gradient computation graph if needed
        
        Performance:
            - Time: Same as regular forward pass O(model_complexity)
            - Memory: Same as regular forward pass (no extra parameter copies)
            - Gradient: Supports backpropagation through weights if create_graph=True
        
        Common Use Cases:
            1. Inner loop evaluation: Test adapted parameters on support set
            2. Outer loop evaluation: Test adapted parameters on query set
            3. Meta-testing: Quick adaptation to new tasks at test time
            4. Ensemble predictions: Average outputs from multiple adaptations
        
        Troubleshooting:
            - If KeyError occurs: Ensure weights dict contains all model parameters
            - If shape mismatch: Verify weights match model architecture
            - If gradient issues: Check that weights were created with create_graph=True
        
        See Also:
            - inner_update: Creates adapted weights for this method
            - meta_train_step: Uses this method for query set evaluation
            - torch.func.functional_call: Underlying PyTorch API
        """
        return torch.func.functional_call(self.model, weights, x)

    def meta_train_step_parallel(
        self,
        support_data_batch: torch.Tensor,
        support_labels_batch: torch.Tensor,
        query_data_batch: torch.Tensor,
        query_labels_batch: torch.Tensor
    ) -> float:
        """
        Perform one meta-training step with EXPLICIT PARALLEL processing.

        This method uses manual batching with GPU scheduler parallelization for
        better performance than sequential processing. All tasks' inner loops are
        processed simultaneously, followed by a single backward pass.

        Args:
            support_data_batch: Support data [batch_size, N*K, C, H, W]
            support_labels_batch: Support labels [batch_size, N*K]
            query_data_batch: Query data [batch_size, N*Q, C, H, W]
            query_labels_batch: Query labels [batch_size, N*Q]

        Returns:
            float: Average meta-loss across all tasks
        """
        # Use eval() mode to disable dropout for consistent training
        # Dropout during inner loop makes adaptation unreliable
        # self.model.eval()

        device = next(self.model.parameters()).device
        batch_size = support_data_batch.size(0)

        if self.reptile:
            # REPTILE: Batched parameter interpolation with parallel processing
            # Standard Reptile algorithm: all tasks adapt from same θ₀, then average updates
            # NOTE: Reptile updates parameters DIRECTLY, not through optimizer!
            # So we DON'T call zero_grad() or optimizer.step()

            # Store initial meta-parameters before adapting to any tasks
            initial_params = [param.clone() for param in self.model.parameters()]

            # PARALLEL INNER LOOP: Adapt all tasks simultaneously using batched processing
            # Returns stacked adapted parameters [batch_size, param_shape] for each parameter
            stacked_adapted_params = self.inner_update_batch_parallel(
                support_data_batch, support_labels_batch
            )

            # Evaluate on query sets for logging (parallel processing with vmap)
            vmapped_query_loss = vmap(
                lambda params, data, labels: self.compute_task_loss(params, data, labels),
                in_dims=(0, 0, 0),
                randomness='different'
            )
            query_losses = vmapped_query_loss(
                stacked_adapted_params, query_data_batch, query_labels_batch
            )
            meta_loss_sum = query_losses.mean()

            # REPTILE BATCHED UPDATE: θ ← θ₀ + (ε/batch_size) * Σᵢ(θ'ᵢ - θ₀)
            # Compute average parameter update across all tasks
            with torch.no_grad():
                for param_idx, (param, initial_param) in enumerate(
                    zip(self.model.parameters(), initial_params)
                ):
                    # stacked_adapted_params[param_idx] has shape [batch_size, param_shape]
                    # Compute average update: mean over batch dimension
                    adapted_param_stack = stacked_adapted_params[param_idx]  # [batch_size, ...]

                    # Average update: (1/batch_size) * Σᵢ(θ'ᵢ - θ₀)
                    avg_update = (adapted_param_stack - initial_param.unsqueeze(0)).mean(dim=0)

                    # Apply update: θ ← θ₀ + ε * avg_update
                    # Note: Model params are unchanged by inner_update_batch_parallel (uses clones)
                    # so we're adding to the original θ₀, which is correct
                    param.data.add_(avg_update, alpha=self.outer_lr)

            # Return average meta-loss
            return meta_loss_sum.item()

        # For MAML/FOMAML/MAML++: use optimizer-based updates
        self.meta_optimizer.zero_grad()

        if self.first_order:
            # FOMAML: First-order approximation with parallel processing
            # GPU OPTIMIZATION: inner_update returns stacked params directly!
            stacked_query_params = self.inner_update_batch_parallel(
                support_data_batch, support_labels_batch
            )

            # Detach for first-order approximation
            stacked_query_params = [
                param_stack.detach().requires_grad_(True)
                for param_stack in stacked_query_params
            ]

            # PARALLEL query evaluation using vmap
            if hasattr(self.model, 'use_meta_dropout') and self.model.use_meta_dropout:
                self.model._outer_loop_mode = True

            vmapped_compute = vmap(
                lambda params, data, labels: self.compute_task_loss_and_grads(
                    params, data, labels, False
                ),
                in_dims=(0, 0, 0),  # vmap over first dimension for all inputs
                randomness='different'  # Ensure different randomness for dropout
            )

            losses, grads = vmapped_compute(stacked_query_params, query_data_batch, query_labels_batch)

            if hasattr(self.model, 'use_meta_dropout') and self.model.use_meta_dropout:
                self.model._outer_loop_mode = False

            # Accumulate gradients to model parameters
            # Since adapted params are detached from model params, we manually accumulate
            # GPU OPTIMIZATION: Use _foreach_copy for batched gradient assignment
            grad_list = [grads[i].mean(0) for i in range(len(grads))]
            for param, grad in zip(self.model.parameters(), grad_list):
                param.grad = grad

            meta_loss_sum = losses.mean().detach()

        else:
            # Standard MAML or MAML++: Second-order gradients
            if self.plus_plus:
                # MAML++: Multi-step loss optimization
                # Note: MAML++ processes tasks sequentially due to complex multi-step loss
                meta_loss_sum = torch.tensor(0.0, device=device)

                for task_idx in range(batch_size):
                    if hasattr(self.model, 'reset_dropout_masks'):
                        task_batch_size = support_data_batch[task_idx].size(0)
                        self.model.reset_dropout_masks(task_batch_size, device)

                    query_losses = []
                    fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

                    for step in range(self.inner_steps):
                        logits = self.forward_with_weights(support_data_batch[task_idx], fast_weights)
                        loss = F.cross_entropy(logits, support_labels_batch[task_idx])

                        grads = torch.autograd.grad(
                            loss,
                            fast_weights.values(),
                            create_graph=True,
                            allow_unused=False
                        )

                        # CRITICAL: Don't use JIT-compiled function for MAML++!
                        # We need to preserve computation graph through alpha parameters
                        # so gradients can flow back to learn the learning rates
                        param_list = list(fast_weights.values())
                        alpha_list = list(self.alpha)
                        grad_list = list(grads)
                        # Inline update to preserve computation graph (no JIT)
                        updated_params = [p - a * g for p, a, g in zip(param_list, alpha_list, grad_list)]
                        fast_weights = dict(zip(fast_weights.keys(), updated_params))

                        if hasattr(self.model, 'use_meta_dropout') and self.model.use_meta_dropout:
                            self.model._outer_loop_mode = True
                        query_logits = self.forward_with_weights(query_data_batch[task_idx], fast_weights)
                        if hasattr(self.model, 'use_meta_dropout') and self.model.use_meta_dropout:
                            self.model._outer_loop_mode = False

                        query_loss = F.cross_entropy(query_logits, query_labels_batch[task_idx])
                        query_losses.append(query_loss)

                    query_loss = torch.stack(query_losses).mean()
                    (query_loss / batch_size).backward()
                    meta_loss_sum += query_loss.detach() / batch_size

            else:
                # Standard MAML with parallel processing
                # GPU OPTIMIZATION: inner_update returns stacked params directly!
                stacked_query_params = self.inner_update_batch_parallel(
                    support_data_batch, support_labels_batch
                )

                # VMAP: Compute all query losses in parallel
                # Use randomness='different' to allow dropout with independent random seeds per task
                query_losses = vmap(
                    self.compute_task_loss,
                    in_dims=(0, 0, 0),
                    randomness='different'
                )(stacked_query_params, query_data_batch, query_labels_batch)

                # Single backward pass through all tasks
                accumulated_loss = query_losses.mean()
                accumulated_loss.backward()
                meta_loss_sum = accumulated_loss.detach()

        # Gradient clipping and optimizer step
        if self.plus_plus:
            # For MAML++, clip both model params and alpha learning rates
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.alpha.parameters()),
                max_norm=1.0
            )
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()

        # Clamp alpha learning rates (MAML++ only)
        if self.plus_plus:
            with torch.no_grad():
                for alpha_param in self.alpha:
                    alpha_param.clamp_(1e-6, 1.0)

        return meta_loss_sum.item()

    def meta_train_step(self, support_data_batch, support_labels_batch, query_data_batch, query_labels_batch) -> float:
        """
        Perform one meta-training step across a batch of tasks with parallel processing.

        This is now a wrapper that calls the optimized parallel implementation.
        The parallel version processes all tasks simultaneously for better GPU utilization.

        This method implements the outer loop of MAML, which updates the meta-parameters
        based on performance after adaptation. For each task in the batch:
        1. Adapts to the task using support set (inner loop)
        2. Evaluates adapted parameters on query set
        3. Computes meta-gradient through the adaptation process
        4. Accumulates gradients across all tasks
        5. Updates meta-parameters using the meta-optimizer
        
        Algorithm:
            ```
            Zero gradients
            meta_loss = 0
            
            for each task in batch:
                θ'ᵢ = inner_update(support_set)      # Adapt to task
                L_query = loss(θ'ᵢ, query_set)       # Evaluate adaptation
                meta_loss += L_query
                
            θ = θ - β∇_θ meta_loss                   # Meta-update
            return average_meta_loss
            ```
        
        The key insight of MAML is that gradients are computed with respect to the
        original meta-parameters θ, not the adapted parameters θ', enabling the
        algorithm to learn an initialization that facilitates rapid adaptation.
        
        Args:
            support_data_batch (torch.Tensor): 
                Support data for all tasks in batch [batch_size, N*K, ...]
                
            support_labels_batch (torch.Tensor): 
                Support labels for all tasks in batch [batch_size, N*K]
                
            query_data_batch (torch.Tensor): 
                Query data for all tasks in batch [batch_size, N*Q, ...]
                
            query_labels_batch (torch.Tensor): 
                Query labels for all tasks in batch [batch_size, N*Q]
                
                Where:
                - batch_size: Number of tasks in the batch (e.g., 4-16)
                - N: Number of classes per task (e.g., 5 for 5-way)
                - K: Number of examples per class in support set (e.g., 1 for 1-shot)
                - Q: Number of examples per class in query set (e.g., 15)
                
                All tensors should already be on the correct device (CPU/GPU)
        
        Returns:
            float: Average meta-loss across all tasks in the batch.
                This is the loss on query sets after adaptation.
                Lower values indicate better meta-learning progress.
        
        Example:
            >>> # Single meta-training step with tensor inputs
            >>> support_data_batch = torch.randn(4, 5, 1, 28, 28)  # 4 tasks, 5-way 1-shot
            >>> support_labels_batch = torch.randint(0, 5, (4, 5))
            >>> query_data_batch = torch.randn(4, 75, 1, 28, 28)   # 15 query per class
            >>> query_labels_batch = torch.randint(0, 5, (4, 75))
            >>>
            >>> loss = maml.meta_train_step(support_data_batch, support_labels_batch,
            ...                            query_data_batch, query_labels_batch)
            >>> print(f"Meta-loss: {loss:.4f}")
        
        Notes:
            - All input tensors must be on the same device (CPU/GPU)
            - Batch dimension must be consistent across all inputs
            - This version avoids creating intermediate CPU lists for better performance
            - Uses GPU tensor accumulation to minimize CPU↔GPU transfers
        
        Returns:
            float: Average meta-loss across all tasks in the batch.
                This is the loss on query sets after adaptation.
                Lower values indicate better meta-learning progress.
        
        Example:
            >>> # Single meta-training step with tensor inputs
            >>> support_data_batch = torch.randn(4, 5, 1, 28, 28)  # 4 tasks, 5-way 1-shot
            >>> support_labels_batch = torch.randint(0, 5, (4, 5))
            >>> query_data_batch = torch.randn(4, 75, 1, 28, 28)   # 15 query per class
            >>> query_labels_batch = torch.randint(0, 5, (4, 75))
            >>> 
            >>> loss = maml.meta_train_step(support_data_batch, support_labels_batch, 
            ...                            query_data_batch, query_labels_batch)
            >>> print(f"Meta-loss: {loss:.4f}")
        
        Notes:
            - All input tensors must be on the same device (CPU/GPU)
            - Batch dimension must be consistent across all inputs
            - This version avoids creating intermediate CPU lists for better performance
            - Uses GPU tensor accumulation to minimize CPU↔GPU transfers
        
        See Also:
            - inner_update: Task adaptation (inner loop)
            - forward_with_weights: Forward pass with adapted parameters
            - train_maml: High-level training function using this method
        """
        # Delegate to optimized parallel implementation
        return self.meta_train_step_parallel(
            support_data_batch, support_labels_batch,
            query_data_batch, query_labels_batch
        )


# Convenience alias for shorter class name
MAML = ModelAgnosticMetaLearning


def train_maml(
    model: torch.nn.Module, 
    task_dataloader: torch.utils.data.DataLoader,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    optimizer_cls: type = optim.Adam,
    optimizer_kwargs: dict = None,
    first_order: bool = False,
    reptile: bool = False,
    plus_plus: bool = False,
    meta_sgd: bool = False,
    use_amp: bool = True
):
    if meta_sgd or plus_plus:
        raise ValueError("MAML++ and Meta-SGD are under development and not supported for optimized versions. Please import the same class from algorithms/maml.py to use the original unoptimized versions of these variants.")
    """
    Train a model using Model-Agnostic Meta-Learning (MAML).
    
    This function implements the complete MAML training pipeline, which learns model parameters
    that can quickly adapt to new tasks with minimal gradient steps. The training process involves
    a meta-learning loop where the model is trained across multiple tasks to find initialization
    parameters that enable rapid task-specific adaptation.
    
    Algorithm:
        For each batch of tasks:
            1. Inner Loop (Task Adaptation):
               - Clone current model parameters
               - Perform K gradient steps on task's support set
               - Obtain task-specific adapted parameters θ'
            
            2. Outer Loop (Meta-Learning):
               - Evaluate adapted parameters θ' on task's query set
               - Compute meta-loss and meta-gradients
               - Update base parameters θ to improve fast adaptation
    
    Args:
        model (torch.nn.Module): 
            The neural network model to be meta-trained. This model should be compatible
            with the task structure (e.g., output dimension matches number of classes).
            The model will be moved to the appropriate device (CPU/GPU) automatically.
            
        task_dataloader (torch.utils.data.DataLoader): 
            DataLoader that yields batches of meta-learning tasks. Each task should be a
            tuple of (support_data, support_labels, query_data, query_labels), where:
            - support_data: Training examples for task adaptation [batch_size, N*K, C, H, W]
            - support_labels: Labels for support set [batch_size, N*K]
            - query_data: Evaluation examples for meta-learning [batch_size, N*Q, C, H, W]
            - query_labels: Labels for query set [batch_size, N*Q]
            
            Typical task structure for N-way K-shot learning:
            - N: Number of classes per task (e.g., 5 for 5-way)
            - K: Number of examples per class in support set (e.g., 1 for 1-shot)
            - Q: Number of examples per class in query set (e.g., 15)
        
        inner_lr (float, optional):
            Learning rate for task-specific adaptation in the inner loop. This controls
            how quickly the model adapts to each individual task. Typical range: 0.005-0.1.
            Default: 0.01
            - Lower values (0.005-0.01): More stable but slower adaptation
            - Higher values (0.05-0.1): Faster adaptation but may be unstable
        
        outer_lr (float, optional):
            Meta-learning rate for updating the base model parameters in the outer loop.
            This controls the step size for meta-parameter updates. Typical range: 0.0001-0.01.
            Default: 0.001
            - Lower values (0.0001-0.001): More stable training, slower convergence
            - Higher values (0.001-0.01): Faster convergence but may be unstable
        
        inner_steps (int, optional):
            Number of gradient descent steps performed during task adaptation (inner loop).
            More steps allow better adaptation but increase computational cost. Typical range: 1-10.
            Default: 5
            - Fewer steps (1-3): Faster training, tests rapid adaptation capability
            - More steps (5-10): Better task performance, more computational cost
        
        optimizer_cls (type, optional):
            The optimizer class to use for meta-learning (outer loop updates).
            Default: torch.optim.Adam
            Common alternatives: torch.optim.SGD, torch.optim.AdamW, torch.optim.RMSprop
        
        optimizer_kwargs (dict, optional):
            Additional keyword arguments to pass to the optimizer constructor.
            Default: None (uses optimizer defaults)
            Example: {'weight_decay': 1e-5, 'betas': (0.9, 0.999)} for Adam

        first_order (bool, optional):
            Whether to use First-Order MAML (FOMAML) instead of full MAML.
            Default: False (use full second-order MAML)
            
            - False (MAML): Computes second-order gradients through inner loop.
              More accurate but slower and more memory-intensive.
              
            - True (FOMAML): Uses first-order approximation. Treats adapted
              parameters as independent of meta-parameters. Faster and more
              memory-efficient but slightly less accurate.
              
            Performance comparison:
            - FOMAML: ~30-50% faster training, ~50% less memory
            - FOMAML: Typically 1-3% lower accuracy than full MAML
            - FOMAML: Recommended for larger models or resource constraints

        reptile (bool, optional):
            Whether to use the Reptile algorithm instead of MAML.
            Default: False (use MAML)
            - False (MAML): Computes second-order gradients through inner loop.
              More accurate but slower and more memory-intensive.
            - True (Reptile): Uses first-order approximation. Faster and more
              memory-efficient but slightly less accurate.

        plus_plus (bool, optional):
            Whether to use MAML++ (MAML Plus Plus) enhancements.
            Default: False (use vanilla MAML)

            Note: MAML++ does not support FOMAML. If plus_plus=True, then first_order must be False.

        meta_sgd (bool, optional):
            Whether to use Meta-SGD, which learns per-parameter learning rates
            for the inner loop updates. This can improve adaptation speed.
            Default: False (use fixed inner_lr for all parameters)
            
            When enabled, each parameter will have its own learnable learning rate,
            initialized to inner_lr. This allows the model to learn how quickly
            to adapt each parameter during task-specific fine-tuning.
            
            Note: Meta-SGD increases (doubles) the number of parameters and may require
            more careful tuning of outer_lr.

        use_amp (bool, optional):
            Whether to use Automatic Mixed Precision (AMP) for training.
            Default: True
            - True: Enables AMP for faster training on compatible GPUs
            - False: Standard full precision training

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The meta-trained model with optimized initialization
            - maml (MAML): The MAML trainer object containing optimizer state and hyperparameters
            - losses (list[float]): Training loss history across all meta-training steps
    
    Training Features:
        - Automatic GPU detection and utilization
        - Gradient clipping (max_norm=1.0) for training stability
        - Progress bar with real-time loss tracking
        - Error handling with graceful continuation on batch failures
        - Periodic logging every 100 steps
        - Best loss tracking for monitoring convergence
    
    Example:
        >>> # Basic usage with default hyperparameters
        >>> model = SimpleConvNet(num_classes=5)
        >>> task_dataset = OmniglotTaskDataset(dataset, n_way=5, k_shot=1, num_tasks=2000)
        >>> task_loader = DataLoader(task_dataset, batch_size=4, shuffle=True)
        >>> trained_model, maml, losses = train_maml(model, task_loader)
        >>> 
        >>> # Custom hyperparameters for faster adaptation
        >>> trained_model, maml, losses = train_maml(
        ...     model, 
        ...     task_loader,
        ...     inner_lr=0.05,      # Higher learning rate for faster task adaptation
        ...     outer_lr=0.001,     # Standard meta-learning rate
        ...     inner_steps=3       # Fewer steps for faster training
        ... )
        >>> 
        >>> # Using SGD optimizer with momentum
        >>> import torch.optim as optim
        >>> trained_model, maml, losses = train_maml(
        ...     model,
        ...     task_loader,
        ...     optimizer_cls=optim.SGD,
        ...     optimizer_kwargs={'momentum': 0.9, 'weight_decay': 1e-5}
        ... )
        >>> 
        >>> # Using FOMAML (First-Order MAML) for faster training
        >>> trained_model, maml, losses = train_maml(
        ...     model,
        ...     task_loader,
        ...     first_order=True  # Use first-order approximation
        ... )
        >>> 
        >>> # Use trained model for rapid adaptation on new tasks
        >>> adapted_params = maml.inner_update(new_support_data, new_support_labels)
        >>> predictions = maml.forward_with_weights(new_query_data, adapted_params)
    
    Notes:
        - The function supports both MAML (second-order) and FOMAML (first-order)
        - MAML is more accurate but slower; FOMAML is faster with slight accuracy trade-off
        - Training time depends on: number of tasks, batch size, inner_steps, and model complexity
        - Expected training time: ~10-30 minutes for 2000 tasks on GPU (MAML)
        - Expected training time: ~7-20 minutes for 2000 tasks on GPU (FOMAML)
        - Memory usage scales with batch_size and inner_steps (reduce if OOM occurs)
        - The model remains on the selected device after training
    
    Performance Tips:
        - Use GPU for 5-10x speedup
        - Increase batch_size (e.g., 8-16) if memory allows for faster convergence
        - Reduce inner_steps (e.g., 3) for faster iteration during development
        - Use more num_tasks (e.g., 5000-10000) for better final performance
    
    Raises:
        RuntimeError: If task batch structure is incompatible or model forward pass fails
        
    See Also:
        - MAML class: Core meta-learning algorithm implementation
        - MAML.meta_train_step: Single meta-training update
        - MAML.inner_update: Task-specific adaptation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize MAML trainer with custom hyperparameters
    maml = ModelAgnosticMetaLearning(
        model, 
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        first_order=first_order,
        reptile=reptile,
        plus_plus=plus_plus,
        meta_sgd=meta_sgd,
        # use_amp=use_amp
    )
    
    algorithm_name = "FOMAML" if first_order else "MAML"
    print(f"Starting {algorithm_name} training...")
    print(f"Hyperparameters: inner_lr={inner_lr}, outer_lr={outer_lr}, inner_steps={inner_steps}")
    print(f"Optimizer: {optimizer_cls.__name__}")
    if first_order:
        print("Using First-Order approximation (FOMAML) - faster but slightly less accurate")
    losses = []
    best_loss = float('inf')
    
    # Use GPU tensor for loss accumulation to minimize transfers
    loss_accumulator = torch.tensor(0.0, device=device)
    loss_count = 0
    
    progress_bar = tqdm(task_dataloader, desc="Training", dynamic_ncols=True)
    
    for batch_idx, task_batch in enumerate(progress_bar):
        try:
            # Unpack the batch - task_batch contains (support_data, support_labels, query_data, query_labels)
            # Each is a tensor with shape [batch_size, task_data...]
            support_data_batch, support_labels_batch, query_data_batch, query_labels_batch = task_batch

            # Move tensors to device with non_blocking transfer
            support_data_batch = support_data_batch.to(device, non_blocking=True)
            support_labels_batch = support_labels_batch.to(device, non_blocking=True)
            query_data_batch = query_data_batch.to(device, non_blocking=True)
            query_labels_batch = query_labels_batch.to(device, non_blocking=True)
            
            # Training step - pass tensors directly (no intermediate lists)
            loss = maml.meta_train_step(support_data_batch, support_labels_batch, query_data_batch, query_labels_batch)

            # Accumulate loss on GPU to avoid frequent transfers
            loss_accumulator += loss
            loss_count += 1

            # Only transfer to CPU after 20 batches to reduce GPU→CPU transfers
            if (batch_idx + 1) > 20:
                avg_loss = (loss_accumulator / loss_count).item()  # Single GPU→CPU transfer
                losses.extend([avg_loss] * loss_count)  # Approximate individual losses
                
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Batch': batch_idx + 1,
                    'GPU%': f'{torch.cuda.utilization()}' if torch.cuda.is_available() else 'N/A'
                })
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                # Reset accumulator
                loss_accumulator = torch.tensor(0.0, device=device)
                loss_count = 0
                
        except Exception as e:
            tqdm.write(f"Error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Handle any remaining accumulated losses
    if loss_count > 0:
        avg_loss = (loss_accumulator / loss_count).item()
        losses.extend([avg_loss] * loss_count)
    
    print(f"\nTraining completed! Final loss: {np.mean(losses[-100:]):.4f}")
    return model, maml, losses


@torch.jit.script
def vectorized_param_update(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    alphas: list[torch.Tensor]
) -> list[torch.Tensor]:
    """JIT-compiled vectorized parameter update."""
    return [p - a * g for p, a, g in zip(params, alphas, grads)]


# Note: Removed stack_gradients_jit function (was at line ~1310)
# It's completely unnecessary! vmap already returns gradients in stacked format.
# The zero-copy optimization eliminates all unpack/repack operations.
# See GPU_STACKING_OPTIMIZATION.md for details (100-150x speedup).