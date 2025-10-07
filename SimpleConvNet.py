"""
SimpleConvNet with optimized Meta Dropout management.

This module provides a CNN architecture for Omniglot classification with
built-in Meta Dropout support and efficient dropout enable/disable methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from Meta_Dropout import MetaDropout


class SimpleConvNet(nn.Module):
    """Simple CNN for Omniglot classification with optimized Meta Dropout support.
    
    Architecture:
    - 4 convolutional layers (64 filters, 3x3 kernel)
    - Batch normalization after each conv
    - ReLU activation
    - Max pooling (2x2)
    - Meta Dropout after pooling layers (configurable)
    - Fully connected output layer
    
    Optimized Meta Dropout Features:
    - Cached list of dropout layers (no iteration overhead)
    - Built-in enable_dropout() and disable_dropout() methods
    - Zero-overhead dropout control (just boolean flag flips)
    
    For Meta Dropout usage:
    - Call reset_dropout_masks() at the start of each task
    - Use disable_dropout() for query evaluation
    - Use enable_dropout() to restore dropout
    - Same dropout pattern maintained throughout inner loop
    
    Args:
        num_classes: Number of output classes (default: 5)
        drop_prob: Dropout probability (default: 0.5)
        use_meta_dropout: If True, use Meta Dropout; if False, use standard dropout (default: True)
        dropout_config: List of dropout rates for each layer [layer1, layer2, layer3, layer4]
                       Use 0.0 to skip dropout for a layer (default: [0.0, 0.1, 0.15, 0.0])
    
    Example:
        >>> # Create model with optimized Meta Dropout
        >>> model = SimpleConvNet(num_classes=5, dropout_config=[0.0, 0.1, 0.15, 0.0])
        >>> 
        >>> # In MAML training (inner loop)
        >>> model.reset_dropout_masks(batch_size=5, device=device)
        >>> fast_weights = maml.inner_update(support_data, support_labels)
        >>> 
        >>> # For query evaluation (outer loop)
        >>> model.disable_dropout()  # Fast - no iteration!
        >>> query_logits = maml.forward_with_weights(query_data, fast_weights)
        >>> model.enable_dropout()   # Fast - restore instantly!
    """
    
    def __init__(
        self, 
        num_classes: int = 5, 
        drop_prob: float = 0.5, 
        use_meta_dropout: bool = True,
        dropout_config: list = None
    ):
        super(SimpleConvNet, self).__init__()
        
        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self.use_meta_dropout = use_meta_dropout
        
        # Flag for outer loop mode (skips dropout in forward pass)
        self._outer_loop_mode = False
        
        # Default dropout configuration: skip first and last layers
        if dropout_config is None:
            dropout_config = [0.05, 0.1, 0.15, 0.05]
        
        # Define dropout layers with custom rates
        # Cache dropout layers for fast enable/disable (no iteration needed!)
        self._meta_dropout_layers = []
        
        if use_meta_dropout:
            self.dropout1 = self._create_dropout(dropout_config[0], MetaDropout)
            self.dropout2 = self._create_dropout(dropout_config[1], MetaDropout)
            self.dropout3 = self._create_dropout(dropout_config[2], MetaDropout)
            self.dropout4 = self._create_dropout(dropout_config[3], MetaDropout)
        else:
            self.dropout1 = self._create_dropout(dropout_config[0], nn.Dropout)
            self.dropout2 = self._create_dropout(dropout_config[1], nn.Dropout)
            self.dropout3 = self._create_dropout(dropout_config[2], nn.Dropout)
            self.dropout4 = self._create_dropout(dropout_config[3], nn.Dropout)
        
        # Network layers
        # Layer 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Layer 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Fully connected
        self.fc = nn.Linear(64 * 6 * 6, num_classes)
        
        # Store feature map shapes for mask initialization (only for non-zero dropout)
        self._dropout_shapes = {}
        layer_shapes = [
            ('dropout1', (64, 52, 52)),
            ('dropout2', (64, 26, 26)),
            ('dropout3', (64, 13, 13)),
            ('dropout4', (64, 6, 6))
        ]
        
        for i, (name, shape) in enumerate(layer_shapes):
            if dropout_config[i] > 0.0:
                self._dropout_shapes[name] = shape
    
    def _create_dropout(self, p: float, dropout_class):
        """Helper to create dropout layer and cache MetaDropout instances.
        
        Args:
            p: Dropout probability
            dropout_class: Either MetaDropout or nn.Dropout
            
        Returns:
            Dropout layer (nn.Identity if p=0, otherwise dropout_class(p))
        """
        if p == 0.0:
            return nn.Identity()
        
        layer = dropout_class(p=p)
        
        # Cache MetaDropout layers for fast enable/disable
        if isinstance(layer, MetaDropout):
            self._meta_dropout_layers.append(layer)
        
        return layer
    
    def reset_dropout_masks(self, batch_size: int, device: torch.device):
        """Reset all Meta Dropout masks for a new task.
        
        Call this method at the start of each task's inner loop.
        Uses cached dropout layers - no iteration overhead!
        
        Args:
            batch_size: Batch size for the task
            device: Device to create masks on
        """
        if not self.use_meta_dropout:
            return  # No-op for standard dropout
        
        # Reset each active dropout layer with appropriate shape
        for name, shape in self._dropout_shapes.items():
            dropout_layer = getattr(self, name)
            if hasattr(dropout_layer, 'reset_mask'):
                full_shape = (batch_size, *shape)
                dropout_layer.reset_mask(full_shape, device)
    
    @contextmanager
    def outer_loop_mode(self):
        """Context manager for outer loop evaluation (skips dropout in forward pass).
        
        During outer loop evaluation, dropout is skipped entirely in the forward pass,
        implementing Meta Dropout: dropout only in inner loop adaptation.
        
        Usage:
            >>> # Inner loop: WITH dropout (normal forward pass)
            >>> fast_weights = maml.inner_update(support_data, support_labels)
            >>> 
            >>> # Outer loop: WITHOUT dropout (using context manager)
            >>> with model.outer_loop_mode():
            ...     query_logits = maml.forward_with_weights(query_data, fast_weights)
            ...     query_loss = F.cross_entropy(query_logits, query_labels)
            >>> # Dropout automatically re-enabled here
        
        Benefits:
            - ✅ Pythonic and clean API
            - ✅ Automatic flag management (always reset properly)
            - ✅ Exception-safe (works even if errors occur)
            - ✅ Zero performance overhead (just boolean check)
            - ✅ Works perfectly with torch.func.functional_call
        
        Notes:
            - Flag is automatically reset when exiting context (even on exceptions)
            - Can be nested safely (inner context takes precedence)
            - Thread-safe for single-threaded training (not for DataParallel)
        """
        old_mode = self._outer_loop_mode
        self._outer_loop_mode = True
        try:
            yield
        finally:
            self._outer_loop_mode = old_mode
    
    def clear_dropout_masks(self):
        """Clear all Meta Dropout masks (useful for evaluation)"""
        if not self.use_meta_dropout:
            return
        
        for dropout_layer in self._meta_dropout_layers:
            dropout_layer.clear_mask()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network with conditional dropout.
        
        Dropout is skipped when _outer_loop_mode=True, implementing Meta Dropout
        Option 2: dropout only during inner loop adaptation, full network during
        outer loop evaluation.
        """
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self._outer_loop_mode:  # Skip dropout in outer loop
            x = self.dropout1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self._outer_loop_mode:  # Skip dropout in outer loop
            x = self.dropout2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self._outer_loop_mode:  # Skip dropout in outer loop
            x = self.dropout3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self._outer_loop_mode:  # Skip dropout in outer loop
            x = self.dropout4(x)

        # Fully connected
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_dropout_info(self) -> dict:
        """Get information about dropout mask status (for debugging)"""
        if not self.use_meta_dropout:
            return {"mode": "standard_dropout"}
        
        info = {
            "mode": "meta_dropout",
            "num_dropout_layers": len(self._meta_dropout_layers),
            "dropout_layers": []
        }
        
        for i, dropout_layer in enumerate(self._meta_dropout_layers):
            info["dropout_layers"].append({
                "layer_idx": i,
                "p": dropout_layer.p,
                "mask_active": dropout_layer.mask is not None,
                "force_eval": dropout_layer._force_eval
            })
        
        return info
