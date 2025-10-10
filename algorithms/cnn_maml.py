"""
SimpleConvNet with optimized Meta Dropout management.

This module provides a CNN architecture for Omniglot classification with
built-in Meta Dropout support and efficient dropout enable/disable methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.meta_dropout import MetaDropout


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
    - Single unified dictionary for dropout layers and shapes (zero redundancy)
    - O(1) access to both layer instances and their feature map shapes
    - Zero-overhead dropout control via _outer_loop_mode flag

    For Meta Dropout usage:
    - Call reset_dropout_masks() at the start of each task
    - Dropout is automatically skipped during outer loop via _outer_loop_mode
    - Same dropout pattern maintained throughout inner loop
    
    Args:
        num_classes: Number of output classes (default: 5)
        drop_prob: Dropout probability (default: 0.5)
        use_meta_dropout: If True, use Meta Dropout; if False, use standard dropout (default: True)
        dropout_config: List of dropout rates for each layer [layer1, layer2, layer3, layer4]
                       Use 0.0 to skip dropout for a layer (default: [0.05, 0.1, 0.15, 0.05])

    Example:
        >>> # Create model with optimized Meta Dropout
        >>> model = SimpleConvNet(num_classes=5, dropout_config=[0.05, 0.1, 0.15, 0.05])
        >>>
        >>> # In MAML training (inner loop)
        >>> model.reset_dropout_masks(batch_size=5, device=device)
        >>> fast_weights = maml.inner_update(support_data, support_labels)
        >>> 
        >>> # For query evaluation (outer loop) - automatic via _outer_loop_mode
        >>> model._outer_loop_mode = True
        >>> query_logits = maml.forward_with_weights(query_data, fast_weights)
        >>> model._outer_loop_mode = False
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
        
        # Default dropout configuration
        if dropout_config is None:
            dropout_config = [0.05, 0.1, 0.15, 0.05]
        
        # Unified dropout info: stores both layer instances and their shapes
        # Structure: {'dropout1': {'layer': MetaDropout(...), 'shape': (C, H, W)}, ...}
        # Only includes layers with p > 0.0 (skips nn.Identity layers)
        self._dropout_info = {}

        # Define feature map shapes after each pooling layer
        layer_configs = [
            ('dropout1', dropout_config[0], (64, 52, 52)),
            ('dropout2', dropout_config[1], (64, 26, 26)),
            ('dropout3', dropout_config[2], (64, 13, 13)),
            ('dropout4', dropout_config[3], (64, 6, 6))
        ]

        # Create dropout layers and populate unified dictionary
        dropout_class = MetaDropout if use_meta_dropout else nn.Dropout

        for name, p, shape in layer_configs:
            layer = self._create_dropout(p, dropout_class)
            setattr(self, name, layer)

            # Only store in _dropout_info if it's an actual dropout layer (not Identity)
            if p > 0.0:
                self._dropout_info[name] = {'layer': layer, 'shape': shape}

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

    def _create_dropout(self, p: float, dropout_class):
        """Helper to create dropout layer.

        Args:
            p: Dropout probability
            dropout_class: Either MetaDropout or nn.Dropout
            
        Returns:
            Dropout layer (nn.Identity if p=0, otherwise dropout_class(p))
        """
        return nn.Identity() if p == 0.0 else dropout_class(p=p)

    def reset_dropout_masks(self, batch_size: int, device: torch.device):
        """Reset all Meta Dropout masks for a new task.
        
        Call this method at the start of each task's inner loop.
        Uses unified dictionary - single iteration over active dropout layers!

        Args:
            batch_size: Batch size for the task
            device: Device to create masks on
        """
        if not self.use_meta_dropout:
            return  # No-op for standard dropout
        
        # Iterate over unified dictionary (layer instance + shape in one place)
        for name, info in self._dropout_info.items():
            layer = info['layer']
            shape = info['shape']
            if self.use_meta_dropout:
                full_shape = (batch_size, *shape)
                layer.reset_mask(full_shape, device)

    def clear_dropout_masks(self):
        """Clear all Meta Dropout masks (useful for evaluation)"""
        if not self.use_meta_dropout:
            return
        
        for info in self._dropout_info.values():
            info['layer'].clear_mask()

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
        if not self._outer_loop_mode:
            x = self.dropout1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self._outer_loop_mode:
            x = self.dropout2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self._outer_loop_mode:
            x = self.dropout3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self._outer_loop_mode:
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
            "num_dropout_layers": len(self._dropout_info),
            "dropout_layers": []
        }
        
        for name, dropout_info in self._dropout_info.items():
            layer = dropout_info['layer']
            info["dropout_layers"].append({
                "name": name,
                "shape": dropout_info['shape'],
                "p": layer.p,
                "mask_active": layer.mask is not None,
                "force_eval": layer._force_eval
            })
        
        return info
