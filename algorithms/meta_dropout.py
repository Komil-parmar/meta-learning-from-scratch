import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


class MetaDropout(nn.Module):
    """Dropout layer that maintains consistent masks across inner loop.

    For Meta Dropout in MAML:
    - Call reset_mask() at the start of each new task
    - Same mask is reused throughout the task's inner loop adaptation
    - Falls back to standard dropout if mask not initialized

    **Dropout Disabling Approaches:**

    1. **Model-level control (Recommended):**
       Check `_outer_loop_mode` in your model's forward pass to skip dropout layers.
       Used in SimpleConvNet: `if not self._outer_loop_mode: x = self.dropout1(x)`

    2. **Layer-level control (Alternative):**
       Use `eval_mode()` and `train_mode()` methods for models where you can't
       easily modify the forward pass logic.
       ```python
       model.disable_dropout()  # Calls eval_mode() on all MetaDropout layers
       outputs = model(query_data)
       model.enable_dropout()   # Calls train_mode() on all MetaDropout layers
       ```

    3. **Standard PyTorch:**
       Calling `model.eval()` automatically disables all MetaDropout layers
       via the `not self.training` check.
       Note that this also affects BatchNorm and other layers. And is extremely slow /
       done frequently like changing between inner and outer loop.

    The mask is batch-size agnostic (shape: [1, C, H, W]) and broadcasts
    across different batch sizes, making it efficient for MAML where
    support set and query set have different batch sizes.

    Args:
        p: Dropout probability (default: 0.5)
        inplace: If True, modifies input in-place (default: False)
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(MetaDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be between 0 and 1, got {p}")
        
        self.p = p
        self.inplace = inplace
        self.mask = None
        self._spatial_shape = None  # Store (C, H, W) without batch dimension
        self._force_eval = False  # Flag for temporary eval mode (fast dropout disable)
        
    def reset_mask(self, shape: tuple, device: torch.device):
        """Generate a new dropout mask for a task.
        
        Creates a batch-size agnostic mask with shape [1, C, H, W] that
        broadcasts across different batch sizes during forward passes.
        
        Args:
            shape: Shape of the input (batch_size, channels, height, width)
                   Only the spatial dimensions (C, H, W) are used for the mask
            device: Device to create mask on
        """
        # Extract spatial dimensions (ignore batch size)
        if len(shape) == 4:
            spatial_shape = shape[1:]  # (C, H, W)
        else:
            spatial_shape = shape
        
        # Generate Bernoulli mask with batch dimension = 1 for broadcasting
        self.mask = torch.bernoulli(
            torch.ones(1, *spatial_shape, device=device) * (1 - self.p),
        ).to(device)
        # Scale to maintain expected value: E[output] = E[input]
        # self.mask = self.mask / (1 - self.p)
        self._spatial_shape = spatial_shape
        
    def clear_mask(self):
        """Clear stored mask (useful for evaluation mode)"""
        self.mask = None
        self._spatial_shape = None
    
    def eval_mode(self):
        """Temporarily disable dropout without changing training flag.
        
        This is faster than model.eval() because it only affects this dropout layer,
        keeping BatchNorm and other layers in training mode.
        """
        self._force_eval = True
        
    def train_mode(self):
        """Re-enable dropout after temporary disable"""
        self._force_eval = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check force_eval flag first (fast path - just a boolean check)
        if self._force_eval or not self.training:
            # No dropout during evaluation
            return x
            
        if self.mask is not None:
            # Use stored mask (Meta Dropout mode)
            # Mask broadcasts across batch dimension automatically
            # No shape check needed - broadcasting handles different batch sizes!
            return x * self.mask
        else:
            # Fall back to standard dropout if mask not initialized
            return F.dropout(x, p=self.p, training=True, inplace=self.inplace)
    
    def extra_repr(self) -> str:
        """String representation for printing model architecture"""
        return f'p={self.p}, inplace={self.inplace}, meta_mode={self.mask is not None}'