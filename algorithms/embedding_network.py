import torch
import torch.nn as nn
import torch.nn.functional as F
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
