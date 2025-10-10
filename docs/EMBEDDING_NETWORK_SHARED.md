# 🔗 Shared EmbeddingNetwork Architecture

## Overview

The `EmbeddingNetwork` class serves as a shared CNN feature extractor used across multiple meta-learning implementations in this repository. This design ensures consistency, code reusability, and fair comparisons between different algorithms.

## Used By

### 🎯 Embedding-based Meta Networks (`EB_Meta_Network.py`)
- **Purpose**: Extracts features for metric-based classification
- **Usage**: Embeddings are used to compute distances/similarities between support and query examples
- **Algorithm**: Generates class prototypes and classifies based on nearest embeddings

### 🧠 Original Meta Networks (`Original_Meta_Network.py`)
- **Purpose**: Extracts features for weight prediction-based classification
- **Usage**: Embeddings are processed to generate actual FC layer weights and biases
- **Algorithm**: Meta-learner predicts classifier parameters from support set embeddings

## Architecture Details

### CNN Structure
```
Input: [batch_size, 1, 105, 105] (Omniglot grayscale images)

Layer 1: Conv2d(1→64, 3×3) → BatchNorm → ReLU → MetaDropout → MaxPool(2×2)
         Output: [batch_size, 64, 52, 52]

Layer 2: Conv2d(64→64, 3×3) → BatchNorm → ReLU → MetaDropout → MaxPool(2×2)
         Output: [batch_size, 64, 26, 26]

Layer 3: Conv2d(64→64, 3×3) → BatchNorm → ReLU → MetaDropout → MaxPool(2×2)
         Output: [batch_size, 64, 13, 13]

Layer 4: Conv2d(64→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
         Output: [batch_size, 64, 6, 6]

Flatten: [batch_size, 64 × 6 × 6] = [batch_size, 2304]

FC Layer: Linear(2304 → embedding_dim)
          Output: [batch_size, embedding_dim] (default: 64)
```

### Key Features

#### 🎭 Meta Dropout Integration
- **Consistent Masks**: Same dropout pattern across support and query sets within a task
- **Task-level Reset**: New masks generated for each new task
- **Configuration**: Three dropout layers with rates [0.05, 0.10, 0.15] by default

#### 🔧 Flexible Configuration
```python
# Default configuration
embedding_network = EmbeddingNetwork(
    embedding_dim=64,           # Output feature dimension
    dropout_rates=[0.05, 0.10, 0.15]  # Dropout rates for layers 1-3
)
```

#### 🚀 Efficient Implementation
- **Shared Computation**: Same feature extraction for both meta-learning variants
- **Memory Efficient**: Reuses learned representations
- **GPU Compatible**: Automatic device handling

## Implementation Benefits

### Code Reusability ♻️
- **Single Source**: One implementation used by multiple algorithms
- **Maintenance**: Updates benefit all meta-learning implementations
- **Consistency**: Same architecture ensures fair algorithm comparisons

### Performance Benefits 📈
- **Meta Dropout**: Improved few-shot learning performance
- **Standardized**: Well-tested architecture optimized for Omniglot
- **Efficient**: Optimized for 105×105 grayscale images

### Research Benefits 🔬
- **Fair Comparison**: Same feature extractor isolates algorithm differences
- **Controlled Experiments**: Only meta-learning strategy varies
- **Reproducible**: Consistent results across different runs

## Usage Examples

### Basic Usage
```python
from algorithms.EmbeddingNetwork import EmbeddingNetwork

# Create embedding network
embedding_net = EmbeddingNetwork(embedding_dim=64)

# Extract features
features = embedding_net(input_images)  # [batch_size, 64]
```

### Custom Configuration
```python
# Custom dropout rates and embedding dimension
embedding_net = EmbeddingNetwork(
    embedding_dim=128,
    dropout_rates=[0.1, 0.2, 0.3]
)
```

### Meta Dropout Reset (automatic in forward pass)
```python
# Reset dropout masks for new task (handled automatically)
embedding_net.reset_dropout_masks(input_shape, device)
features = embedding_net(support_images)  # Same masks
query_features = embedding_net(query_images)  # Same masks
```

## Integration with Meta-Learning Algorithms

### Embedding-based Meta Networks Flow
```
1. EmbeddingNetwork extracts features from support and query sets
2. MetaLearner generates fast weights (U, V, e) from support embeddings
3. Classification uses metric learning on embeddings
```

### Original Meta Networks Flow
```
1. EmbeddingNetwork extracts features from support and query sets
2. MetaLearner generates actual FC weights and biases from support embeddings
3. Classification uses predicted parameters: logits = query_embeddings @ W + b
```

## File Location
`algorithms/EmbeddingNetwork.py`

## Dependencies
- `torch.nn`: Neural network modules
- `torch.nn.functional`: Activation functions
- `Meta_Dropout`: Custom dropout implementation

---

**This shared architecture ensures that when comparing Embedding-based vs Original Meta Networks, the only difference is in the meta-learning strategy, not the feature extraction capability.** 🎯