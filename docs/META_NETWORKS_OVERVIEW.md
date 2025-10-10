# Meta Networks Overview

## 📚 What are Meta Networks?

Meta Networks is a meta-learning algorithm that learns to generate task-specific parameters (called "fast weights") through a meta-learner network. Unlike MAML which learns a good initialization for gradient-based adaptation, Meta Networks directly produce classifier parameters from support set examples.

**Paper**: [Meta Networks](https://arxiv.org/abs/1703.00837) - Munkhdalai & Yu, ICML 2017

---

## ⚠️ Important: This is an Embedding-Based Variant

**This implementation is a variant of the original Meta Networks algorithm**, commonly known as **Embedding-based Meta Networks**. 

### 🔍 Key Differences from Original Meta Networks:

| Aspect | **This Implementation** | **Original Meta Networks** |
|--------|------------------------|----------------------------|
| **Name** | Embedding-based Meta Networks | Meta Networks |
| **Category** | 🎯 **Metric-based Meta Learning** | 🏗️ **Model-based Meta Learning** |
| **Approach** | Generates task-specific embeddings for metric learning | Generates weights for the entire base network |
| **Fast Weights** | Used for computing similarity metrics | Used as actual network parameters |
| **Similarity** | Closer to Matching Networks / Prototypical Networks | More general weight generation framework |

### 📝 What This Means:

- ✅ **This variant** uses the meta-learner to generate task-specific embeddings that are used in a metric-based classification approach (similar to prototypical networks)
- 🔜 **Original approach** (coming soon): The meta-learner predicts the actual weights of the base network, making it a true model-based meta-learning algorithm

### 🚀 Roadmap:

The **original Meta Networks implementation** (where the meta-learner predicts base network weights) will be added to this repository next. This is the more commonly used and cited approach from the original paper.

---

## 🏗️ Architecture

### Three Main Components:

1. **Embedding Network** (`EmbeddingNetwork`)
   - 4 convolutional layers with batch normalization
   - Extracts fixed-dimensional feature embeddings from images
   - Input: 105×105 grayscale images
   - Output: 64-dimensional embeddings

2. **Meta-Learner** (`MetaLearner`)
   - Learns three key parameters:
     - **U Matrix** (hidden_dim × embedding_dim): Projects support embeddings
     - **V Matrix** (hidden_dim × embedding_dim): Projects query embeddings
     - **e Vector** (hidden_dim): Base embedding capturing task structure
   - Generates task-specific classifier weights from support set

3. **Meta Network** (`MetaNetwork`)
   - Combines EmbeddingNetwork and MetaLearner
   - End-to-end trainable system

## 🔄 How It Works

### Training Algorithm:

For each batch of tasks:
1. **Extract embeddings** from support and query sets using EmbeddingNetwork
2. **Generate fast weights** from support embeddings using MetaLearner
   - For each support example (x_i, y_i):
     - Compute embedding: h_i = EmbeddingNetwork(x_i)
     - Project: r_i = U @ h_i
     - Add base: w_i = r_i + e
   - Average per class: W_c = mean(w_i for all i where y_i = c)
3. **Classify queries** using generated weights
   - For each query x:
     - Compute embedding: h = EmbeddingNetwork(x)
     - Project: query_proj = V @ h
     - Compute logits: logits_c = query_proj^T @ W_c
4. **Backpropagate** loss to update U, V, e and EmbeddingNetwork

### Inference:
- **Single forward pass** - no gradient-based adaptation needed!
- Very fast compared to MAML's inner loop optimization
- Meta-learner directly generates optimal classifier

## 📊 Performance

### Omniglot (5-way 1-shot):
- ~ Accuracy: **75.5%**
- Competitive with MAML
- Much faster inference (no adaptation loop)

## 💡 Key Differences from MAML

| Aspect | MAML | Meta Networks |
|--------|------|---------------|
| **Approach** | Learn good initialization | Learn to generate parameters |
| **Adaptation** | Gradient-based (inner loop) | Direct generation (meta-learner) |
| **Inference Speed** | Slower (requires gradient steps) | Faster (single forward pass) |
| **Parameters** | Model parameters | U, V, e matrices + embedding network |
| **Memory** | Higher (computation graph) | Lower (no inner loop gradients) |

## 🚀 Usage

### Training:

```python
from algorithms.eb_meta_network import MetaNetwork, train_meta_network
from utils.load_omniglot import OmniglotDataset, OmniglotTaskDataset
from torch.utils.data import DataLoader

# Load data
dataset = OmniglotDataset("omniglot/images_background")
task_dataset = OmniglotTaskDataset(dataset, n_way=5, k_shot=1, k_query=15, num_tasks=2000)
dataloader = DataLoader(task_dataset, batch_size=4, shuffle=True)

# Create and train model
model = MetaNetwork(embedding_dim=64, hidden_dim=128, num_classes=5)
model, optimizer, losses = train_meta_network(
	model=model,
	task_dataloader=dataloader,
	learning_rate=0.001
)
```

### Evaluation:

```python
from algorithms.eb_meta_network import evaluate_meta_network
from evaluation.eval_visualization import plot_evaluation_results

# Evaluate on test tasks
eval_results = evaluate_meta_network(
	model=model,
	eval_dataloader=test_dataloader,
	num_classes=5
)

# Visualize results
plot_evaluation_results(eval_results)
```

## 📁 Files

- **`EB_Meta_Network.py`**: Complete Meta Networks implementation
  - `EmbeddingNetwork`: Feature extractor
  - `MetaLearner`: Fast weight generator
  - `MetaNetwork`: Combined system
  - `train_meta_network()`: Training function
  - `evaluate_meta_network()`: Evaluation function

- **`examples/embedding_based_meta_network.ipynb`**: Complete tutorial notebook

- **Shared utilities** (used by both MAML and Meta Networks):
  - `utils/load_omniglot.py`: Dataset loaders
  - `utils/evaluate.py`: Visualization functions

## 🎯 Hyperparameters

### Default Configuration:
- **Embedding dimension**: 64
- **Hidden dimension**: 128 (for U, V, e)
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Gradient clipping**: max_norm=1.0
- **Batch size**: 4 tasks

### Task Configuration:
- **N-way**: 5 (5 characters per task)
- **K-shot**: 1 (1 support example per class)
- **K-query**: 15 (15 query examples per class)
- **Training tasks**: 2000
- **Test tasks**: 200

## 🔬 What the Meta-Learner Learns

The U, V, and e parameters learn to:
- **U**: Extract task-relevant features from support embeddings
- **V**: Transform query embeddings for classification
- **e**: Capture base task structure (what's common across tasks)

Together, they form a powerful mechanism for rapid classifier generation!

## 📚 References

1. **Meta Networks Paper**: Munkhdalai & Yu, "Meta Networks", ICML 2017
   - https://arxiv.org/abs/1703.00837

2. **Omniglot Dataset**: Lake et al., "Human-level concept learning through probabilistic program induction"
   - https://github.com/brendenlake/omniglot

3. **Related**: MAML (Model-Agnostic Meta-Learning)
   - See `docs/MAML_vs_FOMAML.md` for comparison

## 🎓 Tutorial

See `examples/embedding_based_meta_network.ipynb` for a complete step-by-step tutorial with:
- Data loading and visualization
- Model architecture explanation
- Training from scratch
- Evaluation and visualization
- Comparison with MAML

---

**Happy Meta-Learning!** 🤖💡
