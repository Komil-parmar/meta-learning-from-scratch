# 🧠 Meta-Learning From Scratch

A modular collection of meta-learning algorithm implementations built from scratch. Designed for clarity, flexibility, and easy experimentation — plug in your own datasets, define tasks, and explore how models learn to learn.

> 🎓 **Learning Journey**: This repository is part of my meta-learning exploration as a second-year undergraduate student at IIT Guwahati. I'm learning meta-learning through multiple sources, primarily following the book **"Meta-Learning: Theory, Algorithms and Applications"**. I'm building these implementations from the ground up to deeply understand how models can "learn to learn."

## 🌟 Overview

Meta-learning (or "learning to learn") enables models to quickly adapt to new tasks with minimal training data. Unlike traditional machine learning that learns a specific task, meta-learning algorithms learn how to efficiently learn new tasks.

**Current Status**: ✅ MAML & FOMAML Complete | ✅ Embedding-based Meta Networks Complete | ✅ Meta Dropout Implemented | 🚧 More algorithms coming soon!

## 🎯 Features

- **Modular Design**: Clean, reusable components that work across different datasets and tasks
- **Easy Experimentation**: Plug in your own datasets, customize task sampling, and tune hyperparameters
- **Meta Dropout Support**: Consistent dropout masks for improved few-shot learning performance
- **Well-Documented**: Comprehensive docstrings, inline comments, and tutorial notebook
- **Research-Ready**: Built for both learning and experimentation

## 📚 Algorithms Implemented

### ✅ Model-Agnostic Meta-Learning (MAML)
A flexible meta-learning approach that trains a model's initial parameters to enable rapid adaptation to new tasks with just a few gradient steps.

**Key Features:**
- Works with any gradient-based model
- Learns optimal parameter initialization
- Adapts quickly with minimal data
- [Original Paper](https://arxiv.org/abs/1703.03400)

**Documentation:** See [MAML vs FOMAML Comparison](docs/MAML_vs_FOMAML.md)

### ✅ First-Order MAML (FOMAML)
A memory-efficient variant of MAML that omits second-order derivatives, offering:
- **Faster training**: ~40% faster than full MAML
- **Lower memory usage**: No need to store computation graph for second derivatives
- **Comparable performance**: Often matches MAML accuracy with reduced computational cost

**Documentation:** See [MAML vs FOMAML Comparison](docs/MAML_vs_FOMAML.md)

### ✅ Meta Dropout
An optimized dropout strategy for meta-learning that maintains consistent dropout masks during inner loop adaptation:
- **Improved performance**: 80.1% ± 10.48% accuracy vs 78.9% ± 11.5% baseline
- **Reduced variance**: 16.9% reduction in performance variance
- **Context manager API**: Clean, exception-safe implementation

**Documentation:** See [Meta Dropout Usage Guide](docs/META_DROPOUT_USAGE.md)

### ✅ Embedding-based Meta Networks
A metric-based meta-learning approach that generates task-specific embeddings for fast adaptation. Unlike model-based approaches, it uses similarity-based classification for few-shot learning.

**Key Features:**
- **Single forward pass**: No gradient-based adaptation needed at test time
- **Metric-based learning**: Uses embeddings and similarity for classification
- **Fast inference**: ~50ms per task vs gradient-based methods
- **Meta Dropout support**: Consistent masks across support/query sets improve performance by +1.5%

**Performance:** 77.3% ± 11.9% accuracy on 5-way 1-shot Omniglot with Meta Dropout

**Documentation:** See [Meta Networks Overview](docs/META_NETWORKS_OVERVIEW.md) and [Meta Dropout in Meta Networks](docs/META_DROPOUT_IN_META_NETWORKS.md)

**Note:** This is the Embedding-based variant (Metric-based Meta Learning). The original Meta Networks (Model-based) will be added next.

### 🚧 Coming Soon
- Original Meta Networks (Model-based variant)
- Prototypical Networks
- Matching Networks
- Reptile

## 🗂️ Repository Structure

```
meta-learning-from-scratch/
├── MAML.py                      # Core MAML & FOMAML algorithm implementation
├── EB_Meta_Network.py           # Embedding-based Meta Networks implementation
├── SimpleConvNet.py             # CNN model with Meta Dropout support
├── Meta_Dropout.py              # Meta Dropout layer implementation
├── evaluate_maml.py             # MAML-specific evaluation functions
├── test_meta_dropout.py         # Meta Dropout test suite
├── test_meta_network_dropout.py # Meta Networks with Meta Dropout tests
├── utils/
│   ├── load_omniglot.py         # Dataset loaders (easily adaptable)
│   ├── evaluate.py              # Algorithm-agnostic evaluation utilities
│   └── visualize_omniglot.py    # Dataset visualization tools
├── docs/
│   ├── MAML_vs_FOMAML.md        # MAML and FOMAML comparison
│   ├── META_DROPOUT_USAGE.md    # Meta Dropout usage guide
│   ├── META_NETWORKS_OVERVIEW.md        # Meta Networks architecture guide
│   └── META_DROPOUT_IN_META_NETWORKS.md # Meta Dropout integration details
├── examples/
│   ├── maml_on_omniglot.ipynb   # Complete MAML tutorial notebook
│   ├── embedding_based_meta_network.ipynb # Meta Networks tutorial
│   └── compare_maml_fomaml.py   # MAML vs FOMAML comparison script
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/meta-learning-from-scratch.git
cd meta-learning-from-scratch

# Install dependencies
pip install torch torchvision numpy matplotlib pillow tqdm
```

### Download Omniglot Dataset

```bash
# Download from official source
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip

# Extract
unzip images_background.zip -d omniglot/
unzip images_evaluation.zip -d omniglot/
```

### Basic Usage

#### Training MAML/FOMAML

```python
import torch
from MAML import train_maml, ModelAgnosticMetaLearning
from SimpleConvNet import SimpleConvNet
from utils.load_omniglot import OmniglotDataset, OmniglotTaskDataset
from utils.evaluate import plot_evaluation_results, plot_training_progress
from torch.utils.data import DataLoader

# 1. Load your dataset
dataset = OmniglotDataset("omniglot/images_background")

# 2. Create task dataset (5-way 1-shot)
task_dataset = OmniglotTaskDataset(
    dataset, 
    n_way=5, 
    k_shot=1, 
    k_query=15, 
    num_tasks=2000
)

task_dataloader = DataLoader(task_dataset, batch_size=4, shuffle=True)

# 3. Define your model (with optional Meta Dropout)
model = SimpleConvNet(
    num_classes=5,
    dropout_rates=[0.05, 0.10, 0.15, 0.05]  # Validated configuration
)

# 4. Train with MAML or FOMAML
model, maml, losses = train_maml(
    model=model,
    task_dataloader=task_dataloader,
    inner_lr=0.01,          # Task adaptation learning rate
    outer_lr=0.001,         # Meta-learning rate
    inner_steps=5,          # Adaptation steps per task
    first_order=False       # Set True for FOMAML
)

# 5. Evaluate on test tasks
eval_results = evaluate_maml(model, maml, test_dataloader, num_classes=5)
plot_evaluation_results(eval_results)
```

#### Training Embedding-based Meta Networks

```python
from EB_Meta_Network import MetaNetwork
from utils.evaluate import evaluate_model

# 1. Setup dataset (same as above)
# 2. Define Meta Network with Meta Dropout
meta_model = MetaNetwork(
    embedding_dim=64,
    hidden_dim=128,
    num_classes=5,
    dropout_rates=[0.05, 0.10, 0.15],  # Meta Dropout rates
    dropout_query=True  # Apply dropout to query embeddings
)

# 3. Train Meta Network (standard PyTorch training loop)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in task_dataloader:
        support_data, support_labels, query_data, query_labels = batch
        
        # Forward pass
        logits = meta_model(support_data, support_labels, query_data)
        loss = criterion(logits, query_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Evaluate (single forward pass, no adaptation needed)
accuracy = evaluate_model(meta_model, test_dataloader)
```

## 📖 Module Documentation

### Core Algorithm Implementations

**`MAML.py`** - Model-Agnostic Meta-Learning
- Complete MAML & FOMAML implementation with inner/outer loop optimization
- Supports first-order approximation for memory efficiency
- GPU optimized with gradient clipping and batch processing

**`EB_Meta_Network.py`** - Embedding-based Meta Networks
- Metric-based meta-learning with single forward pass inference
- EmbeddingNetwork (CNN) + MetaLearner (U, V, e matrices) architecture
- Integrated Meta Dropout support with automatic mask management

**`SimpleConvNet.py`** - CNN Model with Meta Dropout
- 4-layer convolutional network optimized for few-shot learning
- Meta Dropout integration with context manager API
- Compatible with both MAML and Meta Networks

**`Meta_Dropout.py`** - Consistent Dropout for Meta-Learning
- Maintains consistent dropout masks during task adaptation
- Batch-size agnostic with automatic broadcasting
- Significant performance improvements across algorithms

### Evaluation and Utilities

**`evaluate_maml.py`** - MAML-specific evaluation with before/after adaptation metrics

**`utils/evaluate.py`** - Algorithm-agnostic visualization and analysis tools

**`utils/load_omniglot.py`** - Dataset loading with configurable N-way K-shot task generation

**`utils/visualize_omniglot.py`** - Dataset exploration and task visualization

### Testing

**`test_meta_dropout.py`** - Meta Dropout functionality tests

**`test_meta_network_dropout.py`** - Meta Networks with Meta Dropout integration tests

## 🎓 Tutorial Notebooks

**`maml_on_omniglot.ipynb`** - Complete MAML walkthrough:
1. Dataset exploration and task visualization
2. Model architecture and Meta Dropout integration
3. MAML training with step-by-step explanations
4. Evaluation and performance analysis
5. Results visualization and interpretation

**`embedding_based_meta_network.ipynb`** - Meta Networks tutorial:
1. Understanding metric-based meta-learning
2. Meta Networks architecture and fast weight generation
3. Training with Meta Dropout for improved performance
4. Comparison with MAML and analysis of results

**Perfect for**: Learning meta-learning concepts, understanding different approaches, adapting to your own problems

## 📊 Expected Results (5-way 1-shot on Omniglot)

| Metric | Value |
|--------|-------|
| Before Adaptation | 20-30% |
| After Adaptation | 60-90% |
| Improvement | 40-60% |
| Training Time (GPU) | 10-30 min |

## 🔧 Adapting to Your Dataset

The modular design makes it easy to use your own dataset:

1. **Create your dataset class** following the `OmniglotDataset` pattern:
   ```python
   class YourDataset(Dataset):
       def __getitem__(self, idx):
           # Return (images_tensor, class_idx)
           return images, class_idx
   ```

2. **Create task dataset** following `OmniglotTaskDataset`:
   ```python
   task_dataset = YourTaskDataset(
       your_dataset,
       n_way=5,    # Number of classes per task
       k_shot=1,   # Examples per class for training
       k_query=15  # Examples per class for testing
   )
   ```

3. **Use the same training pipeline**:
   ```python
   model, maml, losses = train_maml(model, task_dataloader, ...)
   ```

That's it! The MAML algorithm and evaluation code work with any dataset following this pattern.

## 🛠️ Hyperparameter Guidelines

| Parameter | Range | Description |
|-----------|-------|-------------|
| `inner_lr` | 0.005 - 0.1 | Learning rate for task adaptation (inner loop) |
| `outer_lr` | 0.0001 - 0.01 | Meta-learning rate (outer loop) |
| `inner_steps` | 1 - 10 | Number of gradient steps per task adaptation |
| `batch_size` | 2 - 32 | Number of tasks per meta-update |
| `n_way` | 5 - 20 | Number of classes per task |
| `k_shot` | 1 - 5 | Training examples per class |

**Tips**:
- Start with defaults: `inner_lr=0.01`, `outer_lr=0.001`, `inner_steps=5`
- Higher `inner_lr` → faster adaptation but potentially unstable
- More `inner_steps` → better task performance but slower training
- Larger `batch_size` → more stable meta-gradients but more memory

## 📝 Requirements

```
python >= 3.8
torch >= 1.12.0
torchvision >= 0.13.0
numpy >= 1.21.0
matplotlib >= 3.5.0
pillow >= 9.0.0
tqdm >= 4.64.0
```

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Open issues for bugs or questions
- Suggest new meta-learning algorithms to implement
- Share your own experiments and results

## 📚 References

- [MAML Paper](https://arxiv.org/abs/1703.03400) - Finn et al., 2017
- [Omniglot Dataset](https://github.com/brendenlake/omniglot) - Lake et al., 2015
- [Stanford CS330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/)
- Book: "Meta-Learning: Theory, Algorithms and Applications"

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Chelsea Finn and team for the MAML algorithm
- Brenden Lake for the Omniglot dataset
- The meta-learning research community

---

**Made with ❤️ while learning meta-learning from scratch**

*If you find this helpful for your learning journey, please ⭐ star the repo!*
