# 🧠 Meta-Learning From Scratch

A modular collection of meta-learning algorithm implementations built from scratch. Designed for clarity, flexibility, and easy experimentation — plug in your own datasets, define tasks, and explore how models learn to learn.

> 🎓 **Learning Journey**: This repository is part of my meta-learning exploration as a second-year undergraduate student at IIT Guwahati. I'm learning meta-learning through multiple sources, primarily following the book **"Meta-Learning: Theory, Algorithms and Applications"**. I'm building these implementations from the ground up to deeply understand how models can "learn to learn."

## 🌟 Overview

Meta-learning (or "learning to learn") enables models to quickly adapt to new tasks with minimal training data. Unlike traditional machine learning that learns a specific task, meta-learning algorithms learn how to efficiently learn new tasks.

**Current Status**: ✅ MAML & FOMAML Complete | ✅ MAML++ Complete | ✅ Meta-SGD Complete | ✅ ANIL Complete | ✅ Both Meta Networks Variants Complete | ✅ Meta Dropout Implemented | ✅ LEO Complete | 🚧 More algorithms coming soon!

## 🎯 Features

- **Modular Design**: Clean, reusable components that work across different datasets and tasks
- **Easy Experimentation**: Plug in your own datasets, customize task sampling, and tune hyperparameters
- **Meta Dropout Support**: Consistent dropout masks for improved few-shot learning performance across all algorithms
- **Shared Components**: EmbeddingNetwork shared across implementations for fair comparisons
- **⚡ Prefetched Dataset**: Load entire Omniglot into RAM for 10-50x faster data access (~300MB overhead)
- **Well-Documented**: Comprehensive docstrings, inline comments, and tutorial notebooks
- **Research-Ready**: Built for both learning and experimentation

## 📚 Algorithms Implemented

### ✅ Model-Agnostic Meta-Learning (MAML)
A flexible meta-learning approach that trains a model's initial parameters to enable rapid adaptation to new tasks with just a few gradient steps.

**Key Features:**
- Works with any gradient-based model
- Learns optimal parameter initialization
- Adapts quickly with minimal data
- **Meta Dropout**: +1.2% accuracy, -8.9% variance
- [Original Paper](https://arxiv.org/abs/1703.03400)

**Documentation:** See [MAML vs FOMAML Comparison](docs/MAML_vs_FOMAML_vs_MAMLpp.md)

### ✅ Meta-SGD (Meta Stochastic Gradient Descent)
An extension of MAML that learns **per-parameter learning rates** during meta-training, allowing each weight to have its own personalized adaptation speed.

**Key Features:**
- **Personalized learning rates**: Each parameter learns its own optimal learning rate
- **Better adaptation**: Parameters converge more efficiently with tailored step sizes
- **Double the parameters**: Meta-learns both θ (weights) and α (learning rates)
- **Second-order only**: Requires full MAML (incompatible with first-order approximation)
- **79.5% accuracy**: Achieves strong performance on Omniglot 5-way 1-shot
- [Original Paper](https://arxiv.org/abs/1707.09835)

**Performance (5-way 1-shot Omniglot):**
- Test Accuracy: 79.5%
- Parameters: 2× MAML (learns both weights and learning rates)
- Training: Requires second-order gradients (create_graph=True)

**Key Insights:**
- Learning rates must be tensor-shaped (same shape as parameters)
- Cannot use first-order approximation (needs full computation graph)
- Learns which parameters need aggressive vs careful updates
- Particularly effective when different layers need different adaptation speeds

**Documentation:** See [Meta-SGD Guide](docs/META_SGD.md)

### ✅ First-Order MAML (FOMAML)
A memory-efficient variant of MAML that omits second-order derivatives, offering:
- **Faster training**: ~40% faster than full MAML
- **Lower memory usage**: No need to store computation graph for second derivatives
- **Comparable performance**: Often matches MAML accuracy with reduced computational cost

**Documentation:** See [MAML vs FOMAML Comparison](docs/MAML_vs_FOMAML_vs_MAMLpp.md)

### ✅ ANIL (Almost No Inner Loop)
A simplified and faster variant of MAML that only adapts the head (final layer) during the inner loop while keeping the body (feature extractor) frozen. Achieves 3-10x speedup with minimal accuracy loss.

**Key Features:**
- **3-10x faster than MAML**: Only adapts head during inner loop
- **Multiple variants**: 4 training scenarios (second-order, first-order, pretrained trainable, pretrained frozen)
- **Transfer learning ready**: Excellent for pretrained models (ResNet, VGG)
- **Best generalization**: Frozen body variant achieves 90.5% accuracy on Omniglot
- **BatchNorm adaptation**: Critical for domain adaptation with frozen features
- **Prevents meta-overfitting**: Parameter-efficient design avoids task memorization
- [Original Paper](https://arxiv.org/abs/1909.09157)

**Performance (5-way 1-shot Omniglot):**
- Scenario 1 (Second-Order): 77.12% accuracy
- Scenario 2 (First-Order): 77.19% accuracy (1.47x faster)
- Scenario 3 (Pretrained Trainable): 72.45% accuracy (⚠️ meta-overfitting)
- Scenario 4 (Pretrained Frozen): **90.45% accuracy** (best!)

**Key Insights:**
- Lower training loss ≠ better performance (Scenario 3 paradox)
- BatchNorm training essential for frozen body (enables domain adaptation)
- Parameter-to-task ratio matters: <100 good, >1000 dangerous

**Documentation:** See [ANIL Guide](docs/ANIL.md) for comprehensive explanation and comparison of all 4 scenarios

### ✅ Meta Dropout
An optimized dropout strategy for meta-learning that maintains consistent dropout masks during task adaptation:
- **5x speedup**: Optimized boolean flag implementation (no context manager overhead)
- **Universal improvement**: Benefits all meta-learning algorithms
- **Best results**: +2.16% accuracy with Original Meta Networks

**Documentation:** See [Meta Dropout Usage Guide](docs/META_DROPOUT_USAGE.md)

### ✅ Embedding-based Meta Networks
A metric-based meta-learning approach that generates task-specific embeddings for fast adaptation. Unlike model-based approaches, it uses similarity-based classification for few-shot learning.

**Key Features:**
- **Single forward pass**: No gradient-based adaptation needed at test time
- **Metric-based learning**: Uses embeddings and similarity for classification
- **Fast inference**: ~50ms per task vs gradient-based methods
- **Meta Dropout support**: +1.5% accuracy improvement

**Performance:** 77.3% ± 11.9% accuracy on 5-way 1-shot Omniglot with Meta Dropout

**Documentation:** See [Meta Networks Overview](docs/META_NETWORKS_OVERVIEW.md) and [Meta Dropout in EB Meta Networks](docs/META_DROPOUT_IN_EB_META_NETWORKS.md)

**Note:** This is the Embedding-based variant (Metric-based Meta Learning).

### ✅ Original Meta Networks
The true implementation of Meta Networks from the original paper (Munkhdalai & Yu, 2017). A model-based meta-learning approach where one neural network learns to generate the parameters of another neural network.

**Key Features:**
- **Weight prediction**: Meta-learner generates actual FC layer weights W and biases b
- **Model-based learning**: One model predicts another model's parameters
- **Single forward pass**: Direct parameter generation, no adaptation loop
- **Best Meta Dropout results**: +2.16% accuracy, -11.7% variance (winner!)
- **Highest accuracy**: 86.31% ± 9.07% on 5-way 1-shot Omniglot
- [Original Paper](https://arxiv.org/abs/1703.00837)

**Performance:** 86.31% ± 9.07% accuracy on 5-way 1-shot Omniglot with Meta Dropout

**Architecture:**
- **U, V matrices** and **e vector**: Core meta-learner parameters
- **Weight generator**: Predicts W [embedding_dim × num_classes]
- **Bias generator**: Predicts b [num_classes]
- **Shared EmbeddingNetwork**: Same CNN as Embedding-based variant

**Documentation:** See [Original Meta Networks Overview](docs/ORIGINAL_META_NETWORK_OVERVIEW.md) and [Meta Dropout in Meta Networks](docs/META_DROPOUT_IN_META_NETWORKS.md)

**Note:** This is the original Meta Networks (Model-based Meta Learning) - the true paper implementation.

### ✅ LEO (Latent Embedding Optimization)
A state-of-the-art meta-learning algorithm that learns to adapt to new tasks by optimizing in a learned **low-dimensional latent space** rather than directly in the high-dimensional parameter space.

**Key Features:**
- **Low-dimensional optimization**: Operates in 64D latent space instead of 100K+ parameter space
- **Encoder-Decoder architecture**: Encoder maps data to latent codes, decoder generates parameters
- **Excellent 1-shot performance**: 95-98% accuracy on Omniglot 5-way 1-shot
- **Efficient**: ~30% faster training and 40% less memory than MAML
- **Parameter generation**: Decoder generates all 112K model parameters from 64D latent codes
- **Smooth optimization**: Learned latent space provides better optimization landscape
- [Original Paper](https://arxiv.org/abs/1807.05960)

**Performance (5-way 1-shot Omniglot):**
- Test Accuracy: 95-98%
- Training Time: ~15-25 minutes (vs 20-30 for MAML)
- GPU Memory: 4-6 GB (vs 6-8 GB for MAML)
- Parameters: 64D latent + encoder/decoder (~50K) generates 112K classifier params

**Architecture:**
- **LEOEncoder**: CNN that maps images to 64D latent codes
- **LEODecoder**: MLP that generates 112K parameters from latent codes
- **LEORelationNetwork**: Processes pairs of examples for task structure
- **LEOClassifier**: CNN that uses generated parameters

**Key Insights:**
- Latent space optimization much easier than parameter space
- Decoder learns to generate good parameter initializations
- Particularly effective for very few-shot scenarios (1-shot)
- Lower memory footprint enables larger models

**Documentation:** See [LEO Guide](docs/LEO.md), [LEO README](LEO_README.md), and [MAML vs LEO Comparison](docs/MAML_vs_LEO.md)

### ✅ TAML (Task Agnostic Meta Learning)
A meta-learning algorithm designed to prevent overfitting to meta-training tasks by learning task-agnostic initializations and adaptation mechanisms. TAML uses a shared CNN feature extractor and adapts only the final layers for each task, leveraging an encoder, relation network, and decoder to process support examples and generate modulations and learning rates for adaptation.

**Key Features:**
- **Task-agnostic adaptation**: Prevents overfitting to meta-training tasks
- **Efficient inner loop**: Only adapts head parameters, not the full network
- **Relation network**: Refines class codes using cross-class relationships
- **Flexible modulations**: Decoder can output global or per-layer scaling factors
- **Shared feature extractor**: CNN backbone is meta-learned and shared
- [Original Paper](https://arxiv.org/abs/1905.03684)

**Performance (5-way 1-shot Omniglot):**
- Speed: ~3.3s/iteration
- Final Loss: < 0.6
- Evaluation Accuracy: ~73% (std: 14%)

**Key Insights:**
- Current results show good final loss but moderate evaluation accuracy, indicating some meta-overfitting
- Further optimization and hyperparameter tuning are needed to match or surpass state-of-the-art algorithms
- Vectorization of some steps could improve speed and efficiency

**Documentation:** See [TAML Guide](docs/TAML.md)
### 🚧 Coming Soon
- Prototypical Networks
- CAVIA

## 🗂️ Repository Structure

```
meta-learning-from-scratch/
├── algorithms/                  # Core algorithm implementations
│   ├── maml.py                  # MAML, FOMAML & Meta-SGD implementation
│   ├── anil.py                  # ANIL implementation (4 training scenarios)
│   ├── leo.py                  # LEO implementation
│   ├── taml.py                  # TAML implementation
│   ├── eb_meta_network.py       # Embedding-based Meta Networks
│   ├── original_meta_network.py # Original Meta Networks (model-based)
│   ├── embedding_network.py     # Shared CNN feature extractor
│   ├── cnn_maml.py              # CNN model with Meta Dropout support
│   └── meta_dropout.py          # Meta Dropout layer implementation
├── evaluation/                  # Evaluation and visualization tools
│   ├── evaluate_maml.py         # MAML-specific evaluation functions
│   ├── evaluate_anil.py         # ANIL-specific evaluation functions
│   └── eval_visualization.py    # Plotting and analysis utilities
├── tests/                       # Test suites
│   ├── test_meta_dropout.py     # Meta Dropout functionality tests
│   └── test_meta_network_dropout.py # Meta Networks integration tests
├── utils/                       # Dataset utilities
│   ├── load_omniglot.py         # Dataset loaders (standard + prefetched)
│   └── visualize_omniglot.py    # Dataset visualization tools
├── docs/                        # Documentation
│   ├── MAML_vs_FOMAML_vs_MAMLpp.md # MAML, FOMAML, and Meta-SGD comparison
│   ├── META_SGD.md              # Meta-SGD comprehensive guide
│   ├── ANIL.md                  # ANIL comprehensive guide (4 scenarios)
│   ├── LEO.md                   # LEO comprehensive guide
│   ├── TAML.md                  # TAML comprehensive guide
│   ├── META_DROPOUT_USAGE.md    # Meta Dropout usage guide
│   ├── PREFETCHED_DATASET.md    # Prefetched dataset guide (10-50x faster!)
│   ├── META_NETWORKS_OVERVIEW.md      # Embedding-based Meta Networks guide
│   ├── ORIGINAL_META_NETWORK_OVERVIEW.md # Original Meta Networks guide
│   └── META_DROPOUT_IN_META_NETWORKS.md # Meta Dropout integration (both variants)
├── examples/                    # Tutorial notebooks and scripts
│   ├── maml_on_omniglot.ipynb   # Complete MAML & Meta-SGD tutorial notebook
│   ├── anil_on_omniglot.ipynb   # ANIL tutorial (4 training scenarios)
│   ├── embedding_based_meta_network.ipynb # Embedding-based Meta Networks tutorial
│   ├── meta_network.ipynb       # Original Meta Networks tutorial
│   └── compare_maml_fomaml.py   # MAML vs FOMAML comparison script
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/meta-learning-from-scratch.git
cd meta-learning-from-scratch

# Install dependencies (recommended)
pip install -r requirements.txt

# Or install manually
pip install torch numpy matplotlib pillow tqdm
```

> **Note:** The `requirements.txt` file specifies minimum tested versions that are confirmed to work with this codebase. You can safely use newer versions of these packages.

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

#### Option 1: Standard Dataset (Load on-demand)

```python
import torch
from algorithms.maml import train_maml, ModelAgnosticMetaLearning
from algorithms.cnn_maml import SimpleConvNet
from utils.load_omniglot import OmniglotDataset, OmniglotTaskDataset
from evaluation.eval_visualization import plot_evaluation_results, plot_training_progress
from torch.utils.data import DataLoader

# 1. Load your dataset (standard - loads from disk)
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
```

#### Option 2: Prefetched Dataset (⚡ 10-50x Faster!)

```python
from utils.load_omniglot import PrefetchedOmniglotDataset, OmniglotTaskDataset
from torch.utils.data import DataLoader

# 1. Load entire dataset into RAM (one-time ~20-30s loading)
dataset = PrefetchedOmniglotDataset("omniglot/images_background")
# Memory usage: ~200-300 MB (entire background set in RAM)

# 2. Create task dataset (same as before)
task_dataset = OmniglotTaskDataset(
	dataset,
	n_way=5,
	k_shot=1,
	k_query=15,
	num_tasks=2000
)

# 3. DataLoader with optimized settings for prefetched data
task_dataloader = DataLoader(
	task_dataset, 
	batch_size=4, 
	shuffle=True,
	num_workers=4,  # Fewer workers needed (no I/O bottleneck)
	pin_memory=True
)

# Enjoy 10-50x faster data loading! 🚀
```

**When to use prefetched dataset:**
- ✅ Training on Omniglot (fits easily in RAM)
- ✅ Multiple experiments (faster iteration)
- ✅ You have 1-2 GB free RAM
- ❌ Very limited memory (<2 GB)

See [Prefetched Dataset Guide](docs/PREFETCHED_DATASET.md) for details.

#### Training MAML/FOMAML/Meta-SGD

```python
import torch
from algorithms.maml import train_maml, ModelAgnosticMetaLearning
from algorithms.cnn_maml import SimpleConvNet
from utils.load_omniglot import OmniglotDataset, OmniglotTaskDataset
from evaluation.eval_visualization import plot_evaluation_results, plot_training_progress
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

# 4. Train with MAML, FOMAML, or Meta-SGD
model, maml, losses = train_maml(
	model=model,
	task_dataloader=task_dataloader,
	inner_lr=0.01,  # Task adaptation learning rate
	outer_lr=0.001,  # Meta-learning rate
	inner_steps=5,  # Adaptation steps per task
	first_order=False,  # Set True for FOMAML
	meta_sgd=False  # Set True for Meta-SGD (requires first_order=False)
)

# 5. Evaluate on test tasks
eval_results = evaluate_maml(model, maml, test_dataloader, num_classes=5)
plot_evaluation_results(eval_results)
```

**Key Training Options:**
- **MAML**: `first_order=False, meta_sgd=False` (full second-order gradients)
- **FOMAML**: `first_order=True, meta_sgd=False` (faster, first-order approximation)
- **Meta-SGD**: `first_order=False, meta_sgd=True` (learnable per-parameter learning rates)

#### Training Embedding-based Meta Networks

```python
from algorithms.eb_meta_network import MetaNetwork
from evaluation.eval_visualization import evaluate_model

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

#### Training ANIL (Almost No Inner Loop)

```python
from algorithms.anil import train_anil, ANIL
from evaluation.evaluate_anil import evaluate_anil
from torchvision import models
import torch.nn as nn

# Option 1: Train from scratch (Scenarios 1 & 2)
def create_anil_network(num_classes=5):
    body = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    )
    head = nn.Sequential(nn.Flatten(), nn.Linear(2304, num_classes))
    return body, head

body, head = create_anil_network(num_classes=5)

# Train with first-order ANIL (recommended)
body, head, anil, losses = train_anil(
    body, head, task_dataloader,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5,
    first_order=True,  # 2-3x faster, minimal accuracy loss
    freeze_body=False  # Original ANIL (body trainable in outer loop)
)

# Option 2: Pretrained frozen body (Scenario 4 - Best generalization!)
resnet = models.resnet18(pretrained=True)
# Adapt for grayscale
resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
body = nn.Sequential(*list(resnet.children())[:-1])
head = nn.Linear(512, 5)

body, head, anil, losses = train_anil(
    body, head, task_dataloader,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5,
    first_order=True,
    freeze_body=True,  # Freeze body (only head + BatchNorm trainable)
    bn_warmup_batches=50  # Warm up BatchNorm statistics
)

# Evaluate ANIL
eval_results = evaluate_anil(body, head, anil, test_dataloader, num_classes=5)
print(f"Test Accuracy: {eval_results['after_adaptation_accuracy']:.1%}")
```

**ANIL Key Points:**
- 🚀 **Scenario 2 (First-Order)**: 3x faster than MAML, 77% accuracy
- 🏆 **Scenario 4 (Frozen)**: Best accuracy (90.5%), prevents meta-overfitting
- ⚠️ **Scenario 3 (Trainable Pretrained)**: Meta-overfits with limited tasks
- 🔑 **BatchNorm**: Must stay trainable with frozen body for domain adaptation

See [ANIL Guide](docs/ANIL.md) for detailed explanation of all 4 training scenarios.

## 📖 Module Documentation

### Core Algorithm Implementations

**`MAML.py`** - Model-Agnostic Meta-Learning
- Complete MAML, FOMAML & Meta-SGD implementation with inner/outer loop optimization
- Supports first-order approximation for memory efficiency
- Meta-SGD: Learnable per-parameter learning rates for personalized adaptation
- GPU optimized with gradient clipping and batch processing

**`ANIL.py`** - Almost No Inner Loop
- 3-10x faster than MAML by freezing body during inner loop adaptation
- 4 training scenarios: second-order, first-order, pretrained trainable, pretrained frozen
- BatchNorm training for domain adaptation with frozen body
- Prevents meta-overfitting with parameter-efficient design
- Integrated evaluation and visualization tools

**`EB_Meta_Network.py`** - Embedding-based Meta Networks
- Metric-based meta-learning with single forward pass inference
- EmbeddingNetwork (CNN) + MetaLearner (U, V, e matrices) architecture
- Integrated Meta Dropout support with automatic mask management

**`Original_Meta_Network.py`** - Original Meta Networks (Model-based)
- True implementation of Meta Networks from the original paper
- One neural network predicts the parameters of another neural network
- Integrated Meta Dropout support for improved performance

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

**`evaluate_anil.py`** - ANIL-specific evaluation for all 4 training scenarios

**`utils/evaluate.py`** - Algorithm-agnostic visualization and analysis tools

**`utils/load_omniglot.py`** - Dataset loading with standard and prefetched options
- **Standard**: Load images from disk on-demand
- **Prefetched**: Load entire dataset into RAM (10-50x faster, ~300MB)
- Configurable N-way K-shot task generation

**`utils/visualize_omniglot.py`** - Dataset exploration and task visualization

### Testing

**`test_meta_dropout.py`** - Meta Dropout functionality tests

**`test_meta_network_dropout.py`** - Meta Networks with Meta Dropout integration tests

## 🎓 Tutorial Notebooks

**`maml_on_omniglot.ipynb`** - Complete MAML & Meta-SGD walkthrough:
1. Dataset exploration and task visualization
2. Model architecture and Meta Dropout integration
3. MAML training with step-by-step explanations
4. Meta-SGD: Training with learnable per-parameter learning rates
5. Evaluation and performance analysis
6. Results visualization and interpretation
7. Debugging tools for analyzing learned learning rates

**`anil_on_omniglot.ipynb`** - ANIL comprehensive tutorial:
1. Introduction to ANIL and comparison with MAML
2. Network architecture (custom CNN and pretrained ResNet)
3. **4 training scenarios** with side-by-side comparison:
   - Scenario 1: Original ANIL (Second-Order)
   - Scenario 2: Original ANIL (First-Order) - Recommended
   - Scenario 3: Pretrained ANIL (Trainable Body) - Meta-overfitting example
   - Scenario 4: Pretrained ANIL (Frozen Body) - Best generalization
4. Meta-overfitting analysis and training loss vs test accuracy paradox
5. BatchNorm training importance for domain adaptation
6. Complete evaluation and visualization of all scenarios

**`embedding_based_meta_network.ipynb`** - Meta Networks tutorial:
1. Understanding metric-based meta-learning
2. Meta Networks architecture and fast weight generation
3. Training with Meta Dropout for improved performance
4. Comparison with MAML and analysis of results

**`meta_network.ipynb`** - Original Meta Networks tutorial:
1. Understanding model-based meta-learning
2. Original Meta Networks architecture and weight prediction
3. Training with Meta Dropout for best performance
4. Comparison with embedding-based variant and analysis of results

**Perfect for**: Learning meta-learning concepts, understanding different approaches, adapting to your own problems

## 📊 Expected Results (5-way 1-shot on Omniglot)

| Algorithm | Before Adaptation | After Adaptation | Training Time | Memory |
|-----------|------------------|------------------|---------------|---------|
| **MAML** | 20-30% | 75-80% | Baseline (1x) | High |
| **FOMAML** | 20-30% | 75-80% | 1.4x faster | Medium |
| **Meta-SGD** | 20-30% | **79.5%** | 1x (same as MAML) | Very High (2×) |
| **ANIL (First-Order)** | 20% | **77%** | **3x faster** | Medium |
| **ANIL (Frozen)** | 20% | **90.5%** 🏆 | **3x faster** | Medium |
| **EB Meta Networks** | - | 77% | Very Fast | Low |
| **Original Meta Networks** | - | **86%** | Very Fast | Low |

**Key Takeaways:**
- Meta-SGD achieves best gradient-based accuracy (79.5%) but doubles memory usage
- ANIL (Frozen) achieves highest overall accuracy (90.5%) with 3× speedup
- Original Meta Networks achieve 86% with single forward pass (no gradient-based adaptation)
- Improvement from random: 45-60% → 75-90%
- Training time (GPU): 3-10 minutes per algorithm

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

### Python Version
```
python >= 3.8
```

### Package Dependencies
The following packages are required (minimum tested versions):

```
torch >= 2.6.0
numpy >= 2.2.6
matplotlib >= 3.10.0
tqdm >= 4.67.1
Pillow >= 11.1.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

> 💡 **Tip:** The versions specified in `requirements.txt` are minimum versions that have been tested and confirmed to work. Feel free to use newer versions of these packages.

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Open issues for bugs or questions
- Suggest new meta-learning algorithms to implement
- Share your own experiments and results

## 📚 References

- [MAML Paper](https://arxiv.org/abs/1703.03400) - Finn et al., 2017
- [Meta-SGD Paper](https://arxiv.org/abs/1707.09835) - Li et al., 2017
- [ANIL Paper](https://arxiv.org/abs/1909.09157) - Raghu et al., ICLR 2020
- [Meta Networks Paper](https://arxiv.org/abs/1703.00837) - Munkhdalai & Yu, 2017
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
