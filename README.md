# 🧠 Meta-Learning From Scratch

A modular collection of meta-learning algorithm implementations built from scratch. Designed for clarity, flexibility, and easy experimentation — plug in your own datasets, define tasks, and explore how models learn to learn.

> 🎓 **Learning Journey**: This repository is part of my meta-learning exploration as a second-year undergraduate student at IIT Guwahati. I'm learning meta-learning through multiple sources, primarily following the book **"Meta-Learning: Theory, Algorithms and Applications"**. I'm building these implementations from the ground up to deeply understand how models can "learn to learn."

## 🌟 Overview

Meta-learning (or "learning to learn") enables models to quickly adapt to new tasks with minimal training data. Unlike traditional machine learning that learns a specific task, meta-learning algorithms learn how to efficiently learn new tasks.

**Current Status**: ✅ MAML & FOMAML Complete | ✅ Meta Dropout Implemented | 🚧 More algorithms coming soon!

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

### 🚧 Coming Soon
- Prototypical Networks
- Matching Networks
- Reptile

## 🗂️ Repository Structure

```
meta-learning-from-scratch/
├── MAML.py                      # Core MAML & FOMAML algorithm implementation
├── SimpleConvNet.py             # CNN model with Meta Dropout support
├── Meta_Dropout.py              # Meta Dropout layer implementation
├── evaluate_maml.py             # MAML-specific evaluation functions
├── test_meta_dropout.py         # Meta Dropout test suite
├── utils/
│   ├── load_omniglot.py         # Dataset loaders (easily adaptable)
│   ├── evaluate.py              # Algorithm-agnostic evaluation utilities
│   └── visualize_omniglot.py    # Dataset visualization tools
├── docs/
│   ├── MAML_vs_FOMAML.md        # MAML and FOMAML comparison
│   └── META_DROPOUT_USAGE.md    # Meta Dropout usage guide
├── examples/
│   ├── maml_on_omniglot.ipynb   # Complete tutorial notebook
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

## 📖 Module Documentation

### `MAML.py`

**Core Classes**:
- `ModelAgnosticMetaLearning`: Complete MAML & FOMAML implementation with inner/outer loop optimization
  - `inner_update(support_data, support_labels)`: Adapt model to a task using support set
  - `forward_with_weights(x, weights)`: Forward pass with custom parameter values
  - Supports `first_order` flag for FOMAML variant

### `SimpleConvNet.py`

**Core Classes**:
- `SimpleConvNet`: CNN model with Meta Dropout support
  - 4 convolutional blocks with optional dropout
  - `outer_loop_mode()`: Context manager for optimized dropout control
  - Automatically detected and used by MAML during meta-training

### `Meta_Dropout.py`

**Core Classes**:
- `MetaDropout`: Dropout layer that maintains consistent masks during task adaptation
  - `reset_mask()`: Generate new dropout mask (called between tasks)
  - Batch-size agnostic with automatic broadcasting
  - See [META_DROPOUT_USAGE.md](docs/META_DROPOUT_USAGE.md) for details
  - `meta_train_step(support_batch, query_batch)`: Single meta-training step on task batch

**Training Function**:
- `train_maml(model, task_dataloader, ...)`: High-level training pipeline with progress tracking and loss visualization

**Key Features**:
- Second-order gradient computation for true MAML
- Gradient clipping for stability
- GPU memory optimized batch processing
- Compatible with any PyTorch model

### `evaluate_maml.py`

**MAML-Specific Evaluation**:
- `evaluate_maml(model, maml, eval_dataloader, num_classes)`: Comprehensive MAML evaluation on test tasks
  - Measures accuracy before and after adaptation
  - Returns: accuracy metrics, improvement, per-task statistics, loss values
  - Computes baseline and random baseline for comparison

**Features**:
- MAML-specific inner loop adaptation
- Detailed performance metrics
- Statistical analysis (mean, std, improvement)
- Progress tracking with tqdm

### `utils/evaluate.py`

**Algorithm-Agnostic Visualization Functions**:

These functions work with **any meta-learning algorithm** (MAML, Reptile, Prototypical Networks, etc.):

- `plot_evaluation_results(eval_results)`: Generate 4-panel evaluation visualization
  - Before vs After adaptation comparison
  - Accuracy distributions
  - Per-task improvement histogram
  - Loss vs Accuracy correlation
  
- `plot_training_progress(losses, window_size)`: Training loss curves with smoothing
  - Raw and smoothed loss curves
  - Loss distribution histogram
  - Training statistics summary

**Features**:
- Works with any meta-learning algorithm
- Professional publication-quality plots
- Customizable figure sizes
- Automatic statistics computation

### `load_omniglot.py`

**Dataset Classes**:
- `OmniglotDataset(data_path)`: Load Omniglot character classes
  - Returns character images and class indices
  - Supports both background and evaluation splits
  
- `OmniglotTaskDataset(omniglot_dataset, n_way, k_shot, k_query, num_tasks)`: Generate N-way K-shot tasks
  - Randomly samples character classes
  - Splits into support (training) and query (test) sets
  - Configurable task difficulty

**Key Features**:
- Easy to adapt for other datasets (just follow the same structure)
- Automatic task sampling
- Consistent support/query split

### `utils/visualize_omniglot.py`

**Visualization Functions**:
- `visualize_task_sample(task_dataset, task_idx)`: Display a complete task with support and query sets
- `visualize_character_variations(dataset, num_chars, max_examples)`: Show handwriting variations within classes

**Features**:
- Clear task structure visualization
- Educational plots for understanding few-shot learning
- Customizable display parameters

## 🎓 Tutorial Notebook

The `maml_on_omniglot.ipynb` notebook provides a complete walkthrough:

1. **Dataset Exploration**: Visualize Omniglot characters and task structure
2. **Model Architecture**: Simple ConvNet implementation
3. **MAML Training**: Step-by-step training with explanations
4. **Evaluation**: Performance analysis on unseen character classes
5. **Results Visualization**: Comprehensive plots and metrics

**Perfect for**: Learning meta-learning concepts, understanding MAML, adapting to your own problems

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
