# 🧠 Meta-Learning From Scratch

A modular collection of meta-learning algorithm implementations built from scratch. Designed for clarity, flexibility, and easy experimentation — plug in your own datasets, define tasks, and explore how models learn to learn.

> 🎓 **Learning Journey**: This repository is part of my meta-learning exploration as a second-year undergraduate student at IIT Guwahati. I'm learning meta-learning through multiple sources, primarily following the book **"Meta-Learning: Theory, Algorithms and Applications"**. I'm building these implementations from the ground up to deeply understand how models can "learn to learn."

## 🌟 Overview

Meta-learning (or "learning to learn") enables models to quickly adapt to new tasks with minimal training data. Unlike traditional machine learning that learns a specific task, meta-learning algorithms learn how to efficiently learn new tasks.

**Current Status**: ✅ MAML Implementation Complete | 🚧 More algorithms coming soon!

## 🎯 Features

- **Modular Design**: Clean, reusable components that work across different datasets and tasks
- **Easy Experimentation**: Plug in your own datasets, customize task sampling, and tune hyperparameters
- **Well-Documented**: Comprehensive docstrings, inline comments, and tutorial notebook
- **Research-Ready**: Built for both learning and experimentation

## 📚 Algorithms Implemented

### ✅ MAML (Model-Agnostic Meta-Learning)
*Status: Complete and tested on Omniglot*

**Paper**: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) (Finn et al., 2017)

MAML learns an initialization of model parameters that enables rapid adaptation to new tasks with just a few gradient steps. The algorithm uses bi-level optimization with inner loop (task adaptation) and outer loop (meta-learning).

**Key Features**:
- Second-order gradient computation through inner loop
- Supports any PyTorch model architecture
- GPU-optimized training with mixed precision support
- Comprehensive evaluation metrics

### 🚧 Coming Soon
- First-Order MAML (FOMAML)
- Reptile
- Prototypical Networks
- Matching Networks
- ANIL (Almost No Inner Loop)

## 🗂️ Repository Structure

```
meta-learning-from-scratch/
├── MAML.py                      # Core MAML algorithm implementation
├── evaluate_maml.py             # MAML-specific evaluation functions
├── load_omniglot.py             # Dataset loaders (easily adaptable)
├── utils/
│   ├── evaluate.py              # Algorithm-agnostic visualization utilities
│   └── visualize_omniglot.py    # Dataset visualization tools
├── maml_on_omniglot.ipynb       # Complete tutorial notebook
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
from load_omniglot import OmniglotDataset, OmniglotTaskDataset
from evaluate_maml import evaluate_maml
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

# 3. Define your model
model = YourNeuralNetwork(num_classes=5)

# 4. Train with MAML
model, maml, losses = train_maml(
    model=model,
    task_dataloader=task_dataloader,
    inner_lr=0.01,      # Task adaptation learning rate
    outer_lr=0.001,     # Meta-learning rate
    inner_steps=5       # Adaptation steps per task
)

# 5. Evaluate on test tasks
eval_results = evaluate_maml(model, maml, test_dataloader, num_classes=5)
plot_evaluation_results(eval_results)
```

## 📖 Module Documentation

### `MAML.py`

**Core Classes**:
- `ModelAgnosticMetaLearning`: Complete MAML implementation with inner/outer loop optimization
  - `inner_update(support_data, support_labels)`: Adapt model to a task using support set
  - `forward_with_weights(x, weights)`: Forward pass with custom parameter values
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
