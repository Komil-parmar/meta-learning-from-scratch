# CLAUDE.md - AI Assistant Guide

> **Purpose**: This document provides comprehensive guidance for AI assistants (like Claude) working on the meta-learning-from-scratch repository. It covers codebase structure, development workflows, conventions, and best practices.

**Last Updated**: 2025-11-14
**Repository**: [meta-learning-from-scratch](https://github.com/Komil-parmar/meta-learning-from-scratch)

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Project Architecture](#project-architecture)
3. [Codebase Structure](#codebase-structure)
4. [Algorithm Implementations](#algorithm-implementations)
5. [Development Workflows](#development-workflows)
6. [Code Conventions](#code-conventions)
7. [Testing Guidelines](#testing-guidelines)
8. [Documentation Standards](#documentation-standards)
9. [Common Patterns](#common-patterns)
10. [Working with Algorithms](#working-with-algorithms)
11. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
12. [Quick Reference](#quick-reference)

---

## Repository Overview

### Purpose
An educational repository implementing meta-learning algorithms from scratch. Built by a second-year undergraduate student at IIT Guwahati following the book "Meta-Learning: Theory, Algorithms and Applications."

### Philosophy
- **Educational First**: Code clarity over optimization
- **Modular Design**: Reusable components across algorithms
- **Well-Documented**: Comprehensive docstrings and guides
- **Research-Ready**: Real implementations with benchmarks

### Target Audience
- Students learning meta-learning
- Researchers experimenting with few-shot learning
- Practitioners adapting algorithms to custom datasets

### Current Status (as of Latest Commit)
- ✅ **Implemented**: MAML, FOMAML, Meta-SGD, ANIL, Meta Networks (2 variants), Meta Dropout, LEO, TAML, Reptile
- ✅ **Datasets**: Omniglot (with prefetched loader for 10-50x speedup)
- 🚧 **Planned**: Prototypical Networks, CAVIA

---

## Project Architecture

### High-Level Design

```
User Request (Train/Evaluate)
    ↓
Task Dataset Generation (N-way K-shot sampling)
    ↓
Algorithm Implementation (MAML/ANIL/LEO/etc.)
    ↓
Inner Loop Adaptation (Task-specific learning)
    ↓
Outer Loop Meta-Update (Meta-learning)
    ↓
Evaluation & Visualization
```

### Key Design Patterns

#### 1. **Algorithm Wrapper Pattern**
All gradient-based algorithms follow this structure:
```python
class Algorithm:
    def __init__(self, model, inner_lr, outer_lr, inner_steps, ...):
        # Store hyperparameters

    def inner_update(self, support_data, support_labels) -> Dict[str, Tensor]:
        # Adapt model to task using support set
        # Returns adapted parameters (fast_weights)

    def forward_with_weights(self, data, weights) -> Tensor:
        # Forward pass with custom weights

    def meta_train_step(self, support_batch, query_batch) -> float:
        # One meta-training iteration
        # Returns meta-loss
```

#### 2. **Dataset → Task → Batch Pipeline**
```python
Dataset (OmniglotDataset)
    → stores all character classes
    → loads images on-demand or prefetched

TaskDataset (OmniglotTaskDataset)
    → samples N-way K-shot tasks
    → returns (support_data, support_labels, query_data, query_labels)

DataLoader
    → batches multiple tasks together
    → enables parallel meta-learning
```

#### 3. **Shared Component Reuse**
- `EmbeddingNetwork`: Shared CNN used by Meta Networks, LEO, TAML
- `MetaDropout`: Universal dropout layer for all algorithms
- Evaluation functions: Algorithm-agnostic visualization

---

## Codebase Structure

### Directory Layout

```
meta-learning-from-scratch/
├── algorithms/              # Core implementations (10 files, ~5,166 lines)
│   ├── maml.py             # MAML, FOMAML, Meta-SGD (1,148 lines)
│   ├── anil.py             # ANIL with 4 training scenarios (685 lines)
│   ├── leo.py              # Latent Embedding Optimization (885 lines)
│   ├── taml.py             # Task Agnostic Meta Learning (660 lines)
│   ├── reptile_maml.py     # Reptile algorithm (456 lines)
│   ├── eb_meta_network.py  # Embedding-based Meta Networks (413 lines)
│   ├── original_meta_network.py  # Original Meta Networks (465 lines)
│   ├── embedding_network.py      # Shared CNN (115 lines)
│   ├── cnn_maml.py        # CNN with Meta Dropout (226 lines)
│   └── meta_dropout.py    # Meta Dropout layer (113 lines)
│
├── evaluation/             # Evaluation tools (3 files)
│   ├── evaluate_maml.py   # MAML evaluation (220 lines)
│   ├── evaluate_anil.py   # ANIL evaluation (145 lines)
│   └── eval_visualization.py  # Plotting utilities (300+ lines)
│
├── tests/                  # Test suites (6 files, ~1,899 lines)
│   ├── test_meta_dropout.py
│   ├── test_meta_network_dropout.py
│   ├── test_maml_plus_plus.py
│   ├── test_leo_implementation.py
│   ├── test_prefetched_dataset.py
│   └── reptile_test.py
│
├── utils/                  # Dataset utilities (2 files)
│   ├── load_omniglot.py   # Dataset loaders (250+ lines)
│   └── visualize_omniglot.py  # Visualization (150+ lines)
│
├── docs/                   # Documentation (13 markdown files)
│   ├── MAML_vs_FOMAML_vs_MAMLpp.md
│   ├── ANIL.md
│   ├── LEO.md
│   ├── META_DROPOUT_USAGE.md
│   ├── META_NETWORKS_OVERVIEW.md
│   └── ... (8 more guides)
│
├── examples/               # Tutorial notebooks (7 notebooks)
│   ├── maml_on_omniglot.ipynb
│   ├── anil_on_omniglot.ipynb
│   ├── leo_on_omniglot.ipynb
│   └── ... (4 more)
│
├── requirements.txt        # Dependencies
├── README.md              # Main documentation
├── CONTRIBUTING.md        # Contribution guidelines
└── LICENSE               # MIT License
```

### File Organization Principles

1. **Separation of Concerns**
   - `algorithms/`: Pure algorithm logic, no data loading
   - `evaluation/`: Metrics and visualization, no training logic
   - `utils/`: Data handling, no model definitions
   - `tests/`: Verification, no production code

2. **Naming Conventions**
   - **Python files**: `snake_case.py` (e.g., `meta_dropout.py`)
   - **Classes**: `PascalCase` (e.g., `ModelAgnosticMetaLearning`)
   - **Functions**: `snake_case()` (e.g., `inner_update()`)
   - **Documentation**: `UPPERCASE_WITH_UNDERSCORES.md`
   - **Notebooks**: `descriptive_name.ipynb`

3. **Module Independence**
   - Each algorithm can be imported independently
   - Shared utilities are decoupled from specific algorithms
   - Evaluation functions work with any algorithm

---

## Algorithm Implementations

### Overview Table

| Algorithm | File | Lines | Complexity | Performance (5-way 1-shot) |
|-----------|------|-------|------------|----------------------------|
| MAML/FOMAML | `maml.py` | 1,148 | High | 75-80% |
| Meta-SGD | `maml.py` | (included) | High | 79.5% |
| ANIL | `anil.py` | 685 | Medium | 77-90.5% |
| LEO | `leo.py` | 885 | High | 95-98% ⭐ |
| TAML | `taml.py` | 660 | Medium-High | ~73% |
| Reptile | `reptile_maml.py` | 456 | Medium | ~75% |
| EB Meta Networks | `eb_meta_network.py` | 413 | Medium | 77.3% |
| Original Meta Networks | `original_meta_network.py` | 465 | Medium | 86% |

### Key Algorithm Classes

#### 1. **MAML/FOMAML/Meta-SGD** (`algorithms/maml.py`)

**Main Class**: `ModelAgnosticMetaLearning`

**Key Methods**:
```python
__init__(model, inner_lr, outer_lr, inner_steps, first_order, meta_sgd)
inner_update(support_data, support_labels) -> Dict[str, Tensor]
forward_with_weights(query_data, fast_weights) -> Tensor
meta_train_step(task_batch) -> float
```

**Variants**:
- **MAML**: `first_order=False, meta_sgd=False` (full second-order)
- **FOMAML**: `first_order=True` (~40% faster, first-order approx)
- **Meta-SGD**: `meta_sgd=True` (learnable per-parameter learning rates)

**When to Use**:
- MAML: Maximum accuracy, have GPU memory
- FOMAML: Need speed, limited memory
- Meta-SGD: Different layers need different adaptation speeds

#### 2. **ANIL** (`algorithms/anil.py`)

**Main Class**: `ANIL`

**Key Features**:
- Separates network into body (feature extractor) + head (classifier)
- Only adapts head during inner loop → 3-10x faster than MAML
- 4 training scenarios with different freeze_body configurations

**4 Training Scenarios**:
1. **Second-Order Original**: Body meta-learned, head adapted (77.12%)
2. **First-Order Original**: ~1.47x faster (77.19%) ⭐ Recommended for training from scratch
3. **Pretrained Trainable**: Risk of meta-overfitting (72.45%)
4. **Pretrained Frozen**: Best generalization (90.45%) ⭐ Recommended for pretrained models

**When to Use**:
- Training from scratch → Scenario 2 (First-Order)
- Have pretrained model → Scenario 4 (Frozen)
- Need speedup over MAML → Any ANIL variant

#### 3. **LEO** (`algorithms/leo.py`)

**Main Classes**: `LEOEncoder`, `LEODecoder`, `LEORelationNetwork`, `LatentEmbeddingOptimization`

**Architecture**:
- Operates in 64D latent space instead of 100K+ parameter space
- Encoder: Maps features to latent codes
- Decoder: Generates parameters from latent codes
- Relation Network: Processes latent code relationships

**When to Use**:
- Need highest accuracy (95-98% on Omniglot)
- 1-shot learning scenarios
- Have GPU memory (4-6 GB)

#### 4. **Meta Dropout** (`algorithms/meta_dropout.py`)

**Main Class**: `MetaDropout(nn.Module)`

**Key Innovation**: Maintains consistent dropout masks throughout inner loop adaptation

**Methods**:
```python
__init__(p=0.5, inplace=False)
reset_mask(input_shape, device)  # Generate new mask for task
forward(x) -> Tensor
eval_mode() / train_mode()  # Fast enable/disable
```

**Usage Pattern**:
```python
# In SimpleConvNet or custom model
self.dropout1 = MetaDropout(p=0.05)

# At task start (in inner_update)
model.reset_dropout_masks(batch_size, device)

# During inner loop
for step in range(inner_steps):
    logits = model(support_data)  # Same masks

# During outer loop evaluation
model._outer_loop_mode = True
query_logits = model(query_data)  # Dropout disabled
model._outer_loop_mode = False
```

**Benefits**: +1.2% accuracy, -8.9% variance on MAML

---

## Development Workflows

### Adding a New Algorithm

**Step-by-Step Process**:

1. **Create Implementation** (`algorithms/your_algorithm.py`)
   ```python
   # File: algorithms/your_algorithm.py

   import torch
   import torch.nn as nn
   from typing import Dict, Tuple

   class YourAlgorithm:
       """One-line summary.

       Detailed description of algorithm, its purpose,
       and key innovations.

       Args:
           model (nn.Module): Base model to meta-learn
           inner_lr (float): Inner loop learning rate
           outer_lr (float): Outer loop learning rate

       Example:
           >>> model = SimpleConvNet(num_classes=5)
           >>> algo = YourAlgorithm(model, inner_lr=0.01)
           >>> loss = algo.meta_train_step(task_batch)
       """

       def __init__(self, model: nn.Module, inner_lr: float = 0.01):
           self.model = model
           self.inner_lr = inner_lr

       def inner_update(
           self,
           support_data: torch.Tensor,
           support_labels: torch.Tensor
       ) -> Dict[str, torch.Tensor]:
           """Adapt model to task."""
           # Implementation
           pass

       def meta_train_step(self, task_batch: Tuple) -> float:
           """One meta-training iteration."""
           # Implementation
           pass
   ```

2. **Create Tests** (`tests/test_your_algorithm.py`)
   ```python
   import pytest
   import torch
   from algorithms.your_algorithm import YourAlgorithm

   class TestYourAlgorithm:
       def test_initialization(self):
           """Test algorithm initializes correctly."""
           model = SimpleConvNet(num_classes=5)
           algo = YourAlgorithm(model, inner_lr=0.01)
           assert algo.inner_lr == 0.01

       def test_inner_update(self):
           """Test inner loop adaptation."""
           # Create dummy data
           # Run inner_update
           # Verify parameters changed
           pass
   ```

3. **Create Documentation** (`docs/YOUR_ALGORITHM_OVERVIEW.md`)
   ```markdown
   # Your Algorithm Overview

   ## What is Your Algorithm?
   Clear explanation

   ## Architecture
   Detailed breakdown

   ## How It Works
   Step-by-step flow

   ## Performance
   Benchmarks

   ## Usage Example
   Complete code

   ## References
   - [Paper](url) - Authors, Year
   ```

4. **Create Tutorial** (`examples/your_algorithm_on_omniglot.ipynb`)
   - Dataset setup
   - Training walkthrough
   - Evaluation
   - Comparison with other algorithms

5. **Update Integration Files**
   - Add to `README.md` algorithms list
   - Update performance comparison tables
   - Test with existing utilities

### Modifying Existing Code

**Critical Rules**:
1. ✅ **Maintain Backward Compatibility**: Existing code must still work
2. ✅ **Update All Tests**: Verify nothing breaks
3. ✅ **Update Documentation**: Keep docs in sync with code
4. ✅ **Test Notebooks**: Execute affected notebooks to verify

**Verification Checklist**:
```bash
# 1. Run all tests
python -m pytest tests/

# 2. Run specific tests for modified file
python -m pytest tests/test_your_modified_file.py -v

# 3. Test files that import your changes
# Example: If you modified meta_dropout.py
python -m pytest tests/test_meta_network_dropout.py

# 4. Execute affected notebooks
jupyter nbconvert --execute examples/maml_on_omniglot.ipynb

# 5. Check for import errors
python -c "from algorithms.your_modified_file import YourClass"
```

### Bug Fixing Workflow

1. **Reproduce Bug**
   - Create minimal reproducible example
   - Document expected vs actual behavior

2. **Write Test**
   - Add test that fails with bug present
   - Test should pass after fix

3. **Fix Bug**
   - Make minimal changes to fix issue
   - Avoid scope creep

4. **Verify Fix**
   - Run new test (should pass)
   - Run all existing tests
   - Test notebooks if applicable

5. **Document Fix**
   - Update relevant documentation
   - Add comments explaining fix if non-obvious

---

## Code Conventions

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Files | `snake_case.py` | `meta_dropout.py` |
| Classes | `PascalCase` | `ModelAgnosticMetaLearning` |
| Functions | `snake_case()` | `inner_update()` |
| Variables | `snake_case` | `fast_weights` |
| Constants | `UPPERCASE_WITH_UNDERSCORES` | `DEFAULT_LEARNING_RATE` |
| Private | `_prefix` | `_outer_loop_mode` |
| Docs | `UPPERCASE.md` | `META_DROPOUT_USAGE.md` |
| Notebooks | `descriptive_name.ipynb` | `maml_on_omniglot.ipynb` |

### Import Organization

**Always organize imports in 3 sections**:
```python
# 1. Standard library
import os
import time
from typing import Dict, List, Tuple, Optional

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 3. Local/Project
from algorithms.meta_dropout import MetaDropout
from utils.load_omniglot import OmniglotDataset
```

### Type Hints

**Required on all functions**:
```python
def train_maml(
    model: nn.Module,
    task_dataloader: DataLoader,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    first_order: bool = False
) -> Tuple[nn.Module, ModelAgnosticMetaLearning, List[float]]:
    """Train model using MAML."""
    pass
```

### Docstring Format (Google Style)

**Complete docstring template**:
```python
def function_name(
    param1: Type1,
    param2: Type2 = default_value
) -> ReturnType:
    """One-line summary of what function does.

    More detailed description explaining the purpose,
    algorithm, or important implementation details.
    Can span multiple paragraphs.

    Args:
        param1 (Type1): Description of param1.
            Shape: [batch, channels, height, width]
        param2 (Type2, optional): Description of param2.
            Default: default_value

    Returns:
        ReturnType: Description of return value.
            For tuples, describe each element:
            - element1: Description
            - element2: Description

    Raises:
        ValueError: When this specific condition occurs.
        RuntimeError: When that specific condition occurs.

    Example:
        >>> model = SimpleConvNet(num_classes=5)
        >>> result = function_name(model, param2=10)
        >>> print(result)

    Note:
        Any important notes, warnings, or caveats.
    """
    pass
```

### Code Style Guidelines

1. **Line Length**: Max 120 characters (flexible for readability)

2. **Function Length**: Keep functions under 50 lines when possible

3. **Error Handling**:
   ```python
   if value < 0 or value > 1:
       raise ValueError(
           f"Parameter 'value' must be between 0 and 1, got {value}"
       )
   ```

4. **Device Handling**:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)
   data = data.to(device)
   ```

5. **Gradient Context**:
   ```python
   # When you don't need gradients
   with torch.no_grad():
       predictions = model(test_data)
   ```

6. **Comments**:
   - Use comments to explain **why**, not **what**
   - Complex algorithms: add step numbers with paper reference
   - Non-obvious optimizations: explain the reasoning

---

## Testing Guidelines

### Test Structure Pattern

```python
# tests/test_your_feature.py

"""
Tests for YourFeature implementation.

This module tests:
- Initialization and parameter validation
- Core functionality
- Integration with other components
- Edge cases and error handling
"""

import torch
import pytest
from algorithms.your_feature import YourClass


class TestYourClassInitialization:
    """Tests for YourClass initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        obj = YourClass()
        assert obj.param1 == expected_default

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        obj = YourClass(param1=custom_value)
        assert obj.param1 == custom_value

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            YourClass(param1=-1)


class TestYourClassFunctionality:
    """Tests for YourClass core functionality."""

    def test_forward_pass(self):
        """Test forward pass with valid input."""
        obj = YourClass()
        input_data = torch.randn(10, 1, 28, 28)
        output = obj(input_data)
        assert output.shape == (10, 5)

    def test_backward_pass(self):
        """Test gradients flow correctly."""
        obj = YourClass()
        input_data = torch.randn(10, 1, 28, 28)
        output = obj(input_data)
        loss = output.sum()
        loss.backward()
        # Verify gradients exist


class TestYourClassIntegration:
    """Integration tests with other components."""

    def test_with_meta_dropout(self):
        """Test integration with Meta Dropout."""
        pass
```

### Test Coverage Requirements

**Minimum tests for new algorithms**:
1. ✅ Initialization with default parameters
2. ✅ Initialization with custom parameters
3. ✅ Invalid parameter handling (raises appropriate errors)
4. ✅ Forward pass with dummy data
5. ✅ Output shape verification
6. ✅ Backward pass (gradients flow)
7. ✅ Inner loop adaptation (if applicable)
8. ✅ Integration with Meta Dropout (if applicable)

### Running Tests

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_meta_dropout.py

# Specific test class
python -m pytest tests/test_meta_dropout.py::TestMetaDropout

# Specific test method
python -m pytest tests/test_meta_dropout.py::TestMetaDropout::test_mask_consistency

# Verbose output
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=algorithms --cov-report=html

# Stop on first failure
python -m pytest tests/ -x

# Run only failed tests from last run
python -m pytest tests/ --lf
```

---

## Documentation Standards

### Documentation Files Location

**All documentation goes in `docs/` folder**:
```
docs/
├── ALGORITHM_NAME_OVERVIEW.md      # Algorithm explanation
├── FEATURE_USAGE.md                # How to use features
├── ALGORITHM_COMPARISON.md         # Performance comparisons
└── IMPLEMENTATION_DETAILS.md       # Technical deep-dives
```

### Documentation Template

```markdown
# Algorithm/Feature Name

## Overview

Brief description of what this is and why it exists.

## Key Concepts

### Concept 1
Explanation with equations if needed:

```
θ' = θ - α∇L(θ)
```

### Concept 2
Detailed explanation

## Architecture

Detailed architecture breakdown with diagrams if helpful.

## How It Works

Step-by-step algorithm flow:

1. Step 1: Description
2. Step 2: Description
3. Step 3: Description

## Usage Example

```python
# Complete, runnable code example
import torch
from algorithms.your_algorithm import YourAlgorithm

# Setup
model = SimpleConvNet(num_classes=5)
algo = YourAlgorithm(model, inner_lr=0.01)

# Training
for batch in dataloader:
    loss = algo.meta_train_step(batch)

# Evaluation
results = evaluate_algorithm(model, algo, test_loader)
```

## Performance

| Metric | Value |
|--------|-------|
| Accuracy (5-way 1-shot) | XX% |
| Training Time | X minutes |
| Memory Usage | X GB |

## Comparison with Other Algorithms

| Algorithm | Accuracy | Speed | Memory |
|-----------|----------|-------|--------|
| MAML | 75% | 1x | High |
| Your Algo | XX% | XXx | Medium |

## Tips and Best Practices

- Tip 1
- Tip 2
- Tip 3

## Common Issues

### Issue 1
**Problem**: Description
**Solution**: How to fix

## References

- [Paper Title](url) - Authors, Conference Year
- [Related Work](url) - Context
```

---

## Common Patterns

### Pattern 1: Task Adaptation Loop

**Used in**: MAML, ANIL, Meta-SGD

```python
def inner_update(self, support_data, support_labels):
    """Adapt model to task using support set."""

    # Clone current parameters
    fast_weights = {
        name: param.clone()
        for name, param in self.model.named_parameters()
    }

    # Inner loop adaptation
    for step in range(self.inner_steps):
        # Forward with current fast_weights
        logits = self.forward_with_weights(support_data, fast_weights)
        loss = F.cross_entropy(logits, support_labels)

        # Compute gradients
        grads = torch.autograd.grad(loss, fast_weights.values())

        # Update fast_weights
        fast_weights = {
            name: param - self.inner_lr * grad
            for (name, param), grad in zip(fast_weights.items(), grads)
        }

    return fast_weights
```

### Pattern 2: Forward Pass with Custom Weights

**Used in**: MAML, ANIL, Meta-SGD

```python
def forward_with_weights(self, x, weights):
    """Forward pass using custom weight dictionary."""

    # Layer 1
    x = F.conv2d(
        x,
        weights['conv1.weight'],
        weights.get('conv1.bias')
    )
    x = F.relu(x)

    # Layer 2
    x = F.linear(
        x.flatten(1),
        weights['fc.weight'],
        weights['fc.bias']
    )

    return x
```

### Pattern 3: Meta-Training Step

**Universal pattern across algorithms**:

```python
def meta_train_step(self, task_batch):
    """One meta-training iteration on batch of tasks."""

    support_data, support_labels, query_data, query_labels = task_batch

    # Move to device
    support_data = support_data.to(self.device)
    support_labels = support_labels.to(self.device)
    query_data = query_data.to(self.device)
    query_labels = query_labels.to(self.device)

    meta_loss = 0.0

    # Process each task in batch
    for task_idx in range(support_data.size(0)):
        # Inner loop: Adapt to task
        fast_weights = self.inner_update(
            support_data[task_idx],
            support_labels[task_idx]
        )

        # Evaluate on query set with adapted parameters
        query_logits = self.forward_with_weights(
            query_data[task_idx],
            fast_weights
        )
        task_loss = F.cross_entropy(query_logits, query_labels[task_idx])
        meta_loss += task_loss

    # Average across tasks
    meta_loss = meta_loss / support_data.size(0)

    # Meta-update (outer loop)
    self.meta_optimizer.zero_grad()
    meta_loss.backward()
    self.meta_optimizer.step()

    return meta_loss.item()
```

### Pattern 4: Evaluation Loop

**Algorithm-agnostic evaluation**:

```python
def evaluate_algorithm(model, algorithm, eval_dataloader, num_classes=5):
    """Evaluate meta-learned model on test tasks."""

    model.eval()

    all_accuracies = []
    all_before_accuracies = []

    with torch.no_grad():
        for task_batch in tqdm(eval_dataloader):
            support_data, support_labels, query_data, query_labels = task_batch

            # Measure before adaptation
            before_logits = model(query_data)
            before_acc = (before_logits.argmax(1) == query_labels).float().mean()
            all_before_accuracies.append(before_acc.item())

            # Adapt to task
            fast_weights = algorithm.inner_update(support_data, support_labels)

            # Measure after adaptation
            after_logits = algorithm.forward_with_weights(query_data, fast_weights)
            after_acc = (after_logits.argmax(1) == query_labels).float().mean()
            all_accuracies.append(after_acc.item())

    return {
        'before_adaptation_accuracy': np.mean(all_before_accuracies),
        'after_adaptation_accuracy': np.mean(all_accuracies),
        'before_adaptation_std': np.std(all_before_accuracies),
        'after_adaptation_std': np.std(all_accuracies),
        'improvement': np.mean(all_accuracies) - np.mean(all_before_accuracies)
    }
```

### Pattern 5: Meta Dropout Integration

**How to add Meta Dropout to any model**:

```python
class YourModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rates=[0.05, 0.10, 0.15]):
        super().__init__()

        # Layers
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.dropout1 = MetaDropout(p=dropout_rates[0])

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.dropout2 = MetaDropout(p=dropout_rates[1])

        self.fc = nn.Linear(64 * 7 * 7, num_classes)

        # Flag for outer loop (disable dropout during query evaluation)
        self._outer_loop_mode = False

    def reset_dropout_masks(self, batch_size, device):
        """Call at start of each task."""
        if self._outer_loop_mode:
            return

        # Reset all MetaDropout layers
        for module in self.modules():
            if isinstance(module, MetaDropout):
                # Will be set on first forward pass
                module._mask = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = x.flatten(1)
        x = self.fc(x)
        return x
```

**Usage in training**:
```python
# At task start
model.reset_dropout_masks(batch_size, device)

# Inner loop (dropout active, consistent masks)
for step in range(inner_steps):
    logits = model(support_data)

# Outer loop query evaluation (dropout disabled)
model._outer_loop_mode = True
query_logits = model(query_data)
model._outer_loop_mode = False
```

---

## Working with Algorithms

### Quick Algorithm Selection Guide

**Choose MAML when**:
- Learning from scratch on new dataset
- Need strong theoretical foundation
- Have GPU memory available
- Accuracy is priority over speed

**Choose FOMAML when**:
- Need faster training than MAML
- Limited GPU memory
- Acceptable to sacrifice ~1% accuracy for speed

**Choose Meta-SGD when**:
- Different layers need different adaptation speeds
- Have extra GPU memory (2× MAML)
- Want learnable per-parameter learning rates
- Need highest gradient-based accuracy

**Choose ANIL when**:
- Need 3-10x speedup over MAML
- Have pretrained model (use Scenario 4)
- Want to prevent meta-overfitting
- Training from scratch (use Scenario 2)

**Choose LEO when**:
- Need highest accuracy (95-98%)
- 1-shot learning scenarios
- Have sufficient GPU memory
- Want smooth optimization landscape

**Choose Meta Networks when**:
- Need fast inference (no gradient-based adaptation)
- Single forward pass is sufficient
- Metric-based learning preferred
- Lower memory requirements

**Choose TAML when**:
- Preventing task overfitting is critical
- Want task-agnostic initialization
- Working with diverse task distributions

### Hyperparameter Tuning Guide

**Inner Loop Learning Rate (`inner_lr`)**:
- **Range**: 0.005 - 0.1
- **Default**: 0.01
- **Too high**: Instability, overfitting to support set
- **Too low**: Insufficient adaptation, poor task performance
- **Tuning**: Start at 0.01, increase if adaptation is slow

**Outer Loop Learning Rate (`outer_lr`)**:
- **Range**: 0.0001 - 0.01
- **Default**: 0.001
- **Too high**: Meta-learning instability, divergence
- **Too low**: Slow meta-learning, long training time
- **Tuning**: Use Adam with 0.001, reduce if training unstable

**Inner Steps (`inner_steps`)**:
- **Range**: 1 - 10
- **Default**: 5
- **Too many**: Slow training, risk of overfitting
- **Too few**: Insufficient adaptation
- **Tuning**: 5 is good balance, increase for harder tasks

**Batch Size (tasks per meta-update)**:
- **Range**: 2 - 32
- **Default**: 4
- **Larger**: More stable meta-gradients, more memory
- **Smaller**: Less memory, noisier gradients
- **Tuning**: Use largest that fits in memory

**Dropout Rates (Meta Dropout)**:
- **Range**: 0.0 - 0.2 per layer
- **Default**: [0.05, 0.10, 0.15, 0.05]
- **Tuning**: Start low (0.05), increase if overfitting

### Common Algorithm Issues

**Issue 1: NaN Loss During Training**

**Causes**:
- Learning rate too high
- Gradient explosion
- Invalid input data

**Solutions**:
```python
# 1. Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Reduce learning rates
inner_lr = 0.005  # Instead of 0.01
outer_lr = 0.0001  # Instead of 0.001

# 3. Check for invalid data
assert not torch.isnan(data).any()
assert not torch.isinf(data).any()
```

**Issue 2: Poor Adaptation (Low After-Adaptation Accuracy)**

**Causes**:
- Inner learning rate too low
- Not enough inner steps
- Model capacity insufficient

**Solutions**:
```python
# 1. Increase inner_lr
inner_lr = 0.05  # From 0.01

# 2. More inner steps
inner_steps = 10  # From 5

# 3. Larger model
# Increase number of filters or layers
```

**Issue 3: High Variance in Results**

**Causes**:
- Small batch size
- High dropout rates
- Insufficient meta-training

**Solutions**:
```python
# 1. Increase batch size
batch_size = 8  # From 4

# 2. Reduce dropout
dropout_rates = [0.02, 0.05, 0.08]  # From [0.05, 0.10, 0.15]

# 3. Train longer
num_epochs = 100  # From 50
```

---

## Debugging and Troubleshooting

### Debug Mode Pattern

**Add verbose output to algorithms**:

```python
def inner_update(self, support_data, support_labels, debug=False):
    """Adapt model to task."""

    fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

    for step in range(self.inner_steps):
        logits = self.forward_with_weights(support_data, fast_weights)
        loss = F.cross_entropy(logits, support_labels)

        if debug:
            acc = (logits.argmax(1) == support_labels).float().mean()
            print(f"Inner Step {step}: Loss = {loss:.4f}, Acc = {acc:.2%}")

        grads = torch.autograd.grad(loss, fast_weights.values())

        if debug:
            grad_norm = sum((g**2).sum() for g in grads).sqrt()
            print(f"  Gradient Norm: {grad_norm:.4f}")

        fast_weights = {
            name: param - self.inner_lr * grad
            for (name, param), grad in zip(fast_weights.items(), grads)
        }

    return fast_weights
```

### Gradient Flow Checking

**Verify gradients are flowing**:

```python
def check_gradients(model):
    """Print gradient statistics for all parameters."""
    print("\n=== Gradient Flow Check ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            print(f"{name:30s} | Grad Norm: {grad_norm:8.4f} | Param Norm: {param_norm:8.4f}")
        else:
            print(f"{name:30s} | No gradient!")
    print("===========================\n")

# Usage
loss.backward()
check_gradients(model)
optimizer.step()
```

### Memory Profiling

**Track GPU memory usage**:

```python
import torch

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Usage at critical points
print_gpu_memory()  # Before training
for batch in dataloader:
    loss = model(batch)
    print_gpu_memory()  # During training
```

### Common Error Messages

**Error**: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**Cause**: Trying to compute gradients through tensor that doesn't require gradients

**Solution**:
```python
# Ensure model parameters require grad
for param in model.parameters():
    param.requires_grad = True

# Or use create_graph=True for higher-order gradients
grads = torch.autograd.grad(loss, params, create_graph=True)
```

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# 1. Reduce batch size
batch_size = 2  # Instead of 4

# 2. Use FOMAML instead of MAML
first_order = True

# 3. Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()

# 4. Use gradient checkpointing
# 5. Reduce model size
```

---

## Quick Reference

### File Organization Checklist

When creating new files:
- [ ] Correct naming convention (snake_case.py or UPPERCASE.md)
- [ ] In appropriate directory (algorithms/, tests/, docs/, etc.)
- [ ] Type hints on all functions
- [ ] Comprehensive docstrings
- [ ] Import organization (stdlib, third-party, local)

### Code Review Checklist

Before submitting changes:
- [ ] All tests pass (`python -m pytest tests/`)
- [ ] Type hints added/updated
- [ ] Docstrings added/updated
- [ ] Documentation updated in `docs/`
- [ ] Backward compatibility maintained
- [ ] Notebooks tested (if applicable)
- [ ] No breaking changes (or clearly documented)
- [ ] Code follows conventions

### Performance Benchmarks (5-way 1-shot Omniglot)

| Algorithm | Accuracy | Speed | Memory | When to Use |
|-----------|----------|-------|--------|-------------|
| MAML | 75-80% | 1x | High | Baseline, theoretical foundation |
| FOMAML | 75-80% | 1.4x faster | Medium | Speed priority |
| Meta-SGD | 79.5% | 1x | Very High | Per-parameter learning rates |
| ANIL (First-Order) | 77% | 3x faster | Medium | Fast training from scratch |
| ANIL (Frozen) | 90.5% ⭐ | 3x faster | Medium | Pretrained models |
| LEO | 95-98% ⭐ | 0.7x | Medium | Highest accuracy |
| EB Meta Networks | 77.3% | Very Fast | Low | Fast inference |
| Original Meta Networks | 86% | Very Fast | Low | No gradient adaptation |
| TAML | ~73% | Medium | Medium | Task-agnostic learning |

### Essential Commands

```bash
# Setup
git clone https://github.com/Komil-parmar/meta-learning-from-scratch.git
cd meta-learning-from-scratch
pip install -r requirements.txt

# Download Omniglot
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
unzip images_background.zip -d omniglot/
unzip images_evaluation.zip -d omniglot/

# Testing
python -m pytest tests/                              # All tests
python -m pytest tests/test_specific.py              # Specific test
python -m pytest tests/ -v                           # Verbose
python -m pytest tests/ --cov=algorithms             # With coverage
jupyter nbconvert --execute examples/notebook.ipynb  # Test notebook

# Development
git checkout -b feature/your-feature
# ... make changes ...
python -m pytest tests/
git commit -m "feat: description"
git push origin feature/your-feature
```

### Useful Code Snippets

**Load Omniglot (Standard)**:
```python
from utils.load_omniglot import OmniglotDataset, OmniglotTaskDataset
from torch.utils.data import DataLoader

dataset = OmniglotDataset("omniglot/images_background")
task_dataset = OmniglotTaskDataset(dataset, n_way=5, k_shot=1, k_query=15, num_tasks=2000)
dataloader = DataLoader(task_dataset, batch_size=4, shuffle=True)
```

**Load Omniglot (Prefetched - 10-50x faster)**:
```python
from utils.load_omniglot import PrefetchedOmniglotDataset, OmniglotTaskDataset
from torch.utils.data import DataLoader

dataset = PrefetchedOmniglotDataset("omniglot/images_background")  # Loads into RAM
task_dataset = OmniglotTaskDataset(dataset, n_way=5, k_shot=1, k_query=15, num_tasks=2000)
dataloader = DataLoader(task_dataset, batch_size=4, shuffle=True, num_workers=4)
```

**Train MAML**:
```python
from algorithms.maml import train_maml, ModelAgnosticMetaLearning
from algorithms.cnn_maml import SimpleConvNet

model = SimpleConvNet(num_classes=5)
model, maml, losses = train_maml(
    model, dataloader,
    inner_lr=0.01, outer_lr=0.001, inner_steps=5,
    first_order=False  # True for FOMAML
)
```

**Evaluate**:
```python
from evaluation.evaluate_maml import evaluate_maml
from evaluation.eval_visualization import plot_evaluation_results

results = evaluate_maml(model, maml, test_dataloader, num_classes=5)
plot_evaluation_results(results)
print(f"Accuracy: {results['after_adaptation_accuracy']:.2%}")
```

---

## Contact and Resources

### Getting Help

1. **GitHub Issues**: Bug reports, feature requests
2. **GitHub Discussions**: Algorithm discussions, implementation questions
3. **LinkedIn**: [@komil-parmar](https://www.linkedin.com/in/komil-parmar-488967243/)

### Key Resources

- **Book**: "Meta-Learning: Theory, Algorithms and Applications"
- **Course**: [Stanford CS330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/)
- **MAML Paper**: [arxiv.org/abs/1703.03400](https://arxiv.org/abs/1703.03400)
- **Original Repository**: [github.com/Komil-parmar/meta-learning-from-scratch](https://github.com/Komil-parmar/meta-learning-from-scratch)

---

## Version History

- **2025-11-14**: Initial CLAUDE.md creation
  - Comprehensive codebase structure documentation
  - Algorithm implementations overview
  - Development workflows and conventions
  - Testing and documentation standards

---

**This guide is maintained to help AI assistants effectively contribute to the meta-learning-from-scratch repository. Keep it updated as the codebase evolves!**
