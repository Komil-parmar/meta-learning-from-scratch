# 🤝 Contributing to Meta-Learning From Scratch

Thank you for your interest in contributing to this meta-learning repository! This project aims to provide clear, well-documented implementations of meta-learning algorithms for educational purposes. All contributions that align with this goal are welcome! 🎉

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Project Structure](#️-project-structure)
- [Development Guidelines](#-development-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Documentation Standards](#-documentation-standards)
- [Testing Requirements](#-testing-requirements)
- [Getting Help](#-getting-help)

---

## 📜 Code of Conduct

This is a learning project built with passion and dedication. We expect all contributors to:
- Be respectful and constructive in all interactions
- Focus on educational value and code clarity
- Help others learn and understand meta-learning concepts
- Provide thoughtful feedback on contributions

---

## 🎯 How Can I Contribute?

### 1. **Improving Existing Algorithms** ✨
- Bug fixes and performance optimizations
- Better documentation and code comments
- Enhanced error handling
- Improved test coverage

### 2. **Adding New Algorithms** 🆕
- Prototypical Networks
- Matching Networks
- Reptile
- Relation Networks
- Other meta-learning algorithms

### 3. **Documentation Improvements** 📚
- Tutorial notebooks
- Usage guides
- Algorithm comparisons
- Performance benchmarks

### 4. **Bug Reports** 🐛
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, PyTorch version)

---

## 🗂️ Project Structure

Our repository follows a clean, modular structure. **Please maintain this organization:**

```
meta-learning-from-scratch/
├── algorithms/                  # Core algorithm implementations
│   ├── maml.py                  # snake_case naming
│   ├── meta_dropout.py
│   └── ...
├── evaluation/                  # Evaluation and visualization
│   ├── evaluate_maml.py
│   └── eval_visualization.py
├── tests/                       # Test suites
│   ├── test_meta_dropout.py
│   └── test_meta_network_dropout.py
├── utils/                       # Dataset utilities
│   ├── load_omniglot.py
│   └── visualize_omniglot.py
├── docs/                        # Documentation (UPPERCASE.md)
│   ├── MAML_vs_FOMAML.md
│   ├── META_DROPOUT_USAGE.md
│   └── ...
├── examples/                    # Tutorial notebooks
│   ├── maml_on_omniglot.ipynb
│   └── ...
├── CONTRIBUTING.md              # This file
└── README.md
```

### 📝 File Naming Conventions

**Critical: Please follow these naming conventions strictly!**

- **Python files**: `snake_case` (lowercase with underscores)
  - ✅ `meta_dropout.py`, `original_meta_network.py`
  - ❌ `MetaDropout.py`, `Original_Meta_Network.py`

- **Documentation files**: `UPPERCASE_WITH_UNDERSCORES.md`
  - ✅ `META_DROPOUT_USAGE.md`, `MAML_vs_FOMAML.md`
  - ❌ `meta_dropout_usage.md`, `maml-vs-fomaml.md`

- **Notebooks**: `snake_case.ipynb` (descriptive names)
  - ✅ `maml_on_omniglot.ipynb`, `meta_network.ipynb`
  - ❌ `MAML.ipynb`, `notebook1.ipynb`

---

## 💻 Development Guidelines

### Code Quality Standards

#### 1. **Type Hints Are Mandatory** ✅

All functions and class methods must include type hints:

```python
# ✅ Good
def train_maml(
    model: nn.Module,
    task_dataloader: DataLoader,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    first_order: bool = False
) -> tuple[nn.Module, ModelAgnosticMetaLearning, list[float]]:
    """Train a model using MAML."""
    pass

# ❌ Bad
def train_maml(model, task_dataloader, inner_lr=0.01, outer_lr=0.001):
    """Train a model using MAML."""
    pass
```

#### 2. **Meaningful Names** 📝

Use descriptive, self-documenting names:

```python
# ✅ Good
def reset_dropout_masks(self, batch_size: int, device: torch.device):
    """Reset all Meta Dropout masks for a new task."""
    pass

# ❌ Bad
def reset(self, bs, dev):
    """Reset masks."""
    pass
```

#### 3. **Default Values When Appropriate** 🎯

Provide sensible defaults for parameters:

```python
# ✅ Good
class MetaDropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """Initialize Meta Dropout layer with default probability."""
        pass

# ❌ Bad (requires all parameters)
class MetaDropout(nn.Module):
    def __init__(self, p, inplace):
        pass
```

#### 4. **Comprehensive Docstrings** 📖

Every class and function must have detailed docstrings:

```python
def evaluate_maml(
    model: nn.Module,
    maml: ModelAgnosticMetaLearning,
    eval_dataloader: DataLoader,
    num_classes: int = 5,
    verbose: bool = True
) -> dict:
    """
    Evaluate MAML model performance on test tasks.
    
    This function measures the model's ability to adapt to new tasks by
    computing accuracy before and after inner loop adaptation.
    
    Args:
        model (nn.Module):
            The neural network model to evaluate. Should be compatible with MAML.
        
        maml (ModelAgnosticMetaLearning):
            MAML wrapper containing inner_update and forward_with_weights methods.
        
        eval_dataloader (DataLoader):
            DataLoader yielding evaluation tasks. Each task should contain:
            (support_data, support_labels, query_data, query_labels)
        
        num_classes (int, optional):
            Number of classes per task (N-way). Default: 5
            Used to calculate random baseline performance.
        
        verbose (bool, optional):
            Whether to print detailed evaluation results. Default: True
            Set to False for silent evaluation.
    
    Returns:
        dict: Evaluation metrics including:
            - 'before_adaptation_accuracy': float, accuracy before adaptation
            - 'after_adaptation_accuracy': float, accuracy after adaptation
            - 'before_adaptation_std': float, standard deviation before
            - 'after_adaptation_std': float, standard deviation after
            - 'all_accuracies': list[float], per-task accuracies
            - 'num_tasks': int, total number of tasks evaluated
            - 'random_baseline': float, random guess baseline
    
    Example:
        >>> model = SimpleConvNet(num_classes=5)
        >>> maml = ModelAgnosticMetaLearning(model, inner_lr=0.01)
        >>> results = evaluate_maml(model, maml, test_loader)
        >>> print(f"Accuracy: {results['after_adaptation_accuracy']:.2%}")
    
    Note:
        The model should be in eval mode for proper evaluation.
        Dropout and batch normalization will be disabled automatically.
    """
    pass
```

---

## 🔄 Pull Request Process

### One Change Per PR Rule 🚨

**Important: Do NOT mix different types of changes in a single PR!**

#### ✅ Good PR Examples:
- **PR 1**: "Fix bug in MAML inner loop gradient computation"
- **PR 2**: "Add Prototypical Networks implementation"
- **PR 3**: "Improve Meta Dropout documentation with usage examples"

#### ❌ Bad PR Examples:
- **PR**: "Add Prototypical Networks + fix MAML bug + update docs"
- **PR**: "Improve MAML performance + add new algorithm"

### Steps for Contributing

#### 1. **Fork and Clone**
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/Komil-parmar/meta-learning-from-scratch.git
cd meta-learning-from-scratch
```

#### 2. **Create a Branch**
```bash
# For new algorithms
git checkout -b feature/add-prototypical-networks

# For bug fixes
git checkout -b fix/maml-gradient-bug

# For documentation
git checkout -b docs/improve-meta-dropout-guide
```

#### 3. **Make Your Changes**

Follow all guidelines in this document:
- ✅ Correct file naming conventions
- ✅ Type hints on all functions/methods
- ✅ Comprehensive docstrings
- ✅ Tests for new functionality
- ✅ Documentation updates

#### 4. **Test Thoroughly**

```bash
# Run existing tests
python -m pytest tests/

# Run your new tests
python -m pytest tests/test_your_feature.py

# Test affected notebooks (if applicable)
jupyter nbconvert --execute examples/your_notebook.ipynb
```

#### 5. **Verify Integration**

**Critical: Ensure backward compatibility!**

If you modified existing files, verify that:
- ✅ All other files importing your changes still work
- ✅ All notebooks using the modified code still run
- ✅ All tests pass
- ✅ No breaking changes introduced

```bash
# Example: If you modified algorithms/meta_dropout.py
# You MUST test:
python -m pytest tests/test_meta_dropout.py
python -m pytest tests/test_meta_network_dropout.py

# And verify notebooks:
jupyter nbconvert --execute examples/maml_on_omniglot.ipynb
jupyter nbconvert --execute examples/embedding_based_meta_network.ipynb
```

#### 6. **Commit Your Changes**

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "feat: Add Prototypical Networks implementation with Meta Dropout"
git commit -m "fix: Correct gradient computation in MAML inner loop"
git commit -m "docs: Add comprehensive Meta Dropout usage guide"
git commit -m "test: Add integration tests for Original Meta Networks"

# Bad commit messages
git commit -m "updates"
git commit -m "fixed stuff"
git commit -m "wip"
```

#### 7. **Push and Create PR**

```bash
git push origin feature/your-branch-name
```

Then create a Pull Request on GitHub with:
- **Clear title** describing the change
- **Detailed description** of what and why
- **Testing performed** and results
- **Breaking changes** (if any)
- **Related issues** (if applicable)

### PR Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New algorithm implementation
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test addition/improvement

## Changes Made
- Specific change 1
- Specific change 2
- Specific change 3

## Testing Performed
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Tested with affected notebooks
- [ ] Verified backward compatibility

## Documentation
- [ ] Updated relevant documentation in `docs/`
- [ ] Added/updated docstrings
- [ ] Updated README.md (if applicable)
- [ ] Added tutorial notebook (for new algorithms)

## Checklist
- [ ] Code follows project naming conventions
- [ ] All functions have type hints
- [ ] All functions have comprehensive docstrings
- [ ] No breaking changes (or clearly documented)
- [ ] One logical change per PR

## Additional Notes
Any additional information, context, or screenshots.
```

---

## 📚 Documentation Standards

### Where to Put Documentation

All documentation files **MUST** go in the `docs/` folder:

```
docs/
├── ALGORITHM_NAME_OVERVIEW.md      # Algorithm explanation
├── FEATURE_USAGE.md                # Usage guides
├── ALGORITHM_COMPARISON.md         # Comparative analysis
└── IMPLEMENTATION_DETAILS.md       # Technical deep-dives
```

### Documentation Requirements

#### For New Algorithms:

1. **Overview Document** (`docs/ALGORITHM_NAME_OVERVIEW.md`):
   ```markdown
   # Algorithm Name Overview
   
   ## 🎯 What is Algorithm Name?
   Clear, intuitive explanation
   
   ## 🏗️ Architecture
   Detailed architecture breakdown
   
   ## 🔄 How It Works
   Step-by-step algorithm flow
   
   ## 🎯 Key Differences
   Comparison with other algorithms
   
   ## 📊 Expected Performance
   Benchmark results
   
   ## 🚀 Usage Example
   Complete code example
   
   ## 📚 References
   Original paper and related work
   ```

2. **Tutorial Notebook** (`examples/algorithm_name.ipynb`):
   - Dataset exploration (or reference to existing notebook)
   - Architecture explanation with visualizations
   - Step-by-step training
   - Evaluation and analysis
   - Comparison with other methods

3. **Integration Documentation** (update existing docs):
   - Update `README.md` with new algorithm
   - Add to performance comparison tables
   - Update Meta Dropout integration (if applicable)

#### For Bug Fixes/Improvements:

- Update affected documentation files
- Add notes about the fix to relevant guides
- Update examples if behavior changed

### Documentation Style Guide

- **Use emojis** for visual appeal and easy scanning 🎯
- **Code examples** must be complete and runnable
- **Include visualizations** where helpful (architecture diagrams, plots)
- **Provide context** - explain the "why" not just the "how"
- **Link to related docs** for cross-referencing
- **Add performance metrics** with clear methodology
- **Include references** to original papers and resources

---

## 🧪 Testing Requirements

### Test Coverage

**All new code must include tests!** Even if not covering every scenario.

#### Minimum Testing Requirements:

1. **Unit Tests**: Core functionality
   ```python
   # tests/test_your_algorithm.py
   def test_algorithm_initialization():
       """Test that algorithm initializes correctly."""
       model = YourAlgorithm(param1=value1, param2=value2)
       assert model.param1 == value1
       assert model.param2 == value2
   
   def test_forward_pass():
       """Test forward pass with dummy data."""
       model = YourAlgorithm()
       dummy_input = torch.randn(10, 1, 28, 28)
       output = model(dummy_input)
       assert output.shape == (10, 5)  # Expected output shape
   ```

2. **Integration Tests**: Algorithm interaction
   ```python
   def test_algorithm_with_meta_dropout():
       """Test algorithm works with Meta Dropout."""
       model = YourAlgorithm(use_meta_dropout=True)
       # Test mask consistency
       # Test training loop
       # Test evaluation mode
   ```

3. **Edge Cases**: Boundary conditions
   ```python
   def test_empty_support_set():
       """Test handling of edge cases."""
       # Test with minimal data
       # Test with mismatched dimensions
       # Test with invalid parameters
   ```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_your_algorithm.py

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=algorithms --cov-report=html
```

### Test File Structure

```python
"""
Tests for Your Algorithm implementation.

This module tests the core functionality of YourAlgorithm including:
- Initialization and parameter validation
- Forward pass correctness
- Integration with Meta Dropout
- Edge case handling
"""

import torch
import pytest
from algorithms.your_algorithm import YourAlgorithm


class TestYourAlgorithmInitialization:
    """Tests for YourAlgorithm initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        pass
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        pass


class TestYourAlgorithmForward:
    """Tests for YourAlgorithm forward pass."""
    
    def test_forward_with_valid_input(self):
        """Test forward pass with valid input."""
        pass
    
    def test_forward_output_shape(self):
        """Test output shape is correct."""
        pass


class TestYourAlgorithmIntegration:
    """Integration tests for YourAlgorithm."""
    
    def test_training_loop(self):
        """Test complete training loop."""
        pass
    
    def test_with_meta_dropout(self):
        """Test integration with Meta Dropout."""
        pass
```

---

## 🎯 Specific Contribution Guidelines

### Adding a New Algorithm

**Checklist:**

- [ ] **Implementation** (`algorithms/your_algorithm.py`):
  - [ ] Type hints on all functions/methods
  - [ ] Comprehensive docstrings
  - [ ] Default parameter values
  - [ ] Integration with Meta Dropout (if applicable)
  - [ ] Follows existing code style

- [ ] **Tests** (`tests/test_your_algorithm.py`):
  - [ ] Unit tests for core functionality
  - [ ] Integration tests
  - [ ] Edge case handling

- [ ] **Documentation** (`docs/YOUR_ALGORITHM_OVERVIEW.md`):
  - [ ] Clear algorithm explanation
  - [ ] Architecture details
  - [ ] Usage examples
  - [ ] Performance benchmarks
  - [ ] References to original paper

- [ ] **Tutorial** (`examples/your_algorithm.ipynb`):
  - [ ] Step-by-step implementation
  - [ ] Training on Omniglot
  - [ ] Evaluation and analysis
  - [ ] Comparison with other algorithms

- [ ] **Integration**:
  - [ ] Update `README.md`
  - [ ] Update repository structure
  - [ ] Add to algorithm comparison tables
  - [ ] Test with existing utilities

### Improving Existing Code

**Checklist:**

- [ ] **Changes**:
  - [ ] Maintain backward compatibility
  - [ ] Update type hints if needed
  - [ ] Update docstrings to reflect changes
  - [ ] Follow existing code style

- [ ] **Testing**:
  - [ ] All existing tests pass
  - [ ] Add tests for new functionality
  - [ ] Test all files that import modified code
  - [ ] Test all notebooks using modified code

- [ ] **Documentation**:
  - [ ] Update relevant docs in `docs/`
  - [ ] Update code examples in docs
  - [ ] Update README if behavior changed
  - [ ] Add migration notes for breaking changes

- [ ] **Verification**:
  - [ ] Run full test suite
  - [ ] Execute affected notebooks
  - [ ] Verify no breaking changes

---

## 📖 Code Style Guide

### Python Code Style

Follow PEP 8 with these specific guidelines:

```python
# Imports: Standard library, third-party, local
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from algorithms.meta_dropout import MetaDropout
from utils.load_omniglot import OmniglotDataset

# Class definitions
class YourAlgorithm(nn.Module):
    """One-line summary.
    
    Detailed description of what this class does,
    its purpose, and how it fits into the project.
    
    Args:
        param1 (type): Description of param1.
        param2 (type, optional): Description of param2. Default: value.
    
    Attributes:
        attr1 (type): Description of attr1.
        attr2 (type): Description of attr2.
    
    Example:
        >>> model = YourAlgorithm(param1=value1)
        >>> output = model(input_data)
    """
    
    def __init__(self, param1: int, param2: float = 0.5):
        super(YourAlgorithm, self).__init__()
        self.param1 = param1
        self.param2 = param2
        
        # Initialize components
        self.layer1 = nn.Linear(param1, 64)
        self.dropout = MetaDropout(p=param2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the algorithm.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, features]
        
        Returns:
            torch.Tensor: Output tensor [batch_size, output_dim]
        """
        x = self.layer1(x)
        x = self.dropout(x)
        return x

# Function definitions
def helper_function(
    data: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float = 0.001
) -> Tuple[torch.Tensor, float]:
    """One-line summary of what this function does.
    
    Detailed explanation of the function's purpose,
    algorithm, and any important implementation details.
    
    Args:
        data (torch.Tensor): Description. Shape: [batch, features]
        labels (torch.Tensor): Description. Shape: [batch]
        learning_rate (float, optional): Description. Default: 0.001
    
    Returns:
        Tuple[torch.Tensor, float]: 
            - predictions: Model predictions [batch, classes]
            - loss: Computed loss value
    
    Raises:
        ValueError: If data and labels have mismatched batch sizes.
    
    Example:
        >>> data = torch.randn(32, 10)
        >>> labels = torch.randint(0, 5, (32,))
        >>> preds, loss = helper_function(data, labels)
    """
    # Implementation
    pass
```

### Constants and Configuration

```python
# Use uppercase for constants
DEFAULT_LEARNING_RATE = 0.001
MAX_ITERATIONS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use dataclasses or named tuples for configuration
from dataclasses import dataclass

@dataclass
class AlgorithmConfig:
    """Configuration for YourAlgorithm."""
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout_rate: float = 0.5
    num_classes: int = 5
```

---

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Reproduction Steps**: Minimal code to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS (Windows/Linux/Mac)
   - Python version
   - PyTorch version
   - CUDA version (if using GPU)
6. **Traceback**: Full error traceback if applicable

**Bug Report Template:**

```markdown
## Bug Description
Clear and concise description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Code to Reproduce
```python
# Minimal reproducible example
import torch
from algorithms.your_algorithm import YourAlgorithm

model = YourAlgorithm()
# ... code that triggers bug
```

## Environment
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]

## Error Traceback
```
Full error traceback here
```

## Additional Context
Any other relevant information.
```

---

## 📚 References and Resources

When adding new algorithms, **always include references**:

### Required References:

1. **Original Paper**:
   ```markdown
   - [Algorithm Name Paper](paper_url) - Author et al., Conference Year
   ```

2. **Related Work**:
   ```markdown
   - [Related Paper 1](url) - Context of relation
   - [Related Paper 2](url) - Context of relation
   ```

3. **Code References** (if applicable):
   ```markdown
   - [Official Implementation](url)
   - [Other Implementations](url)
   ```

4. **Educational Resources**:
   ```markdown
   - [Stanford CS330](https://cs330.stanford.edu/) - Course covering the algorithm
   - [Blog Post](url) - Clear explanation
   ```

### Citing This Repository

If contributors want to cite your repository:

```bibtex
@misc{meta-learning-from-scratch,
  author = {Your Name},
  title = {Meta-Learning From Scratch},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/meta-learning-from-scratch}
}
```

---

## 💬 Getting Help

### Questions About Contributing?

I'm happy to help! Here are the best ways to reach out:

1. **LinkedIn**: [Connect with me on LinkedIn](https://www.linkedin.com/in/komil-parmar-488967243/)
   - Best for: Specific advice, conceptual questions, collaboration ideas
   - Response time: Usually within 24-48 hours

2. **GitHub Issues**: [Open an issue](https://github.com/Komil-parmar/meta-learning-from-scratch/issues)
   - Best for: Bug reports, feature requests, general questions
   - Use labels: `question`, `help wanted`, `good first issue`

3. **GitHub Discussions**: [Start a discussion](https://github.com/Komil-parmar/meta-learning-from-scratch/discussions)
   - Best for: Algorithm discussions, implementation approaches, brainstorming

### Before Asking:

1. Check existing issues and discussions
2. Review relevant documentation
3. Read through similar code in the repository
4. Try to isolate the problem with a minimal example

### When Asking:

- Be specific about what you're trying to achieve
- Share relevant code and error messages
- Explain what you've already tried
- Ask clear, focused questions

---

## 🌟 Recognition

Contributors will be:
- Added to the Contributors list in README.md
- Mentioned in release notes for their contributions
- Credited in relevant documentation they create

Significant contributions may result in:
- Co-authorship on any potential paper/publication
- Featured case studies showcasing your work
- LinkedIn recommendations

---

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License, the same license as the project.

---

## 🎓 Learning Together

Remember, this is a learning project! Don't be afraid to:
- Ask questions
- Propose ideas
- Make mistakes (we all do!)
- Learn from code reviews

The goal is to build something educational and useful while learning meta-learning concepts together. Every contribution, no matter how small, helps make this resource better for everyone! 🚀

---

**Thank you for contributing to Meta-Learning From Scratch!** 🙏

*Made with ❤️ for the meta-learning community*

---

## 📌 Quick Reference

### Essential Commands

```bash
# Setup
git clone https://github.com/Komil-parmar/meta-learning-from-scratch.git
cd meta-learning-from-scratch
pip install -r requirements.txt

# Development
git checkout -b feature/your-feature
# ... make changes ...
python -m pytest tests/
git commit -m "feat: your feature description"
git push origin feature/your-feature

# Testing
python -m pytest tests/                           # All tests
python -m pytest tests/test_your_file.py          # Specific test
python -m pytest tests/ -v                        # Verbose
python -m pytest tests/ --cov=algorithms          # With coverage
jupyter nbconvert --execute examples/notebook.ipynb  # Test notebook
```

### File Naming Quick Reference

| Type | Convention | Example |
|------|-----------|---------|
| Python files | `snake_case.py` | `meta_dropout.py` |
| Documentation | `UPPERCASE.md` | `META_DROPOUT_USAGE.md` |
| Notebooks | `snake_case.ipynb` | `maml_on_omniglot.ipynb` |
| Test files | `test_*.py` | `test_meta_dropout.py` |

### PR Checklist

- [ ] One logical change per PR
- [ ] Correct file naming conventions
- [ ] Type hints on all functions
- [ ] Comprehensive docstrings
- [ ] Tests included
- [ ] Documentation updated
- [ ] All tests pass
- [ ] Notebooks verified (if applicable)
- [ ] No breaking changes (or documented)
- [ ] References included (for new algorithms)
