<div align="center">

# 🧠 Meta-Learning From Scratch

**Teaching models to learn how to learn** — one algorithm at a time

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick Start](#-quick-start) • [Algorithms](#-what's-implemented) • [Docs](docs/) • [Examples](examples/)

</div>

---

## 👋 Hey there!

I'm **Komil Parmar**, a second-year undergrad at **IIT Guwahati**, and this is my journey into the fascinating world of meta-learning. Instead of just reading about algorithms, I decided to build them from scratch to really understand what's going on under the hood.

**What's meta-learning?** Think of it as teaching an AI to be a quick learner. Instead of training a model for one specific task, we train it to rapidly adapt to *new* tasks with just a handful of examples. It's like the difference between memorizing facts and learning how to learn.

This repo is my playground for experimenting with different meta-learning algorithms — all built from the ground up, heavily documented, and designed to be easily hackable.

<details>
<summary><b>📖 My Learning Philosophy</b></summary>

I'm following the book **"Meta-Learning: Theory, Algorithms and Applications"** along with papers from the community. But here's the thing — I don't just want to run existing code. I want to:

- **Understand deeply**: Build everything from scratch
- **Experiment freely**: Modular code that's easy to modify
- **Share openly**: Comprehensive docs so others can learn too
- **Stay curious**: Keep adding new algorithms as I learn them

If you're learning meta-learning too, I hope this helps you as much as building it helped me! 🚀

</details>

---

## 🎯 What's Implemented?

<table>
<tr>
<td><b>Algorithm</b></td>
<td><b>Accuracy</b></td>
<td><b>Speed</b></td>
<td><b>Best For</b></td>
<td><b>Docs</b></td>
</tr>

<tr>
<td><b>MAML</b></td>
<td>75-80%</td>
<td>1x</td>
<td>Strong baseline, solid theory</td>
<td><a href="docs/MAML_vs_FOMAML_vs_MAMLpp.md">📖</a></td>
</tr>

<tr>
<td><b>FOMAML</b></td>
<td>75-80%</td>
<td>1.4x faster</td>
<td>Speed + performance balance</td>
<td><a href="docs/MAML_vs_FOMAML_vs_MAMLpp.md">📖</a></td>
</tr>

<tr>
<td><b>Meta-SGD</b></td>
<td>79.5%</td>
<td>1x</td>
<td>Per-parameter learning rates</td>
<td><a href="docs/META_SGD.md">📖</a></td>
</tr>

<tr>
<td><b>ANIL</b></td>
<td>77-90.5%</td>
<td>3x faster</td>
<td>Pretrained models, speed</td>
<td><a href="docs/ANIL.md">📖</a></td>
</tr>

<tr>
<td><b>LEO</b></td>
<td><b>95-98%</b> 🏆</td>
<td>0.7x</td>
<td>Highest accuracy, 1-shot tasks</td>
<td><a href="docs/LEO.md">📖</a></td>
</tr>

<tr>
<td><b>Meta Networks</b></td>
<td>77-86%</td>
<td>Very fast</td>
<td>Fast inference, no gradients</td>
<td><a href="docs/META_NETWORKS_OVERVIEW.md">📖</a></td>
</tr>

<tr>
<td><b>TAML</b></td>
<td>~73%</td>
<td>Medium</td>
<td>Task-agnostic learning</td>
<td><a href="docs/TAML.md">📖</a></td>
</tr>

<tr>
<td><b>Reptile</b></td>
<td>~75%</td>
<td>Fast</td>
<td>Simpler than MAML</td>
<td><a href="docs/MAML_vs_FOMAML_vs_REPTILE.md">📖</a></td>
</tr>

</table>

<sub>*All numbers are for 5-way 1-shot classification on Omniglot*</sub>

<details>
<summary><b>🎨 Special Features</b></summary>

- **Meta Dropout**: A dropout technique that maintains consistent masks during task adaptation. Improves accuracy by ~1-2% across algorithms!
- **Prefetched Dataset**: Load Omniglot into RAM once → 10-50x faster training. Because life's too short to wait for data loading.
- **Shared Components**: All algorithms use the same `EmbeddingNetwork` for fair comparisons.

</details>

---

## 🗂️ How This Repo is Organized

```
meta-learning-from-scratch/
│
├── 🧪 algorithms/          # The core implementations
│   ├── maml.py              # MAML/FOMAML/Meta-SGD
│   ├── anil.py              # Almost No Inner Loop
│   ├── leo.py               # Latent Embedding Optimization
│   ├── *_meta_network.py    # Meta Networks variants
│   └── meta_dropout.py      # Our secret sauce
│
├── 📊 evaluation/          # Measure performance & visualize
│   ├── evaluate_*.py        # Algorithm-specific evaluation
│   └── eval_visualization.py # Pretty plots
│
├── 🔬 tests/               # Make sure nothing breaks
│   └── test_*.py            # Pytest test suites
│
├── 🛠️ utils/               # Dataset loading & helpers
│   ├── load_omniglot.py     # Standard + prefetched loaders
│   └── visualize_omniglot.py # See your data
│
├── 📚 docs/                # Deep dives into each algorithm
│   └── *.md                 # Comprehensive guides
│
└── 🎓 examples/            # Learn by doing
    └── *.ipynb              # Interactive tutorials
```

**Philosophy:** Each folder has one job. Want to add a new algorithm? Drop it in `algorithms/`. Need to change how evaluation works? Check `evaluation/`. Everything is self-contained and modular.

---

## 🚀 Quick Start

### Installation

```bash
# Clone this bad boy
git clone https://github.com/Komil-parmar/meta-learning-from-scratch.git
cd meta-learning-from-scratch

# Install dependencies
pip install -r requirements.txt
```

### Get the Dataset

```bash
# Download Omniglot (the "MNIST of few-shot learning")
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip

# Extract
unzip images_background.zip -d omniglot/
unzip images_evaluation.zip -d omniglot/
```

### Train Your First Meta-Learner

```python
from algorithms.maml import train_maml
from algorithms.cnn_maml import SimpleConvNet
from utils.load_omniglot import PrefetchedOmniglotDataset, OmniglotTaskDataset
from torch.utils.data import DataLoader

# Load dataset (prefetched = fast! ⚡)
dataset = PrefetchedOmniglotDataset("omniglot/images_background")
task_dataset = OmniglotTaskDataset(dataset, n_way=5, k_shot=1, k_query=15, num_tasks=2000)
dataloader = DataLoader(task_dataset, batch_size=4, shuffle=True)

# Create model
model = SimpleConvNet(num_classes=5)

# Train with MAML
model, maml, losses = train_maml(
    model, dataloader,
    inner_lr=0.01,      # How fast to adapt to each task
    outer_lr=0.001,     # How fast to meta-learn
    inner_steps=5,      # Adaptation steps per task
    first_order=False   # True for FOMAML (faster)
)

# That's it! You just trained a meta-learner 🎉
```

<details>
<summary><b>🎛️ Want to try different algorithms?</b></summary>

**FOMAML (faster):**
```python
train_maml(model, dataloader, first_order=True)
```

**Meta-SGD (learns learning rates):**
```python
train_maml(model, dataloader, meta_sgd=True)
```

**ANIL (3x faster):**
```python
from algorithms.anil import train_anil
body, head = create_anil_network(num_classes=5)
train_anil(body, head, dataloader, first_order=True, freeze_body=True)
```

Check out the [examples/](examples/) folder for complete tutorials on each algorithm!

</details>

---

## 🎓 Learning Resources

Each algorithm comes with:
- **📖 Comprehensive docs** in the `docs/` folder explaining theory + implementation
- **🎓 Jupyter notebooks** with step-by-step walkthroughs
- **🧪 Tests** that double as usage examples

**Start here:**
1. [`examples/maml_on_omniglot.ipynb`](examples/maml_on_omniglot.ipynb) — Great introduction to meta-learning
2. [`docs/MAML_vs_FOMAML_vs_MAMLpp.md`](docs/MAML_vs_FOMAML_vs_MAMLpp.md) — Compare different MAML variants
3. [`docs/ANIL.md`](docs/ANIL.md) — My favorite: fastest algorithm with great performance

---

## 🔧 Want to Use Your Own Dataset?

The beauty of this repo is everything is modular. Just follow this pattern:

```python
# 1. Create your dataset (like OmniglotDataset)
class MyDataset(Dataset):
    def __getitem__(self, idx):
        return images, class_idx

# 2. Create task sampler (N-way K-shot)
task_dataset = MyTaskDataset(
    my_dataset,
    n_way=5,      # 5 classes per task
    k_shot=1,     # 1 example per class
    k_query=15    # 15 test examples per class
)

# 3. Use any algorithm — they all work the same way!
train_maml(model, DataLoader(task_dataset), ...)
```

That's literally it. The algorithms don't care what data you throw at them.

---

## 💡 Why This Repo?

**For learners:** Every line is documented. Every choice is explained. Read the code, run the notebooks, break things, and learn.

**For researchers:** Modular design means you can quickly test ideas. Want to try a new inner loop optimizer? Just modify `inner_update()`. Want to test on a new dataset? Plug it in.

**For practitioners:** Production-ready implementations with proper testing, type hints, and documentation. Not just research code that barely runs.

---

## 🤝 Contributing

Found a bug? Want to add a new algorithm? Have a cool idea?

I'd love your contributions! Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Some ideas:
- Add Prototypical Networks
- Add Matching Networks
- Improve documentation
- Add more datasets
- Optimize existing code

Just remember: **clarity over cleverness**. This repo is for learning, so keep it readable!

---

## 📚 References & Inspiration

- **Book:** "Meta-Learning: Theory, Algorithms and Applications"
- **Course:** [Stanford CS330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/)
- **Papers:** [MAML](https://arxiv.org/abs/1703.03400), [ANIL](https://arxiv.org/abs/1909.09157), [LEO](https://arxiv.org/abs/1807.05960), [Meta Networks](https://arxiv.org/abs/1703.00837), and more
- **Dataset:** [Omniglot](https://github.com/brendenlake/omniglot) by Brenden Lake

---

## 📬 Let's Connect!

I'm always happy to chat about meta-learning, answer questions, or hear about what you're building!

- **LinkedIn:** [Komil Parmar](https://www.linkedin.com/in/komil-parmar-488967243/)
- **GitHub Issues:** Questions, bugs, or ideas? [Open an issue](https://github.com/Komil-parmar/meta-learning-from-scratch/issues)

---

## 📄 License

MIT License — feel free to use this for learning, research, or building cool stuff!

---

<div align="center">

**Made with ❤️ and lots of ☕ by a curious undergrad**

*If this helped you learn something new, consider giving it a ⭐!*

**Happy meta-learning! 🚀**

</div>
