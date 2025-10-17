# LEO (Latent Embedding Optimization) - Implementation Overview

## 💡 Intuition: Why Optimize in Latent Space?

> **Prerequisites**: Understanding [Meta-SGD](META_SGD.md) will significantly clarify LEO's motivation and design philosophy. Both algorithms tackle the same fundamental question from different angles: *How do we enable task-specific adaptation beyond what MAML offers?*

### The Core Problem with Parameter Space Optimization

With **MAML** and its variants, you're optimizing directly in **parameter space**. After a few inner-loop gradient steps on task τ, you get:

```
θ' = θ - α∇L(θ)
```

Then you optimize the outer loop so that this adapted θ' generalizes well across tasks.

This works, but has a fundamental limitation: **Every task adapts in the same way** — by moving parameters along the gradient direction with a fixed (or per-parameter fixed) learning rate.

### Meta-SGD's Approach: Per-Parameter Learning Rates

**Meta-SGD** ([detailed here](META_SGD.md)) asks: *Does every parameter need the same learning rate?*

It learns per-parameter learning rates α_i, allowing each parameter to adapt at its own optimal pace:

```
θ'_i = θ_i - α_i ∇L(θ_i)  // Different α_i for each parameter
```

**But Meta-SGD still has a limitation:** The learning rates α_i are fixed across all tasks. If parameter `i` is:
- Task-general for most tasks (needs small α_i), but
- Task-specific for task τ (needs large α_i)

Meta-SGD **cannot adapt** — it's locked into that fixed α_i it learned during meta-training.

### LEO's Breakthrough: Task-Conditional Adaptation

**LEO's core insight**: Instead of optimizing in the high-dimensional parameter space (~100K+ parameters), optimize in a **learned low-dimensional latent embedding space** (e.g., 64 dimensions).

```
MAML:     θ → θ' (optimize in parameter space, ~100K dims)
Meta-SGD: θ → θ' (with per-param α, but still ~100K dims)
LEO:      θ → z → z' → θ' (optimize in latent space, ~64 dims!)
```

**Key advantage**: LEO learns a **task-conditional latent code** z_τ for each task. The same parameter can be adapted differently across tasks because the latent representation captures task-specific structure.

### The Magic: Learned Coordinate Systems

Here's the **critical insight** that makes LEO more powerful than Meta-SGD:

**The encoder-decoder isn't learning per-parameter learning rates. It's learning a learned basis (coordinate system) for the parameter space.**

Think of it this way:

1. **Parameter Space** (original): 100,000+ dimensions, one per parameter
2. **Latent Space** (learned): 64 dimensions, learned basis vectors

When you update `z` in this latent basis, you're **implicitly selecting which parameters to move and by how much** — but in a way that can be different for each task!

#### How the Implicit Selection Works

During the inner loop, when you compute `∇_z L_τ(Decoder(z))`, you're asking:

> *"Which dimensions of z, when changed, will most reduce the task loss?"*

The **decoder** creates a mapping `z → θ`. Its architecture determines how latent dimensions map to parameters:

- If the decoder learned to encode **parameter i in multiple latent dimensions** (spread out):
  - Changes to z have **distributed effects** on parameter i
  - Multiple latent dimensions collaborate to control parameter i
  
- If the decoder learned to encode **parameter i in one specific latent dimension**:
  - That dimension becomes **"responsible"** for controlling parameter i
  - Direct, focused control

**The beautiful part**: The decoder self-organizes during meta-training so that:
- **Task-adaptable parameters** are controllable from a small number of latent dimensions
- **Task-general parameters** are encoded in ways that make them stable (compressed into shared dimensions)

Then the **inner-loop gradient** naturally finds and updates the task-specific dimensions!

### Encoder's Role: Compression with Purpose

The **encoder's** role is almost the reverse of the decoder:

```
Encoder: θ → z  (compress parameters → latent code)
Decoder: z → θ  (expand latent code → parameters)
```

But it's not just compression — it's **intelligent compression**:

1. **Extract task-adaptable aspects**: The encoder learns to identify which aspects of the current parameters represent task-specific adaptations
2. **Discard task-general information**: Compress away the parts that don't need adaptation (shared knowledge across tasks)
3. **Create task-conditional representation**: The resulting z_τ captures "what matters for this task"

#### All Parameters Go Through Compression

**Important**: It's not that the encoder "chooses" which parameters to adapt. Instead:

1. **All parameters** are compressed into the latent space
2. The **encoder** learns to:
   - Map all parameters into a compressed latent space
   - Emphasize task-adaptation aspects in the latent representation
   
3. The **decoder** learns to:
   - Use the latent space to find which directions enable task-specific adaptation
   - Reconstruct task-general parameters that don't need much adaptation
   - Spread or concentrate parameters across latent dimensions based on their role

4. The **inner-loop gradient** (∇_z L) automatically focuses on the latent dimensions that matter for the current task

### Why This Works Better Than Meta-SGD

| Aspect | Meta-SGD | LEO |
|--------|----------|-----|
| **Adaptation mechanism** | Per-parameter learning rate α_i | Task-conditional latent code z_τ |
| **Task specificity** | Fixed α_i for all tasks | Different z_τ for each task |
| **Optimization space** | Parameter space (~100K dims) | Latent space (~64 dims) |
| **Implicit selection** | Explicitly set α_i per parameter | Decoder learns which params to adapt |
| **Flexibility** | Parameter i always uses same α_i | Parameter i can adapt differently per task |

**Example scenario:**

Imagine a parameter that controls "edge detection":
- **Task A** (handwritten digits): Needs sharp edge detection → Large adaptation
- **Task B** (blurry images): Needs soft edge detection → Small adaptation

**Meta-SGD**: Must pick one learning rate α_edge for all tasks → Compromise
**LEO**: Encodes "edge detection" into latent space, each task gets different z_τ that implicitly controls the adaptation magnitude

### Visual Intuition

```
┌─────────────────────────────────────────────────────────────┐
│                    MAML vs Meta-SGD vs LEO                  │
└─────────────────────────────────────────────────────────────┘

MAML: Fixed learning rate for all parameters
  θ₁ ──(α)──→ θ'₁
  θ₂ ──(α)──→ θ'₂    Same α for all tasks, all params
  θ₃ ──(α)──→ θ'₃
  ...

Meta-SGD: Per-parameter learning rates
  θ₁ ──(α₁)──→ θ'₁
  θ₂ ──(α₂)──→ θ'₂   Different αᵢ per param, but fixed per task
  θ₃ ──(α₃)──→ θ'₃
  ...

LEO: Task-conditional latent optimization
  θ₁ ┐
  θ₂ ├─→ Encoder ──→ z_τ ──(adapt)──→ z'_τ ──→ Decoder ──→ θ'₁
  θ₃ ┤                                                      θ'₂
  ...┘                                                      θ'₃
                                                            ...
       ↑                                                     ↑
   Compress                                              Expand
   (task-specific)                                    (task-aware)
```

### The Latent Space is a Learned Coordinate System

Think of the latent space as a **coordinate system optimized for task adaptation**:

- **Original parameter space**: Axes are individual parameters (w₁, w₂, ..., w₁₀₀₀₀₀)
  - Messy, high-dimensional, no structure
  
- **Latent space**: Axes are learned basis vectors (z₁, z₂, ..., z₆₄)
  - z₁ might represent "low-level edge features"
  - z₂ might represent "texture complexity"
  - z₃ might represent "object scale"
  - Each dimension captures a **meaningful adaptation direction**

When you do gradient descent in latent space:
```python
z' = z - α ∇_z L_τ(Decoder(z))
```

You're moving along these meaningful axes, not arbitrary parameter directions!

### Summary: LEO's Key Advantages

1. **Task-Conditional Adaptation**: Each task gets its own latent code z_τ that captures task-specific structure

2. **Implicit Parameter Selection**: The decoder learns which parameters need adaptation without explicit per-parameter learning rates

3. **Low-Dimensional Optimization**: 64D latent space vs. 100K+ parameter space → Faster, more efficient

4. **Learned Basis**: Optimization happens in a coordinate system designed for task adaptation, not arbitrary parameter space

5. **Flexible Adaptation**: The same parameter can adapt differently across tasks based on the latent code

**In essence**: LEO doesn't just learn *how fast* to adapt each parameter (Meta-SGD). It learns *which combinations of parameters matter for each task* and adapts those combinations efficiently in a compressed space.

---

## 🔗 Why is the Relation Network Important?

The **Relation Network** is a critical component of LEO that provides considerable performance improvements. Here's why it matters:

### The Fundamental Question

When creating an embedding to classify a specific class, should you consider what the **other classes** in the task are?

**Answer: Absolutely!** 🎯

### Intuitive Example: Cats vs. Dogs

Imagine you need to create an embedding to classify a **cat**. Now consider two scenarios:

#### Scenario A: Cat vs. Dog
```
Classes in task: [Cat, Dog]

Your embedding should focus on:
  - Facial structure (cat: round face, dog: elongated snout)
  - Ears (cat: pointy, dog: floppy)
  - Body proportions (cat: compact, dog: varies)
  
The fact that "dog" is the other class tells you:
  "I should emphasize features that distinguish felines from canines"
```

#### Scenario B: Cat vs. Tiger
```
Classes in task: [Cat, Tiger]

Your embedding should focus on:
  - Size (cat: small, tiger: large)
  - Stripes (cat: solid/tabby, tiger: bold stripes)
  - Facial markings (subtle differences)
  
The fact that "tiger" is the other class tells you:
  "I should emphasize features that distinguish domestic cats from big cats"
```

**Key Insight**: The **same class** (cat) needs **different embeddings** depending on what it's being compared against!

- **Cat vs. Dog**: Emphasize feline vs. canine features
- **Cat vs. Tiger**: Emphasize size and marking differences
- **Cat vs. Bird**: Emphasize mammal vs. avian features

### How the Relation Network Helps

Without considering other classes, you might create a **generic cat embedding** that captures:
- "Has whiskers"
- "Has fur"
- "Has four legs"

But this embedding might not be **discriminative** enough for the specific task!

**The Relation Network solves this** by:

1. **Taking the latent code of a class** (e.g., cat)
2. **Comparing it with all other classes** in the task (dog, bird, fish, etc.)
3. **Refining the latent code** based on these comparisons

### Mathematical Intuition

For each class `i`, the relation network computes:

```python
# For class i (e.g., cat)
refined_code_i = average([
    relation(cat, cat),    # Self-relation: "What defines me?"
    relation(cat, dog),    # "How am I different from dog?"
    relation(cat, bird),   # "How am I different from bird?"
    relation(cat, fish),   # "How am I different from fish?"
    relation(cat, tiger),  # "How am I different from tiger?"
])
```

Each `relation()` call asks:
> "Given that class i exists alongside class j in this task, what aspects of class i should I emphasize?"

### Why This Makes Embeddings Better

The relation network provides **context-aware refinement**:

1. **Discriminative Power**: Embeddings focus on features that actually matter for *this specific task*
   - Cat vs. Dog → Emphasize species-level differences
   - Cat vs. Tiger → Emphasize size and marking differences

2. **Task-Specific Adaptation**: The same class gets different embeddings for different tasks
   - Generic "cat" embedding → Task-specific "cat vs. these particular classes" embedding

3. **Relational Context**: Each class's embedding is informed by the entire task structure
   - "I'm a cat, and in this task I need to be distinguished from a dog and a bird"
   - vs. "I'm a cat, and in this task I need to be distinguished from a tiger and a lion"

4. **Noise Reduction**: Averaging relations across all classes smooths out noise
   - Outlier relations have less impact
   - Consensus emerges from multiple comparisons

### Visual Intuition

```
┌──────────────────────────────────────────────────────────────┐
│              Without Relation Network                        │
└──────────────────────────────────────────────────────────────┘

Cat  → Generic Embedding [whiskers, fur, 4 legs, ...]
Dog  → Generic Embedding [wet nose, tail, 4 legs, ...]
Bird → Generic Embedding [feathers, wings, beak, ...]

Problem: Embeddings don't know what they're competing against!
         Features might not be discriminative for this task.


┌──────────────────────────────────────────────────────────────┐
│               With Relation Network ⭐                        │
└──────────────────────────────────────────────────────────────┘

Cat  → Consider all classes → Refined Embedding
       ├─ relation(cat, cat)  → self-identity
       ├─ relation(cat, dog)  → emphasize feline features
       └─ relation(cat, bird) → emphasize mammal features
       
Dog  → Consider all classes → Refined Embedding
       ├─ relation(dog, dog)  → self-identity
       ├─ relation(dog, cat)  → emphasize canine features
       └─ relation(dog, bird) → emphasize mammal features

Bird → Consider all classes → Refined Embedding
       ├─ relation(bird, bird) → self-identity
       ├─ relation(bird, cat)  → emphasize avian features
       └─ relation(bird, dog)  → emphasize flying/beak features

Result: Task-specific, discriminative embeddings!
```

### Concrete Example: 5-Way Classification

Task: Classify {Cat, Dog, Bird, Fish, Tiger}

**For the "Cat" class:**

```python
# Initial latent code (from encoder)
z_cat = encoder(cat_images)  # Generic cat features

# Relation network refinement
refined_z_cat = average([
    relation_net(z_cat, z_cat),   # "I'm a small feline"
    relation_net(z_cat, z_dog),   # "Unlike dog, I'm feline"
    relation_net(z_cat, z_bird),  # "Unlike bird, I'm mammal, no wings"
    relation_net(z_cat, z_fish),  # "Unlike fish, I have fur, legs"
    relation_net(z_cat, z_tiger), # "Unlike tiger, I'm small, different markings"
])

# Now refined_z_cat emphasizes:
#   - Feline features (vs. dog)
#   - Mammalian features (vs. bird, fish)
#   - Domestic/small size (vs. tiger)
```

**Result**: The cat embedding is now **optimized for distinguishing cats from dogs, birds, fish, and tigers** — not just representing "cat" in isolation!

### Performance Impact

**With Relation Network**: 
- Latent codes are **context-aware**
- Embeddings emphasize **discriminative features**
- Classification is more **robust and accurate**
- Provides **considerable performance improvements**

**Without Relation Network**:
- Latent codes are **isolated**
- Embeddings might focus on **non-discriminative features**
- Classification may struggle with **similar classes**
- Lower accuracy, especially on challenging tasks

### Implementation Detail

In our LEO implementation, the relation network:

1. Takes each class's aggregated latent code [64]
2. Computes relation with all N classes (including itself)
3. Averages N relation outputs to get refined code
4. Returns refined latent codes for all classes

```python
def apply_relation_network(self, class_codes):
    """Refine each class code by comparing with all other classes."""
    refined_codes = []
    
    for i in range(num_classes):
        relations = []
        for j in range(num_classes):
            # "How does class i relate to class j in this task?"
            relation = self.relation_net(class_codes[i], class_codes[j])
            relations.append(relation)
        
        # Average relations to get refined code
        refined_code = torch.stack(relations).mean(dim=0)
        refined_codes.append(refined_code)
    
    return torch.stack(refined_codes)
```

### Summary: Why Relation Network Matters

1. **Context is crucial**: Knowing what you're competing against helps you focus on the right features

2. **Task-specific embeddings**: Same class → different embeddings for different tasks

3. **Discriminative power**: Emphasizes features that actually distinguish classes in *this* task

4. **Relational reasoning**: "I'm X, and I need to be different from Y and Z" → Better X representation

5. **Performance boost**: Provides considerable improvements in accuracy and robustness

**Bottom line**: The relation network transforms generic embeddings into task-aware, discriminative representations by leveraging cross-class relationships. It's not just "what is this class?" but **"what is this class *in the context of these other classes*?"** 🎯

---

## Introduction

This document provides a comprehensive overview of the Latent Embedding Optimization (LEO) implementation for few-shot learning in PyTorch.

## Algorithm Overview

LEO is a meta-learning algorithm that learns to adapt to new tasks by optimizing in a learned low-dimensional latent space rather than directly in the high-dimensional parameter space. This makes optimization more efficient and effective, especially for very few-shot scenarios.

### Key Components

1. **Encoder**: Maps input examples to latent representations
2. **Relation Network**: Processes relationships between encoded examples
3. **Decoder**: Generates model parameters from latent codes
4. **Classifier**: Task-specific CNN that uses decoded parameters

### Algorithm Flow

```
Initialize encoder E, decoder D, relation network R, shared CNN φ

For each batch of tasks:
    For each task τᵢ:
        # Encode support set to latent space
        z = E(support_set)
        
        # Inner loop: Optimize in latent space
        for k steps:
            θ_fc = D(z)                      # Decode to FC parameters
            features = φ(support_set)        # Extract features with shared CNN
            L = loss(θ_fc, features, labels) # Compute loss
            z = z - α∇_z L                   # Update latent code
        
        # Decode final latent code to FC parameters
        θ_fc' = D(z)
        
        # Evaluate on query set
        features_query = φ(query_set)
        L_query = loss(θ_fc', features_query, query_labels)
    
    # Outer loop: Update encoder, decoder, relation network, shared CNN
    E, D, R, φ = E, D, R, φ - β∇_(E,D,R,φ) Σᵢ L_query
```

## Architecture Details

### LEOEncoder

Convolutional neural network that extracts features from input images and projects them to latent space.

**Architecture:**
- Conv2d(1, 64, 3x3) + BatchNorm + ReLU + MaxPool
- Conv2d(64, 64, 3x3) + BatchNorm + ReLU + MaxPool  
- Conv2d(64, 64, 3x3) + BatchNorm + ReLU + MaxPool
- Conv2d(64, 64, 3x3) + BatchNorm + ReLU + MaxPool
- Flatten
- Linear(64×6×6, latent_dim)

**Input:** Images [batch_size, 1, 105, 105]  
**Output:** Latent codes [batch_size, latent_dim]

### LEORelationNetwork

Processes pairs of latent codes to capture task structure.

**Architecture:**
- Linear(latent_dim × 2, latent_dim) + ReLU
- Linear(latent_dim, latent_dim)

**Input:** Two latent codes [batch_size, latent_dim] each  
**Output:** Relation output [batch_size, latent_dim]

### LEODecoder

Generates only the final FC layer parameters from latent codes. The CNN feature 
extractor is shared across all tasks and meta-learned jointly.

**Architecture:**
- Linear(latent_dim, 256) + ReLU
- Linear(256, 512) + ReLU
- Linear(512, total_param_count)

**Generated Parameters (FC layer only):**
```python
fc.weight: [num_classes, 2304]   # num_classes × 2304 params
fc.bias: [num_classes]            # num_classes params
```

**Total Generated Parameters:** num_classes × 2304 + num_classes = num_classes × 2305

For 5-way classification: **11,525 parameters** (vs 111,680 for full model)

### LEOClassifier

CNN with **shared feature extractor** and task-specific FC layer. The convolutional
layers are fixed and shared across all tasks, while the FC layer parameters are
generated by the decoder for each task.

**Shared Feature Extractor (meta-learned):**
- Conv2d(1, 64, 3x3) + BatchNorm + ReLU + MaxPool
- Conv2d(64, 64, 3x3) + BatchNorm + ReLU + MaxPool
- Conv2d(64, 64, 3x3) + BatchNorm + ReLU + MaxPool
- Conv2d(64, 64, 3x3) + BatchNorm + ReLU + MaxPool
- Flatten → 2304-dim features

**Task-Specific FC Layer (decoder-generated):**
- Linear(2304, num_classes)

**Forward Pass:**
```python
# Extract features (shared CNN)
features = classifier.extract_features(x)  # [batch, 2304]

# Use default FC layer
logits = classifier(x)  # [batch, num_classes]

# Use decoder-generated FC layer
fc_params = decoder(latent_codes)
logits = classifier(x, fc_params)  # [batch, num_classes]
```

## Usage Examples

### Basic Training

```python
import torch
from torch.utils.data import DataLoader
from algorithms.leo import train_leo
from utils.load_omniglot import load_omniglot_dataset, OmniglotTaskDataset

# Load dataset
train_dataset, test_dataset = load_omniglot_dataset(root='omniglot')

# Create task dataset
train_task_dataset = OmniglotTaskDataset(
    train_dataset,
    n_way=5,
    k_shot=1,
    num_query=15,
    num_tasks=2000
)

# Create dataloader
train_loader = DataLoader(
    train_task_dataset,
    batch_size=4,
    shuffle=True
)

# Train LEO
leo, losses = train_leo(
    num_classes=5,
    task_dataloader=train_loader,
    latent_dim=64,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5
)
```

### Manual Training Loop

```python
from algorithms.leo import LatentEmbeddingOptimization

# Initialize LEO
leo = LatentEmbeddingOptimization(
    num_classes=5,
    latent_dim=64,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
leo = leo.to(device)

# Training loop
for task_batch in train_loader:
    support_data, support_labels, query_data, query_labels = task_batch
    
    # Move to device
    support_data = support_data.to(device)
    support_labels = support_labels.to(device)
    query_data = query_data.to(device)
    query_labels = query_labels.to(device)
    
    # Meta-training step
    loss = leo.meta_train_step(
        support_data,
        support_labels,
        query_data,
        query_labels
    )
    
    print(f"Loss: {loss:.4f}")
```

### Adaptation to New Tasks

```python
# Set to evaluation mode
leo.eval()

# Get new task
support_data, support_labels, query_data, query_labels = next(iter(test_loader))
support_data = support_data[0].to(device)  # First task in batch
support_labels = support_labels[0].to(device)
query_data = query_data[0].to(device)
query_labels = query_labels[0].to(device)

# Encode support set
initial_codes = leo.encode_task(support_data, support_labels)

# Optimize in latent space
adapted_codes = leo.inner_update(support_data, support_labels, initial_codes)

# Decode to parameters
adapted_params = leo.decode_to_params(adapted_codes)

# Evaluate on query set
query_logits = leo.classifier(query_data, adapted_params)
predictions = query_logits.argmax(dim=1)
accuracy = (predictions == query_labels).float().mean()

print(f"Query accuracy: {accuracy*100:.2f}%")
```

### Parameter Flattening and Unflattening

The decoder generates a flat parameter vector for the FC layer which is then unflattened:

```python
# In LEODecoder.forward()
flattened_params = self.fc3(x)  # Shape: [batch_size, total_param_count]

# Unflatten using parameter shapes (FC layer only)
param_dict = {}
idx = 0

for name, shape in self.param_shapes:
    param_size = np.prod(shape)
    param_dict[name] = flattened_params[idx:idx+param_size].reshape(shape)
    idx += param_size

# Example output for 5-way classification:
# param_dict = {
#     'fc.weight': Tensor[5, 2304],  # 11,520 params
#     'fc.bias': Tensor[5]            # 5 params
# }
# Total: 11,525 parameters generated from 64D latent code
```

## Hyperparameter Guidelines

### Latent Dimension (`latent_dim`)

- **Range:** 32-128
- **Default:** 64
- **Trade-offs:**
  - Smaller (32): More compression, faster, may lose information
  - Larger (128): More capacity, better performance, slower

### Inner Learning Rate (`inner_lr`)

- **Range:** 0.005-0.05
- **Default:** 0.01
- **Trade-offs:**
  - Lower: More stable adaptation, slower convergence
  - Higher: Faster adaptation, may be unstable

### Outer Learning Rate (`outer_lr`)

- **Range:** 0.0001-0.01
- **Default:** 0.001
- **Trade-offs:**
  - Lower: More stable meta-learning, slower convergence
  - Higher: Faster meta-learning, may be unstable

### Inner Steps (`inner_steps`)

- **Range:** 3-10
- **Default:** 5
- **Trade-offs:**
  - Fewer: Faster training, tests rapid adaptation
  - More: Better task performance, more compute

## Advantages of LEO

1. **Low-dimensional Optimization**: Operates in compact latent space (e.g., 64D) instead of high-dimensional parameter space

2. **Efficient Adaptation**: Gradient descent in latent space is more efficient than in parameter space

3. **Better for Few-shot**: Particularly effective for 1-shot learning scenarios

4. **Shared Feature Extractor**: CNN backbone is meta-learned and shared across all tasks, only task-specific FC layer is generated (11.5K params vs 112K)

5. **Learned Initialization**: Decoder learns to generate good FC layer initializations from compact latent codes

6. **Task Structure**: Relation network captures relationships between examples

## Comparison with MAML

| Aspect | MAML | LEO |
|--------|------|-----|
| Optimization Space | Parameter space (~112K) | Latent space (64D) |
| Generated Parameters | N/A (direct optimization) | FC layer only (11.5K) |
| Shared Components | None | CNN feature extractor |
| Adaptation | Direct SGD on all params | Latent optimization + decode |
| 1-shot Performance | Good | Excellent |
| Computational Cost | Higher | Lower |
| Memory Usage | Higher | Lower |

## Expected Performance

### Omniglot 5-way 1-shot

- **Training time:** ~15-25 minutes (2000 tasks, GPU)
- **Expected accuracy:** 95-98%
- **Memory usage:** ~4-6 GB GPU

### Omniglot 5-way 5-shot

- **Training time:** ~20-30 minutes (2000 tasks, GPU)
- **Expected accuracy:** 97-99%
- **Memory usage:** ~6-8 GB GPU

## Files

- `algorithms/leo.py`: Core LEO implementation
- `examples/leo_on_omniglot.py`: Training and evaluation example
- `docs/LEO.md`: This documentation

## References

1. Rusu, A. A., et al. (2019). "Meta-Learning with Latent Embedding Optimization." ICLR 2019.
   - Paper: https://arxiv.org/abs/1807.05960
   
2. Finn, C., et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
   - Paper: https://arxiv.org/abs/1703.03400

## Troubleshooting

### Issue: NaN Loss

**Solution:** Reduce inner_lr or outer_lr, use gradient clipping (already implemented)

### Issue: Low Accuracy

**Solution:** 
- Increase latent_dim (e.g., 128)
- Increase inner_steps (e.g., 7-10)
- Train for more tasks

### Issue: Out of Memory

**Solution:**
- Reduce batch_size
- Reduce latent_dim
- Use smaller inner_steps

### Issue: Slow Training

**Solution:**
- Reduce inner_steps
- Use smaller batch_size
- Use GPU if available

## Future Improvements

1. **Multi-step Relation Network**: Process multiple pairs for better task understanding
2. **Attention Mechanism**: Use attention to weight different examples
3. **Task Conditioning**: Condition decoder on task-specific information
4. **Progressive Latent Optimization**: Use different learning rates at different steps
5. **Mixture of Decoders**: Use multiple decoders for different task types
