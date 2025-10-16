# Meta-SGD: Learning to Learn with Personalized Learning Rates

## 💡 Intuition: Every Parameter Deserves Its Own Teacher

Imagine a classroom where every student learns at a different pace. Some grasp concepts quickly and need to slow down to avoid overshooting. Others need more aggressive instruction to make progress. A good teacher adapts their approach to each student individually.

**Meta-SGD applies this same principle to neural network parameters.** Instead of using a single learning rate for all parameters (like teaching all students the same way), Meta-SGD learns a **personalized learning rate for each parameter**. Some weights need large updates to adapt quickly, while others need tiny, careful adjustments.

This is the key insight: **Just as every student needs personalized teaching, every parameter needs a personalized learning rate to converge optimally.**

---

## 🎯 What is Meta-SGD?

Meta-SGD extends MAML by making the **inner loop learning rates learnable parameters** themselves. While MAML uses a fixed learning rate `α` for all parameters during task adaptation, Meta-SGD learns:

- **θ (meta-parameters)**: The model weights (what MAML learns)
- **α (meta-learning-rates)**: Per-parameter learning rates (Meta-SGD's addition!)

During meta-training, both θ and α are optimized to enable fast adaptation.

---

## 🔍 How It Works

### Standard MAML Inner Loop:
```python
# Fixed learning rate for all parameters
θ' = θ - α * ∇L(θ)  # α is the same for all parameters
```

### Meta-SGD Inner Loop:
```python
# Learnable per-parameter learning rates
θ' = θ - α ⊙ ∇L(θ)  # α is a vector (one learning rate per parameter)
```

### What Changes:

1. **Initialization**: Create learnable learning rates
   ```python
   self.meta_sgd_lrs = torch.nn.ParameterList([
       torch.nn.Parameter(torch.tensor(inner_lr)) 
       for _ in self.model.parameters()
   ])
   ```

2. **Inner Loop Update**: Use per-parameter learning rates
   ```python
   # Standard MAML
   θ' = θ - 0.01 * grad  # Same 0.01 for all
   
   # Meta-SGD
   θ' = θ - α[i] * grad  # Different α[i] for each parameter
   ```

3. **Outer Loop Update**: Optimize both θ and α
   ```python
   meta_optimizer = Adam(
       list(model.parameters()) + list(meta_sgd_lrs.parameters())
   )
   ```

---

## 🚀 How to Use Meta-SGD

### Basic Usage:
```python
from algorithms.maml import train_maml, ModelAgnosticMetaLearning

# Enable Meta-SGD by setting meta_sgd=True
model, maml, losses = train_maml(
    model=model,
    task_dataloader=task_loader,
    inner_lr=0.01,      # Initial value for all learning rates
    outer_lr=0.001,     # Meta-learning rate
    inner_steps=5,
    meta_sgd=True,      # 🔥 Enable Meta-SGD
    first_order=False   # ⚠️ MUST be False (see below)
)
```

### Direct API:
```python
# Create Meta-SGD trainer
maml = ModelAgnosticMetaLearning(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5,
    meta_sgd=True,
    first_order=False  # Required!
)

# Train as usual
for task_batch in task_loader:
    loss = maml.meta_train_step(
        support_data, support_labels,
        query_data, query_labels
    )
```

---

## ⚠️ Why First-Order Approximation is Incompatible

### The Fundamental Problem:

**Learning rates are NOT part of the forward pass.** They only appear during the parameter update step:

```python
# Forward pass (compute loss)
logits = model(x)           # Learning rates α not involved
loss = F.cross_entropy(logits, y)  # Learning rates α not involved

# Backward pass (update parameters)
θ' = θ - α ⊙ ∇L(θ)          # Learning rates α used HERE
```

### Why This Matters:

To update the learning rates α, we need gradients: **∂(query_loss)/∂α**

But the learning rates only affect the query loss *indirectly* through their influence on θ':

```
α → θ' → query_loss
    ↑
    This dependency requires 2nd-order gradients!
```

To compute ∂(query_loss)/∂α, we need:
```python
∂(query_loss)/∂α = ∂(query_loss)/∂θ' × ∂θ'/∂α
                                          ↑
                              This requires gradients
                              through the inner loop!
```

### The Chain of Dependencies:

1. **Learning rates (α)** affect how we update **parameters (θ')**
2. **Parameters (θ')** affect the **query loss**
3. Therefore: **Learning rates (α)** affect **query loss** through **parameters (θ')**

This creates a computational graph:
```
α (learning rates) → θ' (adapted params) → query_loss
```

### Why First-Order Fails:

**First-Order MAML (FOMAML)** breaks this chain by using `.detach()`:
```python
# FOMAML detaches θ' from θ
θ' = θ - α * ∇L(θ)
θ' = θ'.detach()  # ❌ Breaks gradient flow!

# Now α cannot receive gradients because:
query_loss → θ' ✗ α  (gradient flow blocked)
```

### The Catch-2:

Even if we try to keep gradients for α while using first-order for θ:

```python
# Attempt: First-order for θ, second-order for α
θ' = θ - α * ∇L(θ)
θ' = θ'.detach().requires_grad_(True)  # Detach θ only

# Problem: We still need the full computation graph!
# ∂θ'/∂α requires knowing how θ' was computed from θ
# This means keeping the ENTIRE gradient graph anyway!
```

**Result**: You get all the computational cost of second-order MAML with none of the benefits of first-order approximation. The computation graph must be maintained regardless, making first-order approximation pointless.

### Time Complexity Analysis:

| Method | Computation Graph | Speed | Learning Rates |
|--------|------------------|-------|----------------|
| **MAML** | Full (2nd-order) | Slow | Fixed α |
| **FOMAML** | None (1st-order) | Fast ⚡ | Fixed α |
| **Meta-SGD** | Full (2nd-order) | Slow | Learnable α ✨ |
| **Meta-SGD + FOMAML** | Full (needed for α!) | Slow 😞 | Learnable α |

**Conclusion**: Meta-SGD + FOMAML gives you the worst of both worlds—full computational cost with limited benefits.

---

## 🎓 Implementation Details

### What the Code Does:

1. **Initialize per-parameter learning rates** (in `__init__`):
   ```python
   self.meta_sgd_lrs = torch.nn.ParameterList([
       torch.nn.Parameter(torch.tensor(inner_lr, requires_grad=True))
       for _ in self.model.parameters()
   ])
   ```

2. **Clone learning rates for task adaptation** (in `inner_update`):
   ```python
   fast_sgd_lrs_list = [param.clone() for param in self.meta_sgd_lrs.parameters()]
   ```

3. **Use per-parameter learning rates for updates**:
   ```python
   fast_weights_list = vectorized_param_update(
       fast_weights_list,
       grads,
       fast_sgd_lrs_list  # Different α for each parameter
   )
   ```

4. **Meta-optimizer updates both θ and α**:
   ```python
   self.meta_optimizer = optimizer_cls(
       list(self.model.parameters()) + list(self.meta_sgd_lrs.parameters())
   )
   ```

### Key Design Choices:

- **Learning rates initialized to `inner_lr`**: Starts from MAML baseline
- **Learning rates are cloned per task**: Each task gets its own α trajectory
- **Gradients computed with `create_graph=True`**: Enables meta-learning of α
- **Both θ and α in meta-optimizer**: Both updated via outer loop

---

## 📊 Expected Benefits

### Advantages:
- ✅ **Faster adaptation**: Parameters can adapt at their optimal rate
- ✅ **Better final performance**: Typically 2-5% accuracy improvement over MAML
- ✅ **Automatic hyperparameter tuning**: No manual tuning of learning rates per layer

### Trade-offs:
- ⚠️ **2× parameters**: Doubles parameter count (θ + α)
- ⚠️ **Cannot use first-order approximation**: Must use full second-order gradients
- ⚠️ **Slightly more memory**: Stores per-parameter learning rates

### When to Use:
- ✅ You want maximum adaptation performance
- ✅ You have sufficient GPU memory
- ✅ Training time is not the primary bottleneck
- ❌ Don't use if you need fast training (use FOMAML instead)

---

## 📚 Reference

**Meta-SGD: Learning to Learn Quickly for Few-Shot Learning**  
Zhenguo Li, Fengwei Zhou, Fei Chen, Hang Li  
arXiv:1707.09835 (2017)  
https://arxiv.org/abs/1707.09835

---

## 💡 Key Takeaways

1. **Intuition**: Each parameter learns at its own optimal rate (like personalized teaching)
2. **Implementation**: Add learnable per-parameter learning rates
3. **Constraint**: Requires second-order gradients (no first-order approximation)
4. **Trade-off**: Better performance vs. more computation and memory
5. **Use when**: You prioritize adaptation quality over training speed

**Remember**: Just like a great teacher adapts to each student, Meta-SGD adapts to each parameter! 🎓✨
