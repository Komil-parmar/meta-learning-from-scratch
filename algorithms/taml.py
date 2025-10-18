"""
Task Agnostic Meta Learning (TAML) Implementation - FIXED VERSION
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, List, Tuple


class TAMLClassifier(nn.Module):
    """CNN classifier for TAML with shared feature extractor."""
    
    def __init__(self, num_classes: int = 5):
        super(TAMLClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Shared feature extractor
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
        
        # Task-specific Head
        self.head = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using shared CNN body."""
        return self.body(x)
    
    def forward(self, x: torch.Tensor, head_weights: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional parameter modulations or explicit head weights/biases.
        Args:
            x: input tensor
            modulations: dict of modulation tensors (applied to self.head params)
            head_weights: dict of explicit weights/biases (overrides self.head params)
        """
        features = self.body(x)
        if head_weights is not None:
            # Use explicit fast weights (from inner loop)
            x = F.linear(features, head_weights['weight1'], head_weights['bias1'])
            x = F.relu(x)
            logits = F.linear(x, head_weights['weight2'], head_weights['bias2'])
        else:
            logits = self.head(features)
        return logits


class TAMLEncoder(nn.Module):
    """Encoder network that maps CNN features to latent representations."""
    
    def __init__(self, latent_dim: int = 64, feature_dim: int = 2304):
        super(TAMLEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        return self.fc(features)


class TAMLRelationNetwork(nn.Module):
    """Relation network that processes pairs of encoded examples."""
    
    def __init__(self, latent_dim: int = 64):
        super(TAMLRelationNetwork, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(latent_dim * 2, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute relation between two latent codes."""
        x = torch.cat([z1, z2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TAMLDecoder(nn.Module):
    """Decoder network that predicts parameter modulations and learning rates."""
    
    def __init__(self, latent_dim: int = 64, approach: str = 'per_layer_scalars'):
        super(TAMLDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.approach = approach
        
        if approach == 'global_scalars':
            self.modulation_net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh()
            )
            self.lr_net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
        elif approach == 'per_layer_scalars':
            self.modulation_net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Tanh()
            )
            self.lr_net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Sigmoid()
            )
            
        elif approach == 'per_parameter_via_broadcasting':
            self.layer1_mod_net = nn.Sequential(
                nn.Linear(latent_dim, 128 + 2304 + 128)
            )
            self.layer2_mod_net = nn.Sequential(
                nn.Linear(latent_dim, 5 + 128 + 5)
            )
            self.layer1_lr_net = nn.Sequential(
                nn.Linear(latent_dim, 128 * 2304 + 128)
            )
            self.layer2_lr_net = nn.Sequential(
                nn.Linear(latent_dim, 5 * 128 + 5)
            )
            
        elif approach == 'low_rank_factorization':
            self.low_rank_dim = 32
            self.alpha_net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.low_rank_dim)
            )
            self.V_weight1 = nn.Parameter(torch.randn(128 * 2304, self.low_rank_dim) * 0.01)
            self.V_bias1 = nn.Parameter(torch.randn(128, self.low_rank_dim) * 0.01)
            self.V_weight2 = nn.Parameter(torch.randn(5 * 128, self.low_rank_dim) * 0.01)
            self.V_bias2 = nn.Parameter(torch.randn(5, self.low_rank_dim) * 0.01)
            self.layer1_lr_net = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128 * 2304 + 128)
            )
            self.layer2_lr_net = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 5 * 128 + 5)
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def forward(self, z: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Decode latent codes to modulations and learning rates.
        
        CRITICAL: NO .clone().requires_grad_() - it breaks gradient flow!
        """
        if self.approach == 'global_scalars':
            # Modulation: map [-1,1] -> [0.5, 1.5] centered at 1.0
            global_mod = self.modulation_net(z).squeeze()
            global_mod = global_mod * 0.5 + 1.0
            
            modulations = {
                'head_weight1': global_mod,
                'head_bias1': global_mod,
                'head_weight2': global_mod,
                'head_bias2': global_mod
            }
            
            # Learning rate: map [0,1] -> [0.001, 0.1]
            global_lr = self.lr_net(z).squeeze()
            global_lr = global_lr * 0.099 + 0.001
            
            learning_rates = {
                'head_weight1': global_lr,
                'head_bias1': global_lr,
                'head_weight2': global_lr,
                'head_bias2': global_lr
            }
            
        elif self.approach == 'per_layer_scalars':
            # Modulations: map [-1,1] -> [0.5, 1.5]
            layer_mods = self.modulation_net(z).squeeze()
            layer_mods = layer_mods * 0.5 + 1.0
            
            modulations = {
                'head_weight1': layer_mods[0],
                'head_bias1': layer_mods[1],
                'head_weight2': layer_mods[2],
                'head_bias2': layer_mods[3]
            }
            
            # Learning rates: map [0,1] -> [0.001, 0.1]
            layer_lrs = self.lr_net(z).squeeze()
            layer_lrs = layer_lrs * 0.099 + 0.001
            
            learning_rates = {
                'head_weight1': layer_lrs[0],
                'head_bias1': layer_lrs[0],
                'head_weight2': layer_lrs[1],
                'head_bias2': layer_lrs[1]
            }
            
        elif self.approach == 'per_parameter_via_broadcasting':
            # Layer 1
            layer1_params = self.layer1_mod_net(z).squeeze()
            idx = 0
            α_out1 = layer1_params[idx:idx+128]; idx += 128
            α_in1 = layer1_params[idx:idx+2304]; idx += 2304
            β_out1 = layer1_params[idx:idx+128]
            
            head_weight1_mod = α_out1[:, None] * α_in1[None, :]
            head_bias1_mod = β_out1
            
            # Layer 2
            layer2_params = self.layer2_mod_net(z).squeeze()
            idx = 0
            α_out2 = layer2_params[idx:idx+5]; idx += 5
            α_in2 = layer2_params[idx:idx+128]; idx += 128
            β_out2 = layer2_params[idx:idx+5]
            
            head_weight2_mod = α_out2[:, None] * α_in2[None, :]
            head_bias2_mod = β_out2
            
            modulations = {
                'head_weight1': head_weight1_mod,
                'head_bias1': head_bias1_mod,
                'head_weight2': head_weight2_mod,
                'head_bias2': head_bias2_mod
            }
            
            # Learning rates
            layer1_lrs = torch.sigmoid(self.layer1_lr_net(z).squeeze()) * 0.099 + 0.001
            layer2_lrs = torch.sigmoid(self.layer2_lr_net(z).squeeze()) * 0.099 + 0.001
            
            learning_rates = {
                'head_weight1': layer1_lrs[:128*2304].view(128, 2304),
                'head_bias1': layer1_lrs[128*2304:128*2304+128],
                'head_weight2': layer2_lrs[:5*128].view(5, 128),
                'head_bias2': layer2_lrs[5*128:5*128+5]
            }
            
        elif self.approach == 'low_rank_factorization':
            α_code = self.alpha_net(z).squeeze()
            
            head_weight1_mod = (self.V_weight1 @ α_code).view(128, 2304)
            head_bias1_mod = self.V_bias1 @ α_code
            head_weight2_mod = (self.V_weight2 @ α_code).view(5, 128)
            head_bias2_mod = self.V_bias2 @ α_code
            
            modulations = {
                'head_weight1': head_weight1_mod,
                'head_bias1': head_bias1_mod,
                'head_weight2': head_weight2_mod,
                'head_bias2': head_bias2_mod
            }
            
            # Learning rates
            layer1_lrs = torch.sigmoid(self.layer1_lr_net(z).squeeze()) * 0.099 + 0.001
            layer2_lrs = torch.sigmoid(self.layer2_lr_net(z).squeeze()) * 0.099 + 0.001
            
            learning_rates = {
                'head_weight1': layer1_lrs[:128*2304].view(128, 2304),
                'head_bias1': layer1_lrs[128*2304:128*2304+128],
                'head_weight2': layer2_lrs[:5*128].view(5, 128),
                'head_bias2': layer2_lrs[5*128:5*128+5]
            }
        
        return modulations, learning_rates


class TaskAgnosticMetaLearning:
    """Task Agnostic Meta Learning (TAML) algorithm for few-shot learning."""
    
    def __init__(
        self,
        num_classes: int = 5,
        latent_dim: int = 64,
        modulation_approach: str = 'per_layer_scalars',
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        optimizer_cls: type = optim.Adam,
        optimizer_kwargs: dict = None
    ):
        """Initialize TAML algorithm."""
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.modulation_approach = modulation_approach
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        
        # Initialize networks
        self.encoder = TAMLEncoder(latent_dim=latent_dim)
        self.decoder = TAMLDecoder(latent_dim=latent_dim, approach=modulation_approach)
        self.relation_net = TAMLRelationNetwork(latent_dim=latent_dim)
        self.classifier = TAMLClassifier(num_classes=num_classes)
        self.latent_aggregator = nn.Linear(latent_dim * num_classes, latent_dim)
        
        # Meta-parameters
        self.meta_params = (
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.relation_net.parameters()) + 
            list(self.classifier.parameters()) +
            list(self.latent_aggregator.parameters())
        )
        
        self.meta_optimizer = None
        
    def to(self, device: torch.device):
        """Move all networks to device."""
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.relation_net = self.relation_net.to(device)
        self.classifier = self.classifier.to(device)
        self.latent_aggregator = self.latent_aggregator.to(device)  # ✅ FIX: Added this
        return self
    
    def initialize_optimizer(self, device: torch.device = None):
        """Initialize optimizer after moving to device."""
        # Refresh meta-parameter list
        self.meta_params = (
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.relation_net.parameters()) + 
            list(self.classifier.parameters()) +
            list(self.latent_aggregator.parameters())
        )
        
        if device is not None:
            for p in self.meta_params:
                if p.data.device != device:
                    p.data = p.data.to(device)
        
        self.meta_optimizer = self.optimizer_cls(
            self.meta_params,
            lr=self.outer_lr,
            **self.optimizer_kwargs
        )
    
    def train(self):
        """Set all networks to training mode."""
        self.encoder.train()
        self.decoder.train()
        self.relation_net.train()
        self.classifier.train()
        self.latent_aggregator.train()
    
    def eval(self):
        """Set all networks to evaluation mode."""
        self.encoder.eval()
        self.decoder.eval()
        self.relation_net.eval()
        self.classifier.eval()
        self.latent_aggregator.eval()
    
    def apply_relation_network(self, class_codes: torch.Tensor) -> torch.Tensor:
        """Apply relation network to refine class latent codes."""
        num_classes = class_codes.size(0)
        refined_codes = []
        
        for i in range(num_classes):
            class_i = class_codes[i]
            relations = []
            
            for j in range(num_classes):
                class_j = class_codes[j]
                relation_output = self.relation_net(
                    class_i.unsqueeze(0),
                    class_j.unsqueeze(0)
                )
                relations.append(relation_output.squeeze(0))
            
            relations = torch.stack(relations, dim=0)
            refined_code = relations.mean(dim=0)
            refined_codes.append(refined_code)
        
        refined_codes = torch.stack(refined_codes, dim=0)
        return refined_codes
    
    def encode_task(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """Encode support set to latent codes."""
        # Extract features (keeping gradients for CNN learning)
        features = self.classifier.extract_features(support_data)
        
        # Encode to latent space
        latent_codes = self.encoder(features)
        
        # Aggregate per class
        class_codes = []
        for class_idx in range(self.num_classes):
            class_mask = support_labels == class_idx
            class_examples = latent_codes[class_mask]
            
            if len(class_examples) > 0:
                class_code = class_examples.mean(dim=0)
            else:
                class_code = torch.zeros(self.latent_dim, device=latent_codes.device)
            
            class_codes.append(class_code)
        
        class_codes = torch.stack(class_codes, dim=0)
        
        # Refine using relation network
        refined_codes = self.apply_relation_network(class_codes)
        
        return refined_codes
    
    def decode_to_modulations(self, latent_codes: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Decode latent codes to parameter modulations and learning rates."""
        # Aggregate with learnable weights
        task_latent = self.latent_aggregator(latent_codes.flatten())
        
        # Decode
        modulations, learning_rates = self.decoder(task_latent)
        
        return modulations, learning_rates
    
    def inner_update(
        self, 
        support_data: torch.Tensor, 
        support_labels: torch.Tensor,
        modulations: Dict[str, torch.Tensor],
        learning_rates: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        MAML-style inner loop: apply modulations ONCE to original head params, then update fast weights using decoder-provided learning rates.
        """
        # 1. Get original head params
        head_weight1 = self.classifier.head[0].weight
        head_bias1 = self.classifier.head[0].bias
        head_weight2 = self.classifier.head[2].weight
        head_bias2 = self.classifier.head[2].bias

        # 2. Apply modulations ONCE to get initial fast weights
        fast_weights = {
            'weight1': head_weight1 * modulations['head_weight1'],
            'bias1': head_bias1 * modulations['head_bias1'],
            'weight2': head_weight2 * modulations['head_weight2'],
            'bias2': head_bias2 * modulations['head_bias2']
        }

        # 3. Inner loop: update fast weights using learning rates
        for step in range(self.inner_steps):
            logits = self.classifier(support_data, head_weights=fast_weights)
            loss = F.cross_entropy(logits, support_labels)

            grads = torch.autograd.grad(
                loss,
                list(fast_weights.values()),
                create_graph=True,
                # retain_graph=(step < self.inner_steps - 1)
            )

            new_fast_weights = {}
            for i, key in enumerate(fast_weights.keys()):
                lr = learning_rates['head_' + key] if 'head_' + key in learning_rates else learning_rates[key]
                grad = grads[i]
                new_fast_weights[key] = fast_weights[key] - lr * grad
            fast_weights = new_fast_weights

        return fast_weights
    
    def meta_train_step(
        self,
        support_data_batch: torch.Tensor,
        support_labels_batch: torch.Tensor,
        query_data_batch: torch.Tensor,
        query_labels_batch: torch.Tensor
    ) -> float:
        """Perform one meta-training step with debugging.
        Fixes RuntimeError: Trying to backward through the graph a second time.
        """
        self.train()
        if self.meta_optimizer is None:
            raise RuntimeError("Optimizer not initialized.")

        self.meta_optimizer.zero_grad()

        device = next(self.encoder.parameters()).device
        meta_loss_sum = torch.tensor(0.0, device=device)

        batch_size = support_data_batch.size(0)

        for i in range(batch_size):
            support_data = support_data_batch[i]
            support_labels = support_labels_batch[i]
            query_data = query_data_batch[i]
            query_labels = query_labels_batch[i]

            # Encode support set
            latent_codes = self.encode_task(support_data, support_labels)

            # Decode to modulations and learning rates
            modulations, learning_rates = self.decode_to_modulations(latent_codes)

            # Inner loop: Optimize fast weights (MAML-style)
            fast_weights = self.inner_update(
                support_data, support_labels, modulations, learning_rates
            )

            # Evaluate on query set
            query_logits = self.classifier(query_data, head_weights=fast_weights)
            query_loss = F.cross_entropy(query_logits, query_labels)

            # Accumulate meta loss as a list to avoid graph issues
            meta_loss_sum += query_loss

        meta_loss = meta_loss_sum / batch_size

        # Backpropagate (only once per batch)
        meta_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.meta_params, max_norm=10.0)

        # Meta-update
        self.meta_optimizer.step()

        # Print detailed stats for first batch
        if hasattr(self, '_batch_count'):
            self._batch_count += 1
        else:
            self._batch_count = 0

        return meta_loss.item()


# Convenience alias
TAML = TaskAgnosticMetaLearning


def train_taml(
    num_classes: int,
    task_dataloader: torch.utils.data.DataLoader,
    latent_dim: int = 64,
    modulation_approach: str = 'per_layer_scalars',
    inner_steps: int = 5,
    outer_lr: float = 0.001,
    optimizer_cls: type = optim.Adam,
    optimizer_kwargs: dict = None,
    device: torch.device = None
):
    """Train using Task Agnostic Meta Learning (TAML)."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize TAML
    taml = TaskAgnosticMetaLearning(
        num_classes=num_classes,
        latent_dim=latent_dim,
        modulation_approach=modulation_approach,
        inner_steps=inner_steps,
        outer_lr=outer_lr,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs
    )
    
    # Move to device
    taml = taml.to(device)
    
    # Initialize optimizer
    taml.initialize_optimizer(device=device)
    
    print(f"\nStarting TAML training...")
    print(f"Hyperparameters:")
    print(f"  latent_dim={latent_dim}")
    print(f"  modulation_approach={modulation_approach}")
    print(f"  inner_steps={inner_steps}")
    print(f"  outer_lr={outer_lr}")
    print(f"  optimizer={optimizer_cls.__name__}\n")
    
    losses = []
    best_loss = float('inf')
    
    progress_bar = tqdm(task_dataloader, desc="Training", dynamic_ncols=True)
    
    for batch_idx, task_batch in enumerate(progress_bar):
        # Unpack task batch
        support_data_batch, support_labels_batch, query_data_batch, query_labels_batch = task_batch
        
        # Move to device
        support_data_batch = support_data_batch.to(device)
        support_labels_batch = support_labels_batch.to(device)
        query_data_batch = query_data_batch.to(device)
        query_labels_batch = query_labels_batch.to(device)
        
        # Meta-training step
        loss = taml.meta_train_step(
            support_data_batch,
            support_labels_batch,
            query_data_batch,
            query_labels_batch
        )
        
        losses.append(loss)
        
        # Update best loss
        if loss < best_loss:
            best_loss = loss
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss:.4f}',
            'best': f'{best_loss:.4f}',
            'avg': f'{np.mean(losses[-100:]):.4f}' if len(losses) >= 100 else f'{np.mean(losses):.4f}'
        })
        
        # Periodic logging
        if (batch_idx + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            print(f"\nStep {batch_idx + 1}:")
            print(f"  Current Loss: {loss:.4f}")
            print(f"  Avg Loss (last 100): {avg_loss:.4f}")
            print(f"  Best Loss: {best_loss:.4f}\n")
    
    print(f"\nTraining completed!")
    print(f"Final avg loss: {np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses):.4f}")
    print(f"Best loss: {best_loss:.4f}")
    
    return taml, losses