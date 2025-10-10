"""
Compare MAML vs FOMAML performance and training speed.

This script demonstrates the difference between full MAML (second-order) and
FOMAML (first-order) in terms of training speed, memory usage, and accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import sys
sys.path.append('.')

from algorithms.maml import train_maml
from utils.load_omniglot import OmniglotDataset, OmniglotTaskDataset
from evaluation.evaluate_maml import evaluate_maml

class SimpleConvNet(nn.Module):
    """Simple CNN for Omniglot classification"""
    def __init__(self, num_classes=5):
        super(SimpleConvNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, num_classes)
        )
        
    def forward(self, x):
        return self.sequential(x)


def compare_algorithms():
    """Compare MAML and FOMAML on a small subset of Omniglot"""
    
    print("="*80)
    print("MAML vs FOMAML Comparison")
    print("="*80)
    
    # Setup data
    print("\n📂 Loading Omniglot dataset...")
    train_dataset = OmniglotDataset("omniglot/images_background")
    eval_dataset = OmniglotDataset("omniglot/images_evaluation")
    
    # Small training set for quick comparison
    train_task_dataset = OmniglotTaskDataset(
        train_dataset, 
        n_way=5, 
        k_shot=1, 
        k_query=15, 
        num_tasks=200  # Small for quick comparison
    )
    
    eval_task_dataset = OmniglotTaskDataset(
        eval_dataset, 
        n_way=5, 
        k_shot=1, 
        k_query=15, 
        num_tasks=100
    )
    
    train_loader = DataLoader(train_task_dataset, batch_size=4, shuffle=True)
    eval_loader = DataLoader(eval_task_dataset, batch_size=1, shuffle=False)
    
    results = {}
    
    # Test both algorithms
    for algorithm_name, use_first_order in [("MAML", False), ("FOMAML", True)]:
        print(f"\n{'='*80}")
        print(f"Training with {algorithm_name}")
        print(f"{'='*80}")
        
        # Create fresh model
        model = SimpleConvNet(num_classes=5)
        
        # Train
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        trained_model, maml, losses = train_maml(
            model=model,
            task_dataloader=train_loader,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=5,
            first_order=use_first_order,
            use_amp=False
        )
        
        training_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        # Evaluate
        print(f"\n🧪 Evaluating {algorithm_name}...")
        eval_results = evaluate_maml(
            model=trained_model,
            maml=maml,
            eval_dataloader=eval_loader,
            num_classes=5,
            verbose=False
        )
        
        # Store results
        results[algorithm_name] = {
            'training_time': training_time,
            'peak_memory_gb': peak_memory,
            'final_train_loss': losses[-10:],
            'test_accuracy': eval_results['after_adaptation_accuracy'],
            'test_improvement': eval_results['improvement']
        }
        
        print(f"\n✅ {algorithm_name} Results:")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Peak Memory: {peak_memory:.2f} GB")
        print(f"   Final Train Loss: {sum(losses[-10:])/10:.4f}")
        print(f"   Test Accuracy: {eval_results['after_adaptation_accuracy']:.4f}")
        print(f"   Improvement: {eval_results['improvement']:.4f}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    maml_results = results['MAML']
    fomaml_results = results['FOMAML']
    
    speedup = maml_results['training_time'] / fomaml_results['training_time']
    memory_reduction = (1 - fomaml_results['peak_memory_gb'] / maml_results['peak_memory_gb']) * 100
    accuracy_diff = maml_results['test_accuracy'] - fomaml_results['test_accuracy']
    
    print(f"\n⚡ Speed:")
    print(f"   MAML:   {maml_results['training_time']:.2f}s")
    print(f"   FOMAML: {fomaml_results['training_time']:.2f}s")
    print(f"   ➜ FOMAML is {speedup:.2f}x faster")
    
    print(f"\n💾 Memory:")
    print(f"   MAML:   {maml_results['peak_memory_gb']:.2f} GB")
    print(f"   FOMAML: {fomaml_results['peak_memory_gb']:.2f} GB")
    print(f"   ➜ FOMAML uses {memory_reduction:.1f}% less memory")
    
    print(f"\n🎯 Accuracy:")
    print(f"   MAML:   {maml_results['test_accuracy']:.4f}")
    print(f"   FOMAML: {fomaml_results['test_accuracy']:.4f}")
    print(f"   ➜ Difference: {accuracy_diff:.4f} ({accuracy_diff/maml_results['test_accuracy']*100:.2f}%)")
    
    print(f"\n💡 Recommendation:")
    if accuracy_diff < 0.03 and speedup > 1.2:
        print("   ✅ FOMAML is recommended: Similar accuracy with significant speedup")
    elif accuracy_diff > 0.05:
        print("   ✅ MAML is recommended: Significantly better accuracy")
    else:
        print("   ⚖️ Both are viable: Choose based on your priorities (speed vs accuracy)")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    compare_algorithms()
