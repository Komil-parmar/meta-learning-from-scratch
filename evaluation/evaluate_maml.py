"""
Evaluation utilities for Model-Agnostic Meta-Learning (MAML).

This module provides MAML-specific evaluation functions for measuring
performance on few-shot learning tasks. The function works with any
task-based DataLoader that provides support and query sets.

For visualization utilities (plot_evaluation_results, plot_training_progress),
see utils/evaluate.py - these are algorithm-agnostic and can be used with
any meta-learning algorithm.

Functions:
    - evaluate_maml: Evaluate MAML on a set of test tasks
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Any


def evaluate_maml(
    model: torch.nn.Module, 
    maml, 
    eval_dataloader: torch.utils.data.DataLoader,
    num_classes: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate MAML performance on new unseen tasks.
    
    This function measures the model's ability to quickly adapt to new tasks by
    comparing performance before and after task-specific adaptation. It computes
    comprehensive metrics including accuracy, loss, and improvement statistics.
    
    The evaluation process:
        1. For each task in the dataloader:
           a. Measure baseline accuracy (before adaptation)
           b. Perform inner loop adaptation on support set
           c. Measure post-adaptation accuracy on query set
        2. Aggregate statistics across all tasks
        3. Return comprehensive evaluation metrics
    
    Args:
        model (torch.nn.Module):
            The meta-trained model to evaluate. Should be on the appropriate device
            (CPU/GPU) before calling this function. The model will be set to eval mode.
            
        maml (MAML):
            The MAML trainer object containing the inner_update and forward_with_weights
            methods. This should be the same MAML instance used during training, or a
            new instance with the same hyperparameters.
            
        eval_dataloader (torch.utils.data.DataLoader):
            DataLoader yielding evaluation tasks. Each batch should contain:
            - support_data: Training examples for adaptation [batch_size, N*K, ...]
            - support_labels: Labels for support set [batch_size, N*K]
            - query_data: Test examples for evaluation [batch_size, N*Q, ...]
            - query_labels: Labels for query set [batch_size, N*Q]
            
            Note: Batch size should typically be 1 for evaluation to assess
            per-task performance accurately.
            
        num_classes (int, optional):
            Number of classes per task (N-way). Used for computing random baseline
            accuracy in the output. If None, will be inferred from query labels.
            Default: None
            
        verbose (bool, optional):
            Whether to print detailed evaluation results to console.
            Default: True
    
    Returns:
        dict: A dictionary containing comprehensive evaluation metrics:
            - 'after_adaptation_accuracy' (float): Mean accuracy after adaptation
            - 'after_adaptation_std' (float): Standard deviation of post-adaptation accuracy
            - 'before_adaptation_accuracy' (float): Mean accuracy before adaptation
            - 'before_adaptation_std' (float): Standard deviation of pre-adaptation accuracy
            - 'improvement' (float): Mean accuracy improvement (after - before)
            - 'average_loss' (float): Mean cross-entropy loss on query sets
            - 'all_accuracies' (list): Per-task accuracy after adaptation
            - 'all_before_accuracies' (list): Per-task accuracy before adaptation
            - 'all_losses' (list): Per-task losses
            - 'num_tasks' (int): Number of tasks evaluated
    
    Example:
        >>> # Basic evaluation
        >>> eval_results = evaluate_maml(model, maml, test_loader)
        >>> print(f"Accuracy: {eval_results['after_adaptation_accuracy']:.2%}")
        >>> 
        >>> # Custom evaluation with specific parameters
        >>> eval_results = evaluate_maml(
        ...     model, 
        ...     maml, 
        ...     test_loader,
        ...     num_classes=5,
        ...     verbose=False
        ... )
        >>> 
        >>> # Access detailed metrics
        >>> improvement = eval_results['improvement']
        >>> best_task_acc = max(eval_results['all_accuracies'])
    
    Notes:
        - The model is set to eval() mode automatically
        - All computations use torch.no_grad() for efficiency
        - Support and query data are automatically moved to model's device
        - Random baseline is 1/N for N-way classification
        - Typical runtime: 1-2 seconds per task on GPU
        
    Performance Expectations (N-way K-shot):
        - 5-way 1-shot: 60-90% post-adaptation accuracy
        - 5-way 5-shot: 75-95% post-adaptation accuracy
        - 20-way 1-shot: 40-60% post-adaptation accuracy
        - Improvement: typically 40-60% absolute gain from baseline
    
    Raises:
        RuntimeError: If batch structure doesn't match expected format
        ValueError: If dataloader is empty
        
    See Also:
        - plot_evaluation_results: Visualize the returned metrics
        - MAML.inner_update: Task adaptation method
        - MAML.forward_with_weights: Forward pass with custom weights
    """
    model.eval()
    device = next(model.parameters()).device
    
    accuracies = []
    before_adaptation_accuracies = []
    losses = []
    
    # Infer num_classes if not provided
    if num_classes is None:
        # Will be set from first batch
        num_classes = None
    
    for support_data, support_labels, query_data, query_labels in tqdm(
        eval_dataloader, desc="Evaluating", disable=not verbose
    ):
        # Move to device
        support_data = support_data.squeeze(0).to(device)
        support_labels = support_labels.squeeze(0).to(device)
        query_data = query_data.squeeze(0).to(device)
        query_labels = query_labels.squeeze(0).to(device)
        
        # Infer num_classes from first batch if not provided
        if num_classes is None:
            num_classes = len(torch.unique(query_labels))
        
        # Evaluate before adaptation
        with torch.no_grad():
            initial_logits = model(query_data)
            initial_predictions = torch.argmax(initial_logits, dim=1)
            before_accuracy = (initial_predictions == query_labels).float().mean().item()
            before_adaptation_accuracies.append(before_accuracy)
        
        # Adapt to support set
        fast_weights = maml.inner_update(support_data, support_labels)
        
        # Evaluate after adaptation
        with torch.no_grad():
            query_logits = maml.forward_with_weights(query_data, fast_weights)
            query_loss = F.cross_entropy(query_logits, query_labels)
            predictions = torch.argmax(query_logits, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
            
            accuracies.append(accuracy)
            losses.append(query_loss.item())
    
    # Validate that we have results
    if len(accuracies) == 0:
        raise ValueError("Evaluation dataloader is empty or returned no valid batches")
    
    # Calculate metrics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_before_accuracy = np.mean(before_adaptation_accuracies)
    std_before_accuracy = np.std(before_adaptation_accuracies)
    avg_loss = np.mean(losses)
    improvement = avg_accuracy - avg_before_accuracy
    random_baseline = 1.0 / num_classes if num_classes else 0.2
    
    # Print results if verbose
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Tasks Evaluated: {len(accuracies)}")
        print(f"Task Structure: {num_classes}-way classification")
        print(f"")
        print(f"Before Adaptation:")
        print(f"   Average Accuracy: {avg_before_accuracy:.4f} ± {std_before_accuracy:.4f}")
        print(f"   (Random baseline: ~{random_baseline:.4f})")
        print(f"")
        print(f"After Adaptation:")
        print(f"   Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"")
        print(f"Improvement:")
        print(f"   Accuracy Gain: +{improvement:.4f} ({improvement/avg_before_accuracy*100:.1f}% relative)")
        print(f"   Tasks with >50% accuracy: {sum(1 for acc in accuracies if acc > 0.5)}/{len(accuracies)} ({sum(1 for acc in accuracies if acc > 0.5)/len(accuracies)*100:.1f}%)")
        print(f"   Tasks with >80% accuracy: {sum(1 for acc in accuracies if acc > 0.8)}/{len(accuracies)} ({sum(1 for acc in accuracies if acc > 0.8)/len(accuracies)*100:.1f}%)")
        print(f"{'='*70}")
    
    return {
        'after_adaptation_accuracy': avg_accuracy,
        'after_adaptation_std': std_accuracy,
        'before_adaptation_accuracy': avg_before_accuracy,
        'before_adaptation_std': std_before_accuracy,
        'improvement': improvement,
        'average_loss': avg_loss,
        'all_accuracies': accuracies,
        'all_before_accuracies': before_adaptation_accuracies,
        'all_losses': losses,
        'num_tasks': len(accuracies),
        'num_classes': num_classes,
        'random_baseline': random_baseline
    }