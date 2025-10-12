"""
Evaluation utilities for ANIL (Almost No Inner Loop).

This module provides functions to evaluate ANIL models on meta-learning tasks,
measuring performance before and after task-specific adaptation.

Functions:
    evaluate_anil: Evaluate ANIL on test tasks with comprehensive metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Any


def evaluate_anil(
    body: torch.nn.Module,
    head: torch.nn.Module,
    anil,
    eval_dataloader: torch.utils.data.DataLoader,
    num_classes: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate ANIL on test tasks.

    This function measures the model's performance before and after task-specific
    adaptation, providing comprehensive statistics about the adaptation capability.

    Args:
        body (torch.nn.Module): Feature extractor network
        head (torch.nn.Module): Classifier network
        anil: ANIL trainer instance
        eval_dataloader (DataLoader): DataLoader for evaluation tasks
            Should yield batches of (support_data, support_labels, query_data, query_labels)
        num_classes (int): Number of classes per task (default: 5)
        verbose (bool): Whether to print progress and results (default: True)

    Returns:
        dict: Dictionary containing evaluation metrics compatible with plot_evaluation_results

    Example:
        >>> body, head = create_anil_network(num_classes=5)
        >>> anil = ANIL(body, head, inner_lr=0.01, outer_lr=0.001)
        >>> # ... train model ...
        >>> eval_results = evaluate_anil(body, head, anil, eval_dataloader)
        >>> print(f"Accuracy: {eval_results['after_adaptation_accuracy']:.1%}")
    """
    device = next(head.parameters()).device
    body.eval()
    head.eval()

    # Storage for results
    before_accuracies = []
    after_accuracies = []
    before_losses = []
    after_losses = []

    if verbose:
        print("🧪 Evaluating ANIL on test tasks...")
        eval_iter = tqdm(eval_dataloader, desc="Evaluation", dynamic_ncols=True)
    else:
        eval_iter = eval_dataloader

    # Evaluate before adaptation (no gradients needed)
    with torch.no_grad():
        for task_batch in eval_iter:
            support_data, support_labels, query_data, query_labels = task_batch

            # Move to device
            support_data = support_data.squeeze(0).to(device)
            support_labels = support_labels.squeeze(0).to(device)
            query_data = query_data.squeeze(0).to(device)
            query_labels = query_labels.squeeze(0).to(device)

            # Evaluate BEFORE adaptation
            body.eval()
            features = body(query_data)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)

            # Classify with head
            logits_before = head(features)
            loss_before = F.cross_entropy(logits_before, query_labels)
            pred_before = torch.argmax(logits_before, dim=1)
            acc_before = (pred_before == query_labels).float().mean().item()

            before_accuracies.append(acc_before)
            before_losses.append(loss_before.item())

    # Reset iterator for after-adaptation evaluation
    if verbose:
        eval_iter = tqdm(eval_dataloader, desc="Post-adaptation eval", dynamic_ncols=True)
    else:
        eval_iter = eval_dataloader

    # Now perform adaptation and evaluate (need gradients for inner loop)
    for task_batch in eval_iter:
        support_data, support_labels, query_data, query_labels = task_batch

        # Move to device
        support_data = support_data.squeeze(0).to(device)
        support_labels = support_labels.squeeze(0).to(device)
        query_data = query_data.squeeze(0).to(device)
        query_labels = query_labels.squeeze(0).to(device)

        # Adapt head to task (inner loop)
        adapted_head = anil.inner_update(support_data, support_labels)

        # Evaluate AFTER adaptation
        with torch.no_grad():
            logits_after = anil.forward_with_head(query_data, adapted_head)
            loss_after = F.cross_entropy(logits_after, query_labels)
            pred_after = torch.argmax(logits_after, dim=1)
            acc_after = (pred_after == query_labels).float().mean().item()

        after_accuracies.append(acc_after)
        after_losses.append(loss_after.item())

    # Compute statistics
    before_acc_mean = np.mean(before_accuracies)
    after_acc_mean = np.mean(after_accuracies)
    before_acc_std = np.std(before_accuracies)
    after_acc_std = np.std(after_accuracies)

    improvements = [after - before for after, before in zip(after_accuracies, before_accuracies)]
    improvement_mean = np.mean(improvements)
    improvement_std = np.std(improvements)

    # Create results dict with keys expected by plot_evaluation_results
    results = {
        # Summary statistics
        'before_adaptation_accuracy': before_acc_mean,
        'after_adaptation_accuracy': after_acc_mean,
        'improvement': improvement_mean,
        'before_adaptation_loss': np.mean(before_losses),
        'after_adaptation_loss': np.mean(after_losses),
        'before_adaptation_std': before_acc_std,
        'after_adaptation_std': after_acc_std,
        'improvement_std': improvement_std,

        # Per-task data (for backward compatibility)
        'before_accuracies': before_accuracies,
        'after_accuracies': after_accuracies,
        'before_losses': before_losses,
        'after_losses': after_losses,
        'improvements': improvements,

        # Keys expected by plot_evaluation_results
        'all_before_accuracies': before_accuracies,
        'all_accuracies': after_accuracies,
        'all_losses': after_losses,

        # Additional metadata
        'num_tasks': len(before_accuracies),
        'random_baseline': 1.0 / num_classes
    }

    if verbose:
        print(f"\n{'='*70}")
        print("📊 EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"❌ Before Adaptation: {before_acc_mean:.1%} ± {before_acc_std:.1%}")
        print(f"✅ After Adaptation:  {after_acc_mean:.1%} ± {after_acc_std:.1%}")
        print(f"🚀 Improvement:       {improvement_mean:.1%} ± {improvement_std:.1%}")
        print(f"📈 Loss reduction:    {results['before_adaptation_loss']:.4f} → {results['after_adaptation_loss']:.4f}")
        print(f"📦 Tasks evaluated:   {results['num_tasks']}")
        print(f"{'='*70}")

    return results
