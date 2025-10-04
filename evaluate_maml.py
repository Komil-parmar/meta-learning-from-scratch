"""
Evaluation utilities for Model-Agnostic Meta-Learning (MAML).

This module provides dataset-independent evaluation functions for measuring
MAML performance on few-shot learning tasks. The functions work with any
task-based DataLoader that provides support and query sets.

Functions:
    - evaluate_maml: Evaluate MAML on a set of test tasks
    - plot_evaluation_results: Visualize evaluation metrics
    - plot_training_progress: Visualize training loss curves
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


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


def plot_evaluation_results(eval_results: Dict[str, Any], figsize: Tuple[int, int] = (15, 10)) -> np.ndarray:
    """
    Generate comprehensive visualizations of MAML evaluation results.
    
    Creates a 2x2 grid of plots showing different aspects of the evaluation:
    1. Before vs After Adaptation: Bar chart comparing mean accuracies
    2. Accuracy Distributions: Histograms showing accuracy spread
    3. Per-Task Improvement: Distribution of accuracy gains
    4. Loss vs Accuracy: Scatter plot showing correlation
    
    Args:
        eval_results (dict):
            Dictionary returned by evaluate_maml() containing evaluation metrics.
            Must contain keys: 'before_adaptation_accuracy', 'after_adaptation_accuracy',
            'before_adaptation_std', 'after_adaptation_std', 'all_accuracies',
            'all_before_accuracies', 'all_losses'.
            
        figsize (tuple, optional):
            Figure size as (width, height) in inches.
            Default: (15, 10)
    
    Returns:
        np.ndarray: Array of per-task improvement values (after - before accuracy).
                   Useful for further analysis of task difficulty.
    
    Example:
        >>> eval_results = evaluate_maml(model, maml, test_loader)
        >>> improvements = plot_evaluation_results(eval_results)
        >>> 
        >>> # Analyze task difficulty
        >>> hardest_tasks = np.argsort(improvements)[:5]
        >>> print(f"Hardest tasks: {hardest_tasks}")
        >>> 
        >>> # Custom figure size
        >>> plot_evaluation_results(eval_results, figsize=(20, 12))
    
    Plots Created:
        1. **Adaptation Effect** (top-left):
           - Bar chart with error bars
           - Compares before and after adaptation accuracy
           - Shows random baseline reference line
           - Displays mean ± std for each condition
        
        2. **Accuracy Distributions** (top-right):
           - Overlapping histograms (density normalized)
           - Red: before adaptation, Green: after adaptation
           - Shows full distribution of task performance
           - Helps identify bimodal distributions or outliers
        
        3. **Per-Task Improvement** (bottom-left):
           - Histogram of accuracy gains per task
           - Shows how many tasks improved vs degraded
           - Vertical line at zero for reference
           - Text box shows percentage of tasks that improved
        
        4. **Loss vs Accuracy** (bottom-right):
           - Scatter plot showing relationship
           - Each point is one task
           - Correlation coefficient displayed
           - Helps identify if loss is predictive of accuracy
    
    Notes:
        - All plots use consistent color schemes
        - Grid lines enhance readability
        - Text annotations provide key statistics
        - Figure is automatically displayed with plt.show()
        - Use plt.savefig() before calling this function to save
        
    Interpretation Tips:
        - Large improvement: Model adapts well to new tasks
        - Low variance after adaptation: Consistent performance
        - Negative correlation (loss vs acc): Well-calibrated model
        - Bimodal distribution: Some task types are harder
    
    See Also:
        - evaluate_maml: Generate the eval_results dictionary
        - plot_training_progress: Visualize training curves
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Before vs After Adaptation
    categories = ['Before\nAdaptation', 'After\nAdaptation']
    means = [eval_results['before_adaptation_accuracy'], eval_results['after_adaptation_accuracy']]
    stds = [eval_results['before_adaptation_std'], eval_results['after_adaptation_std']]
    
    bars = axes[0, 0].bar(categories, means, yerr=stds, capsize=5, 
                         color=['lightcoral', 'lightgreen'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Adaptation Effect', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim(0, 1)
    
    for bar, mean, std in zip(bars, means, stds):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02, 
                       f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Use random baseline from results if available
    random_baseline = eval_results.get('random_baseline', 0.2)
    axes[0, 0].axhline(y=random_baseline, color='red', linestyle='--', alpha=0.7, 
                      label=f'Random ({random_baseline:.1%})')
    axes[0, 0].legend()
    
    # 2. Accuracy distribution
    axes[0, 1].hist(eval_results['all_before_accuracies'], alpha=0.6, label='Before', 
                   color='red', bins=20, density=True)
    axes[0, 1].hist(eval_results['all_accuracies'], alpha=0.6, label='After', 
                   color='green', bins=20, density=True)
    axes[0, 1].set_xlabel('Accuracy', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Accuracy Distributions', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    random_baseline = eval_results.get('random_baseline', 0.2)
    axes[0, 1].axvline(x=random_baseline, color='red', linestyle='--', alpha=0.7)
    
    # 3. Task difficulty analysis
    task_improvements = np.array(eval_results['all_accuracies']) - np.array(eval_results['all_before_accuracies'])
    axes[1, 0].hist(task_improvements, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('Accuracy Improvement', fontsize=12)
    axes[1, 0].set_ylabel('Number of Tasks', fontsize=12)
    axes[1, 0].set_title('Per-Task Improvement', fontsize=14, fontweight='bold')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    positive_improvements = sum(1 for imp in task_improvements if imp > 0)
    axes[1, 0].text(0.05, 0.95, f'Tasks improved: {positive_improvements}/{len(task_improvements)} ({positive_improvements/len(task_improvements)*100:.1f}%)', 
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Performance vs Loss correlation
    axes[1, 1].scatter(eval_results['all_losses'], eval_results['all_accuracies'], 
                      alpha=0.6, color='purple', s=30, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Loss', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_title('Loss vs Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    correlation = np.corrcoef(eval_results['all_losses'], eval_results['all_accuracies'])[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.suptitle(f"MAML Evaluation Results ({eval_results.get('num_tasks', len(eval_results['all_accuracies']))} tasks)", 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    return task_improvements


def plot_training_progress(
    losses: List[float], 
    window_size: int = 50, 
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize MAML training progress with loss curves and distribution.
    
    Creates a 1x2 grid showing:
    1. Training loss over time with smoothed curve
    2. Distribution of loss values
    
    Args:
        losses (list[float]):
            List of loss values from training, one per meta-training step.
            Typically returned by train_maml() function.
            
        window_size (int, optional):
            Size of moving average window for smoothing the loss curve.
            Larger values create smoother curves but may hide short-term trends.
            Default: 50
            
        figsize (tuple, optional):
            Figure size as (width, height) in inches.
            Default: (12, 6)
    
    Returns:
        None: Displays the plot using plt.show()
    
    Example:
        >>> # After training
        >>> model, maml, losses = train_maml(model, train_loader)
        >>> plot_training_progress(losses)
        >>> 
        >>> # Custom smoothing for noisy training
        >>> plot_training_progress(losses, window_size=100)
        >>> 
        >>> # Larger figure for presentations
        >>> plot_training_progress(losses, figsize=(16, 8))
    
    Plots Created:
        1. **Training Loss** (left):
           - Raw loss values (semi-transparent blue line)
           - Smoothed moving average (solid red line)
           - Shows learning progress over training steps
           - Grid for easier value reading
        
        2. **Loss Distribution** (right):
           - Histogram of loss values
           - Shows typical loss range and outliers
           - Helps identify training stability
           - Green bars with black edges for visibility
    
    Training Insights:
        - Decreasing trend: Model is learning
        - Plateau: May need longer training or different hyperparameters
        - High variance: Consider reducing learning rate
        - Narrow distribution: Stable training
        - Bimodal distribution: May indicate task diversity
    
    Statistics Printed:
        - Initial loss: Mean of first 10 steps
        - Final loss: Mean of last 100 steps
        - Improvement: Absolute decrease in loss
    
    Notes:
        - Requires matplotlib.pyplot
        - Smoothing only applied if len(losses) > window_size
        - Statistics use min(100, len(losses)) for averaging
        - Figure is automatically displayed
        
    See Also:
        - train_maml: Training function that produces loss history
        - plot_evaluation_results: Visualize evaluation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Training loss over time
    axes[0].plot(losses, alpha=0.4, color='blue', label='Loss', linewidth=1)
    
    # Add smoothed curve if enough data points
    if len(losses) > window_size:
        smoothed = [np.mean(losses[max(0, i-window_size):i+1]) for i in range(len(losses))]
        axes[0].plot(smoothed, color='red', linewidth=2, label=f'Smoothed (window={window_size})')
    
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('MAML Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss distribution
    axes[1].hist(losses, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Loss Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Loss Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    initial_window = min(10, len(losses))
    final_window = min(100, len(losses))
    
    print(f"\nTraining Statistics:")
    print(f"   Total steps: {len(losses)}")
    print(f"   Initial loss (first {initial_window}): {np.mean(losses[:initial_window]):.4f}")
    print(f"   Final loss (last {final_window}): {np.mean(losses[-final_window:]):.4f}")
    print(f"   Improvement: {np.mean(losses[:initial_window]) - np.mean(losses[-final_window:]):.4f}")
    print(f"   Min loss: {np.min(losses):.4f}")
    print(f"   Max loss: {np.max(losses):.4f}")
