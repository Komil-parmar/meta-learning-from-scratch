"""
General evaluation and visualization utilities for meta-learning algorithms.

This module provides algorithm-agnostic visualization functions that can be used
with any meta-learning algorithm (MAML, Reptile, Prototypical Networks, etc.).
These functions work with standardized evaluation results dictionaries.

Functions:
    - plot_evaluation_results: Visualize evaluation metrics (before/after adaptation)
    - plot_training_progress: Visualize training loss curves
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


def plot_evaluation_results(eval_results: Dict[str, Any], figsize: Tuple[int, int] = (15, 10)) -> np.ndarray:
    """
    Generate comprehensive visualizations of meta-learning evaluation results.
    
    Creates a 2x2 grid of plots showing different aspects of the evaluation:
    1. Before vs After Adaptation: Bar chart comparing mean accuracies
    2. Accuracy Distributions: Histograms showing accuracy spread
    3. Per-Task Improvement: Distribution of accuracy gains
    4. Loss vs Accuracy: Scatter plot showing correlation
    
    This function is algorithm-agnostic and works with any meta-learning algorithm
    that produces evaluation results in the expected format.
    
    Args:
        eval_results (dict):
            Dictionary containing evaluation metrics with the following keys:
            - 'before_adaptation_accuracy' (float): Mean accuracy before adaptation
            - 'after_adaptation_accuracy' (float): Mean accuracy after adaptation
            - 'before_adaptation_std' (float): Std of accuracy before adaptation
            - 'after_adaptation_std' (float): Std of accuracy after adaptation
            - 'all_accuracies' (list): Per-task accuracy after adaptation
            - 'all_before_accuracies' (list): Per-task accuracy before adaptation
            - 'all_losses' (list): Per-task losses
            - 'random_baseline' (float, optional): Random baseline accuracy
            - 'num_tasks' (int, optional): Number of tasks evaluated
            
        figsize (tuple, optional):
            Figure size as (width, height) in inches.
            Default: (15, 10)
    
    Returns:
        np.ndarray: Array of per-task improvement values (after - before accuracy).
                   Useful for further analysis of task difficulty.
    
    Example:
        >>> # Works with any meta-learning algorithm
        >>> eval_results = evaluate_maml(model, maml, test_loader)
        >>> improvements = plot_evaluation_results(eval_results)
        >>> 
        >>> # Or with other algorithms
        >>> eval_results = evaluate_reptile(model, reptile, test_loader)
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
        - evaluate_maml: Generate eval_results for MAML (in evaluate_maml.py)
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
    
    plt.suptitle(f"Meta-Learning Evaluation Results ({eval_results.get('num_tasks', len(eval_results['all_accuracies']))} tasks)", 
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
    Visualize meta-learning training progress with loss curves and distribution.
    
    Creates a 1x2 grid showing:
    1. Training loss over time with smoothed curve
    2. Distribution of loss values
    
    This function is algorithm-agnostic and can be used with any meta-learning
    algorithm that produces a list of training losses.
    
    Args:
        losses (list[float]):
            List of loss values from training, one per meta-training step.
            Can be from any meta-learning algorithm (MAML, Reptile, etc.).
            
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
        >>> # After training with MAML
        >>> model, maml, losses = train_maml(model, train_loader)
        >>> plot_training_progress(losses)
        >>> 
        >>> # After training with other algorithms
        >>> model, reptile, losses = train_reptile(model, train_loader)
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
        - plot_evaluation_results: Visualize evaluation metrics
        - train_maml: MAML training function (in MAML.py)
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
    axes[0].set_title('Meta-Learning Training Progress', fontsize=14, fontweight='bold')
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
