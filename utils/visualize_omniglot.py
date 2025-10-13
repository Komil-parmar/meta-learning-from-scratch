import torch
import random
import matplotlib.pyplot as plt


def visualize_task_sample(task_dataset, task_idx=0):
    """
    Visualize a sample N-way K-shot task from the dataset.
    
    Creates a visualization showing both the support set (used for adaptation)
    and query set (used for evaluation) for a single few-shot learning task.
    The visualization displays one example per class from both sets in a grid
    layout with support images in the top row and query images in the bottom row.
    
    Args:
        task_dataset (OmniglotTaskDataset): An instance of OmniglotTaskDataset
            that generates few-shot learning tasks.
        task_idx (int, optional): Index of the task to visualize. Note that tasks
            are randomly generated, so different runs may show different characters
            even with the same index. Defaults to 0.
    
    Returns:
        None: Displays a matplotlib figure with the task visualization and prints
            task statistics to the console.
    
    Example:
        >>> task_dataset = OmniglotTaskDataset(omniglot_data, n_way=5, k_shot=1)
        >>> visualize_task_sample(task_dataset, task_idx=0)
    """
    # Get a sample task
    support_data, support_labels, query_data, query_labels = task_dataset[task_idx]
    
    n_way = len(torch.unique(support_labels))
    k_shot = (support_labels == 0).sum().item()
    k_query = (query_labels == 0).sum().item()
    
    print(f"📊 Task Structure: {n_way}-way {k_shot}-shot learning")
    print(f"📚 Support set: {support_data.shape[0]} images ({k_shot} per class)")
    print(f"🧪 Query set: {query_data.shape[0]} images ({k_query} per class)")
    print()
    
    # Create visualization
    fig, axes = plt.subplots(2, n_way, figsize=(15, 6))
    
    # Plot support set (top row)
    for class_idx in range(n_way):
        # Find first image of this class in support set
        mask = support_labels == class_idx
        img_idx = torch.where(mask)[0][0]
        img = support_data[img_idx].squeeze().numpy()
        
        axes[0, class_idx].imshow(img, cmap='gray')
        axes[0, class_idx].set_title(f'Class {class_idx}\n(Support)', fontweight='bold')
        axes[0, class_idx].axis('off')
    
    # Plot query set (bottom row) - show first query image per class
    for class_idx in range(n_way):
        # Find first image of this class in query set
        mask = query_labels == class_idx
        img_idx = torch.where(mask)[0][0]
        img = query_data[img_idx].squeeze().numpy()
        
        axes[1, class_idx].imshow(img, cmap='gray')
        axes[1, class_idx].set_title(f'Class {class_idx}\n(Query)', fontweight='bold')
        axes[1, class_idx].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, '📚 Support Set', fontsize=14, fontweight='bold', 
             rotation=90, verticalalignment='center')
    fig.text(0.02, 0.25, '🧪 Query Set', fontsize=14, fontweight='bold', 
             rotation=90, verticalalignment='center')
    
    plt.suptitle(f'🎯 Sample {n_way}-way {k_shot}-shot Task from Omniglot', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.show()
    
    print("✨ In MAML, the model will:")
    print("   1️⃣ Adapt using the Support Set (top row)")
    print("   2️⃣ Evaluate on the Query Set (bottom row)")
    print("   3️⃣ Learn to quickly recognize new character classes!")


def visualize_character_variations(dataset, num_chars=3, max_examples=10):
    """
    Visualize multiple handwritten examples of the same character classes.
    
    Creates a grid visualization showing different handwritten variations of
    randomly sampled character classes from the Omniglot dataset. This helps
    understand the intra-class variation that makes few-shot learning challenging.
    Each row shows different examples of the same character written by different
    people, demonstrating the variety in handwriting styles.
    
    Args:
        dataset (OmniglotDataset): An instance of OmniglotDataset containing
            the character images to visualize.
        num_chars (int, optional): Number of different character classes to display.
            Each character will be shown in a separate row. Defaults to 3.
        max_examples (int, optional): Maximum number of handwriting examples to show
            per character class. The actual number shown may be less if fewer
            examples are available. Defaults to 10.
    
    Returns:
        None: Displays a matplotlib figure showing the character variations and
            prints explanatory information to the console.
    
    Example:
        >>> omniglot_data = OmniglotDataset('data/omniglot')
        >>> visualize_character_variations(omniglot_data, num_chars=5, max_examples=15)
    
    Notes:
        - Each character in Omniglot typically has ~20 different handwritten examples
        - Characters are randomly sampled, so each call may show different characters
        - Empty cells appear if a character has fewer examples than max_examples
    """
    fig, axes = plt.subplots(num_chars, max_examples, figsize=(15, 5))
    
    # Sample random characters
    char_indices = random.sample(range(len(dataset)), num_chars)
    
    for row_idx, char_idx in enumerate(char_indices):
        char_images, _ = dataset[char_idx]
        num_examples = min(max_examples, len(char_images))
        
        for col_idx in range(max_examples):
            ax = axes[row_idx, col_idx] if num_chars > 1 else axes[col_idx]
            
            if col_idx < num_examples:
                img = char_images[col_idx].squeeze().numpy()
                ax.imshow(img, cmap='gray')
            else:
                ax.axis('off')
            
            if row_idx == 0:
                ax.set_title(f'Example {col_idx+1}', fontsize=10)
            
            if col_idx == 0:
                ax.set_ylabel(f'Character {row_idx+1}', fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('📝 Character Variations in Omniglot Dataset\n(Same character, different handwriting styles)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("🎨 Notice: Each character has ~20 different handwritten examples!")
    print("💡 This variation makes the few-shot learning task challenging and realistic.")