import torch
import random
import matplotlib.pyplot as plt


def visualize_task_sample(task_dataset, task_idx=0):
    """
    Visualize a sample task from the dataset showing support and query sets.
    
    Args:
        task_dataset: OmniglotTaskDataset instance
        task_idx: Index of the task to visualize
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
    Visualize multiple examples from the same character class to show variation.
    
    Args:
        dataset: OmniglotDataset instance
        num_chars: Number of different characters to show
        max_examples: Maximum number of examples to show per character
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