#!/usr/bin/env python3
"""
Test script to verify PrefetchedOmniglotDataset implementation.

This script demonstrates:
1. Loading the prefetched dataset
2. Performance comparison with standard dataset
3. Integration with OmniglotTaskDataset
4. DataLoader compatibility
"""

import sys
import time
import torch
from torch.utils.data import DataLoader

sys.path.append('..')
from utils.load_omniglot import OmniglotDataset, PrefetchedOmniglotDataset, OmniglotTaskDataset


def test_basic_loading():
    """Test basic dataset loading."""
    print("="*70)
    print("TEST 1: Basic Loading")
    print("="*70)
    
    data_path = "/mnt/c/meta-learning-from-scratch/omniglot/images_background"
    
    print("\nLoading with PrefetchedOmniglotDataset...")
    dataset = PrefetchedOmniglotDataset(data_path)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total character classes: {len(dataset)}")
    
    # Test access
    images, idx = dataset[0]
    print(f"   Sample shape: {images.shape}")
    print(f"   Sample dtype: {images.dtype}")
    print(f"   Value range: [{images.min():.2f}, {images.max():.2f}]")
    
    assert len(dataset) == 964, "Background set should have 964 characters"
    assert images.shape[1:] == (1, 105, 105), "Images should be [N, 1, 105, 105]"
    assert images.min() >= 0 and images.max() <= 1, "Values should be normalized [0, 1]"
    
    print("   ✅ All assertions passed!")
    return dataset


def test_performance_comparison(num_samples=100):
    """Compare performance between standard and prefetched datasets."""
    print("\n" + "="*70)
    print("TEST 2: Performance Comparison")
    print("="*70)
    
    data_path = "/mnt/c/meta-learning-from-scratch/omniglot/images_background"
    
    # Test standard dataset
    print(f"\n1️⃣ Testing standard OmniglotDataset ({num_samples} accesses)...")
    standard = OmniglotDataset(data_path)
    
    start = time.time()
    for i in range(num_samples):
        _ = standard[i]
    standard_time = time.time() - start
    
    print(f"   Time: {standard_time:.3f}s ({standard_time/num_samples*1000:.2f}ms per access)")
    
    # Test prefetched dataset
    print(f"\n2️⃣ Testing PrefetchedOmniglotDataset ({num_samples} accesses)...")
    prefetched = PrefetchedOmniglotDataset(data_path)
    
    start = time.time()
    for i in range(num_samples):
        _ = prefetched[i]
    prefetched_time = time.time() - start
    
    print(f"   Time: {prefetched_time:.3f}s ({prefetched_time/num_samples*1000:.2f}ms per access)")
    
    # Calculate speedup
    speedup = standard_time / prefetched_time
    print(f"\n⚡ Speedup: {speedup:.1f}x faster!")
    
    if speedup > 5:
        print("   ✅ Excellent performance boost!")
    elif speedup > 2:
        print("   ✅ Good performance improvement!")
    else:
        print("   ⚠️  Lower than expected speedup (may vary by system)")
    
    return prefetched


def test_task_dataset_integration(dataset):
    """Test integration with OmniglotTaskDataset."""
    print("\n" + "="*70)
    print("TEST 3: Task Dataset Integration")
    print("="*70)
    
    print("\nCreating OmniglotTaskDataset...")
    task_dataset = OmniglotTaskDataset(
        dataset,
        n_way=5,
        k_shot=1,
        k_query=15,
        num_tasks=10
    )
    
    print(f"   Number of tasks: {len(task_dataset)}")
    
    # Test task generation
    print("\nGenerating a sample task...")
    support, support_labels, query, query_labels = task_dataset[0]
    
    print(f"   Support shape: {support.shape} (expected: [5, 1, 105, 105])")
    print(f"   Support labels: {support_labels}")
    print(f"   Query shape: {query.shape} (expected: [75, 1, 105, 105])")
    print(f"   Query labels shape: {query_labels.shape}")
    
    # Assertions
    assert support.shape == (5, 1, 105, 105), "Support set shape incorrect"
    assert query.shape == (75, 1, 105, 105), "Query set shape incorrect"
    assert len(set(support_labels.tolist())) == 5, "Should have 5 classes"
    
    print("   ✅ All assertions passed!")
    
    return task_dataset


def test_dataloader_compatibility(task_dataset):
    """Test DataLoader compatibility."""
    print("\n" + "="*70)
    print("TEST 4: DataLoader Compatibility")
    print("="*70)
    
    print("\nCreating DataLoader...")
    dataloader = DataLoader(
        task_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Batch size: 2")
    print(f"   Num workers: 2")
    
    # Test iteration
    print("\nIterating through first batch...")
    start = time.time()
    for support, support_labels, query, query_labels in dataloader:
        batch_time = time.time() - start
        print(f"   Support batch shape: {support.shape}")
        print(f"   Query batch shape: {query.shape}")
        print(f"   Batch loading time: {batch_time*1000:.2f}ms")
        break
    
    print("   ✅ DataLoader works correctly!")


def test_memory_usage():
    """Display memory usage information."""
    print("\n" + "="*70)
    print("TEST 5: Memory Usage")
    print("="*70)
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"\n   RAM Usage: {memory_info.rss / 1e9:.2f} GB")
    except ImportError:
        print("\n   psutil not installed - cannot measure RAM usage")
    
    if torch.cuda.is_available():
        print(f"   GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   GPU Memory Peak: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("   GPU not available")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 PREFETCHED OMNIGLOT DATASET TESTS")
    print("="*70)
    
    try:
        # Test 1: Basic loading
        dataset = test_basic_loading()
        
        # Test 2: Performance comparison
        dataset = test_performance_comparison(num_samples=100)
        
        # Test 3: Task dataset integration
        task_dataset = test_task_dataset_integration(dataset)
        
        # Test 4: DataLoader compatibility
        test_dataloader_compatibility(task_dataset)
        
        # Test 5: Memory usage
        test_memory_usage()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎉 PrefetchedOmniglotDataset is working correctly!")
        print("   Ready to use for faster training!")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
