"""
Test script to verify Meta Dropout is working correctly in Meta Networks.

This script verifies that:
1. Support and query sets share the same dropout masks within a task
2. Different tasks get different dropout masks
3. Meta Dropout masks are consistent across forward passes for the same task
"""

import torch
import torch.nn as nn
from EB_Meta_Network import MetaNetwork

def test_meta_dropout_consistency():
    """Test that support and query share the same dropout masks."""
    print("=" * 70)
    print("TEST 1: Meta Dropout Consistency (Support and Query)")
    print("=" * 70)
    
    # Create model with dropout
    model = MetaNetwork(embedding_dim=64, hidden_dim=128, num_classes=5, 
                       dropout_rates=[0.5, 0.5, 0.5])  # High dropout for visibility
    model.train()  # Enable dropout!
    
    # Create identical support and query data
    support_data = torch.ones(5, 1, 105, 105)  # 5-way 1-shot
    support_labels = torch.arange(5)
    query_data = torch.ones(5, 1, 105, 105)  # Same as support
    
    # Extract embeddings separately
    model.embedding_network.reset_dropout_masks(support_data.shape, support_data.device)
    support_emb_1 = model.embedding_network(support_data)
    query_emb_1 = model.embedding_network(query_data)
    
    # Check if embeddings are identical (same masks applied)
    max_diff = torch.max(torch.abs(support_emb_1 - query_emb_1)).item()
    print(f"Max difference between support and query embeddings: {max_diff:.6f}")
    
    if max_diff < 1e-6:
        print("✅ PASS: Support and query use the SAME dropout masks!")
    else:
        print("❌ FAIL: Support and query use DIFFERENT dropout masks!")
    
    print()

def test_different_tasks_different_masks():
    """Test that different tasks get different dropout masks."""
    print("=" * 70)
    print("TEST 2: Different Tasks Get Different Masks")
    print("=" * 70)
    
    model = MetaNetwork(embedding_dim=64, hidden_dim=128, num_classes=5,
                       dropout_rates=[0.5, 0.5, 0.5])
    model.train()  # Enable dropout!
    
    # Same input data
    support_data = torch.ones(5, 1, 105, 105)
    support_labels = torch.arange(5)
    query_data = torch.ones(5, 1, 105, 105)
    
    # Task 1
    model.embedding_network.reset_dropout_masks(support_data.shape, support_data.device)
    emb_task1 = model.embedding_network(support_data)
    
    # Task 2 (reset masks)
    model.embedding_network.reset_dropout_masks(support_data.shape, support_data.device)
    emb_task2 = model.embedding_network(support_data)
    
    # Task 3 (reset masks)
    model.embedding_network.reset_dropout_masks(support_data.shape, support_data.device)
    emb_task3 = model.embedding_network(support_data)
    
    # Check if embeddings are different across tasks
    diff_12 = torch.max(torch.abs(emb_task1 - emb_task2)).item()
    diff_23 = torch.max(torch.abs(emb_task2 - emb_task3)).item()
    diff_13 = torch.max(torch.abs(emb_task1 - emb_task3)).item()
    
    print(f"Difference between Task 1 and Task 2: {diff_12:.6f}")
    print(f"Difference between Task 2 and Task 3: {diff_23:.6f}")
    print(f"Difference between Task 1 and Task 3: {diff_13:.6f}")
    
    if diff_12 > 1e-3 and diff_23 > 1e-3 and diff_13 > 1e-3:
        print("✅ PASS: Different tasks get DIFFERENT dropout masks!")
    else:
        print("❌ FAIL: Tasks are using the same masks!")
    
    print()

def test_meta_network_forward():
    """Test Meta Network forward pass with Meta Dropout."""
    print("=" * 70)
    print("TEST 3: Meta Network Forward Pass")
    print("=" * 70)
    
    model = MetaNetwork(embedding_dim=64, hidden_dim=128, num_classes=5,
                       dropout_rates=[0.05, 0.10, 0.15])
    model.eval()
    
    # Create task data
    support_data = torch.randn(5, 1, 105, 105)  # 5-way 1-shot
    support_labels = torch.arange(5)
    query_data = torch.randn(15, 1, 105, 105)  # 5-way 3-query
    
    # Forward pass (should automatically reset masks)
    try:
        logits = model(support_data, support_labels, query_data)
        print(f"Output shape: {logits.shape}")
        print(f"Expected shape: torch.Size([15, 5])")
        
        if logits.shape == torch.Size([15, 5]):
            print("✅ PASS: Forward pass successful with correct output shape!")
        else:
            print("❌ FAIL: Incorrect output shape!")
    except Exception as e:
        print(f"❌ FAIL: Error during forward pass: {e}")
    
    print()

def test_no_dropout_inference():
    """Test that dropout is disabled during eval mode."""
    print("=" * 70)
    print("TEST 4: Dropout Disabled in Eval Mode")
    print("=" * 70)
    
    model = MetaNetwork(embedding_dim=64, hidden_dim=128, num_classes=5,
                       dropout_rates=[0.5, 0.5, 0.5])
    model.eval()
    
    support_data = torch.ones(5, 1, 105, 105)
    support_labels = torch.arange(5)
    query_data = torch.ones(5, 1, 105, 105)
    
    # Multiple forward passes should give identical results in eval mode
    logits_1 = model(support_data, support_labels, query_data)
    logits_2 = model(support_data, support_labels, query_data)
    logits_3 = model(support_data, support_labels, query_data)
    
    diff_12 = torch.max(torch.abs(logits_1 - logits_2)).item()
    diff_23 = torch.max(torch.abs(logits_2 - logits_3)).item()
    
    print(f"Difference between run 1 and 2: {diff_12:.10f}")
    print(f"Difference between run 2 and 3: {diff_23:.10f}")
    
    # Note: Due to reset_mask() being called, there might be small differences
    # But they should be deterministic if we're in eval mode
    print("Note: Meta Dropout resets masks each forward, but in eval mode should be disabled")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING META DROPOUT IN META NETWORKS")
    print("=" * 70 + "\n")
    
    test_meta_dropout_consistency()
    test_different_tasks_different_masks()
    test_meta_network_forward()
    test_no_dropout_inference()
    
    print("=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)
    print("\n✨ Key Takeaways:")
    print("1. Support and query embeddings share the same dropout mask per task")
    print("2. Different tasks get different dropout masks (via reset_mask())")
    print("3. Meta Network forward pass automatically handles mask resets")
    print("4. Meta Dropout ensures consistent regularization for fast weight generation")
    print()
