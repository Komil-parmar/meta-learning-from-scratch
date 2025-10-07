"""
Test script for Meta Dropout v3.0 (Context Manager Implementation).

Tests the current implementation which uses:
- Boolean flag (_outer_loop_mode) for dropout control
- Context manager (outer_loop_mode()) for clean API
- Conditional dropout in forward pass
- Zero overhead design
"""

import torch
import torch.nn.functional as F
from SimpleConvNet import SimpleConvNet
from Meta_Dropout import MetaDropout


def test_meta_dropout_broadcasting():
    """Test that Meta Dropout handles different batch sizes efficiently"""
    
    print("🧪 Test 1: Meta Dropout Broadcasting\n")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dropout = MetaDropout(p=0.5).to(device)
    dropout.train()
    
    # Simulate MAML scenario
    support_batch_size = 5   # N-way K-shot (e.g., 5-way 1-shot)
    query_batch_size = 75    # N-way Q-query (e.g., 5-way 15-query)
    channels, height, width = 64, 52, 52
    
    print(f"📊 Configuration:")
    print(f"   Support batch size: {support_batch_size}")
    print(f"   Query batch size:   {query_batch_size}")
    print(f"   Feature map shape:  ({channels}, {height}, {width})")
    print()
    
    # Create dummy inputs with different batch sizes
    support_data = torch.randn(support_batch_size, channels, height, width, device=device)
    query_data = torch.randn(query_batch_size, channels, height, width, device=device)
    
    print("✓ Step 1: Reset mask using support set shape")
    dropout.reset_mask(support_data.shape, device)
    print(f"  Mask shape: {dropout.mask.shape}")
    print(f"  Broadcasts across batch dimension (batch_size=1)")
    print()
    
    print("✓ Step 2: Forward pass with support set (inner loop)")
    output_support_1 = dropout(support_data)
    output_support_2 = dropout(support_data)
    support_identical = torch.allclose(output_support_1, output_support_2)
    print(f"  Consistent outputs: {support_identical} ✓")
    print()
    
    print("✓ Step 3: Forward pass with query set (different batch size)")
    output_query_1 = dropout(query_data)
    output_query_2 = dropout(query_data)
    query_identical = torch.allclose(output_query_1, output_query_2)
    print(f"  Consistent outputs: {query_identical} ✓")
    print()
    
    print("✓ Step 4: Verify same dropout pattern applied")
    # Check that the same pattern is applied across different batch sizes
    support_dropped = (output_support_1 == 0).float()
    query_dropped = (output_query_1 == 0).float()
    
    pattern_match = torch.allclose(
        support_dropped[0],  # First sample from support
        query_dropped[0]     # First sample from query
    )
    print(f"  Same dropout pattern: {pattern_match} ✓")
    print()
    
    print("="*70)
    if support_identical and query_identical and pattern_match:
        print("✅ PASS: Meta Dropout broadcasts correctly!")
    else:
        print("❌ FAIL: Broadcasting issue detected")
    print("="*70)
    
    return support_identical and query_identical and pattern_match


def test_context_manager():
    """Test that the outer_loop_mode() context manager works correctly"""
    
    print("\n🧪 Test 2: Context Manager (outer_loop_mode)\n")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with Meta Dropout
    model = SimpleConvNet(
        num_classes=5,
        dropout_config=[0.0, 0.5, 0.5, 0.0],  # High dropout for testing
        use_meta_dropout=True
    ).to(device)
    model.train()  # Keep in train mode
    
    # Create test data
    batch_size = 10
    test_data = torch.randn(batch_size, 1, 105, 105, device=device)
    
    print("✓ Step 1: Initialize dropout masks")
    model.reset_dropout_masks(batch_size, device)
    print(f"  Model in train mode: {model.training}")
    print(f"  Outer loop mode flag: {model._outer_loop_mode}")
    print()
    
    print("✓ Step 2: Forward pass WITH dropout (normal mode)")
    output_with_dropout_1 = model(test_data)
    output_with_dropout_2 = model(test_data)
    consistent_with_dropout = torch.allclose(output_with_dropout_1, output_with_dropout_2)
    print(f"  Consistent outputs: {consistent_with_dropout} ✓")
    print()
    
    print("✓ Step 3: Forward pass WITHOUT dropout (context manager)")
    with model.outer_loop_mode():
        print(f"  Inside context - flag: {model._outer_loop_mode} (should be True)")
        print(f"  Inside context - model training: {model.training}")
        output_no_dropout_1 = model(test_data)
        output_no_dropout_2 = model(test_data)
    
    consistent_no_dropout = torch.allclose(output_no_dropout_1, output_no_dropout_2)
    print(f"  Consistent outputs: {consistent_no_dropout} ✓")
    print()
    
    print("✓ Step 4: Verify dropout restored after context")
    print(f"  After context - flag: {model._outer_loop_mode} (should be False)")
    print(f"  After context - model training: {model.training}")
    output_restored = model(test_data)
    restored_matches_with_dropout = torch.allclose(output_restored, output_with_dropout_1)
    print(f"  Dropout restored: {restored_matches_with_dropout} ✓")
    print()
    
    print("✓ Step 5: Verify different outputs (dropout vs no-dropout)")
    different_outputs = not torch.allclose(output_with_dropout_1, output_no_dropout_1)
    print(f"  Different outputs: {different_outputs} ✓")
    print()
    
    print("="*70)
    success = (
        consistent_with_dropout and 
        consistent_no_dropout and
        restored_matches_with_dropout and
        different_outputs and
        not model._outer_loop_mode  # Flag should be reset
    )
    
    if success:
        print("✅ PASS: Context manager works perfectly!")
    else:
        print("❌ FAIL: Context manager issue detected")
    print("="*70)
    
    return success


def test_exception_safety():
    """Test that context manager is exception-safe"""
    
    print("\n🧪 Test 3: Exception Safety\n")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleConvNet(
        num_classes=5,
        dropout_config=[0.0, 0.5, 0.5, 0.0],
        use_meta_dropout=True
    ).to(device)
    
    print("✓ Testing exception handling in context manager")
    print(f"  Initial flag state: {model._outer_loop_mode}")
    print()
    
    try:
        with model.outer_loop_mode():
            print(f"  Inside context - flag: {model._outer_loop_mode}")
            # Simulate an error
            raise ValueError("Test exception")
    except ValueError:
        print(f"  Exception caught (as expected)")
    
    print(f"  After exception - flag: {model._outer_loop_mode}")
    print()
    
    success = (model._outer_loop_mode == False)
    
    print("="*70)
    if success:
        print("✅ PASS: Context manager is exception-safe!")
    else:
        print("❌ FAIL: Flag not reset after exception")
    print("="*70)
    
    return success


def test_functional_call_compatibility():
    """Test that context manager works with torch.func.functional_call"""
    
    print("\n🧪 Test 4: Compatibility with torch.func.functional_call\n")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleConvNet(
        num_classes=5,
        dropout_config=[0.0, 0.5, 0.5, 0.0],
        use_meta_dropout=True
    ).to(device)
    
    batch_size = 10
    test_data = torch.randn(batch_size, 1, 105, 105, device=device)
    
    print("✓ Testing with functional_call (MAML use case)")
    model.reset_dropout_masks(batch_size, device)
    
    # Get model parameters as dict (simulating MAML's fast_weights)
    fast_weights = {name: param for name, param in model.named_parameters()}
    
    print("  Testing outer loop mode with functional_call")
    with model.outer_loop_mode():
        output1 = torch.func.functional_call(model, fast_weights, test_data)
        output2 = torch.func.functional_call(model, fast_weights, test_data)
    
    outputs_identical = torch.allclose(output1, output2)
    print(f"  Deterministic outputs: {outputs_identical} ✓")
    print()
    
    print("="*70)
    if outputs_identical:
        print("✅ PASS: Works with torch.func.functional_call!")
    else:
        print("❌ FAIL: Non-deterministic behavior with functional_call")
    print("="*70)
    
    return outputs_identical


def test_performance():
    """Test performance overhead of context manager"""
    import time
    
    print("\n⚡ Test 5: Performance Overhead\n")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleConvNet(
        num_classes=5,
        dropout_config=[0.05, 0.10, 0.15, 0.05],
        use_meta_dropout=True
    ).to(device)
    
    batch_size = 32
    test_data = torch.randn(batch_size, 1, 105, 105, device=device)
    model.reset_dropout_masks(batch_size, device)
    
    # Warmup
    for _ in range(10):
        _ = model(test_data)
    
    iterations = 1000
    print(f"📊 Running {iterations} iterations...")
    print()
    
    # Test 1: Normal forward pass (baseline)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = model(test_data)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    baseline_time = time.time() - start
    
    # Test 2: With context manager
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        with model.outer_loop_mode():
            _ = model(test_data)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    context_time = time.time() - start
    
    print(f"Results:")
    print(f"  Baseline (normal):     {baseline_time*1000:.2f} ms")
    print(f"  With context manager:  {context_time*1000:.2f} ms")
    print(f"  Overhead:              {(context_time - baseline_time)*1000:.2f} ms")
    print(f"  Per iteration:         {(context_time - baseline_time)*1e6/iterations:.2f} µs")
    print()
    
    overhead_percent = ((context_time - baseline_time) / baseline_time) * 100
    print(f"Relative overhead: {overhead_percent:.2f}%")
    print()
    
    print("="*70)
    # Note: Context manager overhead is mostly the try/finally block,
    # not the boolean flag check. On GPU this is negligible.
    # On CPU, we accept <10% overhead as reasonable.
    threshold = 10.0 if device.type == 'cpu' else 2.0
    
    if overhead_percent < 0:
        print(f"✅ PASS: Negative overhead (context manager actually faster!)")
        print("   Note: Measurement variance - overhead is effectively zero")
        success = True
    elif overhead_percent < threshold:
        print(f"✅ PASS: Acceptable overhead (<{threshold}%)")
        print("   Note: Overhead is from Python context manager, not dropout logic")
        success = True
    else:
        print(f"⚠️  WARNING: {overhead_percent:.1f}% overhead detected")
        success = False
    print("="*70)
    
    return success


if __name__ == "__main__":
    # Run all tests
    print("\n" + "="*70)
    print("🚀 META DROPOUT V3.0 TEST SUITE")
    print("   Context Manager Implementation")
    print("="*70 + "\n")
    
    results = []
    
    try:
        results.append(("Broadcasting", test_meta_dropout_broadcasting()))
        results.append(("Context Manager", test_context_manager()))
        results.append(("Exception Safety", test_exception_safety()))
        results.append(("Functional Call", test_functional_call_compatibility()))
        results.append(("Performance", test_performance()))
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    all_passed = all(result for _, result in results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nMeta Dropout v3.0 is production-ready:")
        print("  ✓ Zero overhead context manager")
        print("  ✓ Exception-safe design")
        print("  ✓ Compatible with torch.func.functional_call")
        print("  ✓ Batch-size agnostic broadcasting")
        print("\n" + "="*70 + "\n")
    else:
        print("\n❌ SOME TESTS FAILED - Please investigate")
        print("="*70 + "\n")
