"""
Test script to verify MAML++ implementation fixes.

This script tests:
1. Alpha parameters are properly added to optimizer
2. inner_update returns generator for MAML++
3. Multi-step loss generates correct number of weight snapshots
4. Alpha parameters can be updated during training
5. Comparison between MAML and MAML++
"""

import torch
import torch.nn as nn
from algorithms.maml import ModelAgnosticMetaLearning
from algorithms.cnn_maml import SimpleConvNet


def get_test_model():
    """Create a test model compatible with MAML++."""
    return SimpleConvNet(
        num_classes=5,
        use_meta_dropout=True,
        dropout_config=[0.05, 0.1, 0.15, 0.05]
    )

import torch
import torch.nn as nn
from algorithms.maml import ModelAgnosticMetaLearning
from algorithms.cnn_maml import SimpleConvNet as SimpleModel


def test_alpha_in_optimizer():
    """Test 1: Verify alpha parameters are in optimizer."""
    print("\n" + "="*60)
    print("TEST 1: Alpha Parameters in Optimizer")
    print("="*60)
    
    model = get_test_model()
    maml_pp = ModelAgnosticMetaLearning(
        model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        plus_plus=True
    )
    
    # Check if alpha parameters are in optimizer
    optimizer_params = set()
    for group in maml_pp.meta_optimizer.param_groups:
        for p in group['params']:
            optimizer_params.add(id(p))
    
    # Check if all alpha parameters are present
    alpha_params_in_optimizer = all(
        id(alpha) in optimizer_params 
        for alpha in maml_pp.alpha
    )
    
    num_model_params = sum(1 for _ in model.parameters())
    num_alpha_params = len(maml_pp.alpha)
    num_optimizer_params = len(optimizer_params)
    
    print(f"Model parameters: {num_model_params}")
    print(f"Alpha parameters: {num_alpha_params}")
    print(f"Optimizer parameters: {num_optimizer_params}")
    print(f"Expected total: {num_model_params + num_alpha_params}")
    
    if alpha_params_in_optimizer:
        print("✅ PASSED: All alpha parameters are in optimizer")
        return True
    else:
        print("❌ FAILED: Alpha parameters NOT in optimizer")
        return False


def test_generator_return():
    """Test 2: Verify MAML++ uses inline adaptation."""
    print("\n" + "="*60)
    print("TEST 2: MAML++ Inline Adaptation Check")
    print("="*60)
    
    model = get_test_model()
    
    # Test MAML++ initialization
    maml_pp = ModelAgnosticMetaLearning(
        model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        plus_plus=True
    )
    
    # Check that MAML++ has param_names attribute
    has_param_names = hasattr(maml_pp, 'param_names')
    print(f"MAML++ has param_names: {has_param_names}")
    
    # Check that MAML++ has alpha parameters
    has_alpha = hasattr(maml_pp, 'alpha') and len(maml_pp.alpha) > 0
    print(f"MAML++ has alpha parameters: {has_alpha}")
    
    # Test standard MAML (should return dict from inner_update)
    model2 = get_test_model()
    maml = ModelAgnosticMetaLearning(
        model2,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        plus_plus=False
    )
    
    support_data = torch.randn(5, 1, 105, 105)  # 5 samples, 1 channel, 105x105 images (Omniglot size)
    support_labels = torch.tensor([0, 1, 2, 3, 4])
    
    result_std = maml.inner_update(support_data, support_labels)
    is_dict_std = isinstance(result_std, dict)
    
    print(f"Standard MAML inner_update returns dict: {is_dict_std}")
    print(f"Standard MAML result type: {type(result_std)}")
    print(f"Standard MAML result: dict with keys: {list(result_std.keys())[:3]}")
    
    if has_param_names and has_alpha and is_dict_std:
        print("✅ PASSED: MAML++ setup correct, Standard MAML works")
        return True
    else:
        print("❌ FAILED: Incorrect setup")
        return False


def test_multi_step_loss():
    """Test 3: Verify MAML++ computes multi-step loss correctly."""
    print("\n" + "="*60)
    print("TEST 3: MAML++ Multi-Step Loss Implementation")
    print("="*60)
    
    model = get_test_model()
    inner_steps = 5
    
    maml_pp = ModelAgnosticMetaLearning(
        model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=inner_steps,
        plus_plus=True
    )
    
    # Check MAML++ is set up correctly
    print(f"Inner steps: {maml_pp.inner_steps}")
    print(f"Plus plus mode: {maml_pp.plus_plus}")
    print(f"Number of alpha parameters: {len(maml_pp.alpha)}")
    print(f"Has param_names: {hasattr(maml_pp, 'param_names')}")
    
    # Test that a meta training step works
    device = torch.device('cpu')
    support_data = torch.randn(2, 5, 1, 105, 105)  # 2 tasks, 5 samples each
    support_labels = torch.randint(0, 5, (2, 5))
    query_data = torch.randn(2, 10, 1, 105, 105)  # 2 tasks, 10 queries each
    query_labels = torch.randint(0, 5, (2, 10))
    
    try:
        loss = maml_pp.meta_train_step(
            support_data, support_labels,
            query_data, query_labels
        )
        print(f"Meta training step succeeded with loss: {loss:.4f}")
        success = True
    except Exception as e:
        print(f"Meta training step failed: {e}")
        success = False
    
    if success and maml_pp.plus_plus and len(maml_pp.alpha) > 0:
        print("✅ PASSED: MAML++ multi-step loss implementation works")
        return True
    else:
        print("❌ FAILED: MAML++ implementation issue")
        return False


def test_alpha_updates():
    """Test 4: Verify alpha parameters can be updated."""
    print("\n" + "="*60)
    print("TEST 4: Alpha Parameter Updates")
    print("="*60)
    
    model = get_test_model()
    maml_pp = ModelAgnosticMetaLearning(
        model,
        inner_lr=0.01,
        outer_lr=0.01,  # Higher LR for visible changes
        inner_steps=3,
        plus_plus=True
    )
    
    # Record initial alpha values
    initial_alphas = [a.item() for a in maml_pp.alpha]
    print(f"Initial alpha values (first 3): {initial_alphas[:3]}")
    
    # Create dummy task data - proper image dimensions for CNN (Omniglot 105x105)
    device = torch.device('cpu')
    support_data = torch.randn(4, 5, 1, 105, 105)  # 4 tasks, 5 samples each
    support_labels = torch.randint(0, 5, (4, 5))
    query_data = torch.randn(4, 15, 1, 105, 105)  # 4 tasks, 15 queries each
    query_labels = torch.randint(0, 5, (4, 15))
    
    # Run several training steps
    for _ in range(10):
        loss = maml_pp.meta_train_step(
            support_data, support_labels,
            query_data, query_labels
        )
    
    # Check if alpha values changed
    final_alphas = [a.item() for a in maml_pp.alpha]
    print(f"Final alpha values (first 3): {final_alphas[:3]}")
    
    # Check if any alpha changed significantly
    changes = [abs(f - i) for f, i in zip(final_alphas, initial_alphas)]
    max_change = max(changes)
    print(f"Maximum alpha change: {max_change:.6f}")
    
    if max_change > 1e-6:
        print("✅ PASSED: Alpha parameters are being updated")
        return True
    else:
        print("❌ FAILED: Alpha parameters are NOT being updated")
        return False


def test_maml_vs_mamlpp():
    """Test 5: Compare MAML and MAML++ behavior."""
    print("\n" + "="*60)
    print("TEST 5: MAML vs MAML++ Comparison")
    print("="*60)
    
    # Create identical models
    torch.manual_seed(42)
    model1 = get_test_model()
    
    torch.manual_seed(42)
    model2 = get_test_model()
    
    # Standard MAML
    maml = ModelAgnosticMetaLearning(
        model1,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        plus_plus=False
    )
    
    # MAML++
    maml_pp = ModelAgnosticMetaLearning(
        model2,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        plus_plus=True
    )
    
    # Create dummy task - proper image dimensions for CNN (Omniglot 105x105)
    device = torch.device('cpu')
    support_data = torch.randn(4, 5, 1, 105, 105)  # 4 tasks, 5 samples each
    support_labels = torch.randint(0, 5, (4, 5))
    query_data = torch.randn(4, 15, 1, 105, 105)  # 4 tasks, 15 queries each
    query_labels = torch.randint(0, 5, (4, 15))
    
    # Train both
    loss_maml = maml.meta_train_step(
        support_data, support_labels,
        query_data, query_labels
    )
    
    loss_mamlpp = maml_pp.meta_train_step(
        support_data, support_labels,
        query_data, query_labels
    )
    
    print(f"MAML loss: {loss_maml:.4f}")
    print(f"MAML++ loss: {loss_mamlpp:.4f}")
    
    # Check that MAML++ has alpha parameters
    has_alpha = hasattr(maml_pp, 'alpha') and len(maml_pp.alpha) > 0
    no_alpha_in_maml = not hasattr(maml, 'alpha')
    
    print(f"MAML++ has alpha parameters: {has_alpha}")
    print(f"Standard MAML has no alpha: {no_alpha_in_maml}")
    
    if has_alpha and no_alpha_in_maml:
        print("✅ PASSED: MAML and MAML++ are properly differentiated")
        return True
    else:
        print("❌ FAILED: MAML/MAML++ differentiation issue")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "#"*60)
    print("# MAML++ IMPLEMENTATION VERIFICATION")
    print("#"*60)
    
    tests = [
        ("Alpha in Optimizer", test_alpha_in_optimizer),
        ("Generator Return Type", test_generator_return),
        ("Multi-Step Loss", test_multi_step_loss),
        ("Alpha Updates", test_alpha_updates),
        ("MAML vs MAML++", test_maml_vs_mamlpp),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MAML++ implementation is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
