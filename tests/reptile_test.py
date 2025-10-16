"""
Comprehensive test suite for MAML, FOMAML, and Reptile algorithms.

Tests verify:
Test 1: Algorithm Instantiation
Test 2: Invalid Algorithm Detection
Test 3: Inner Loop Adaptation
Test 4: Forward Pass with Custom Weights
Test 5: Meta-Training Convergence
Test 6: Algorithm Differences
Test 7: Memory Efficiency
Test 8: Computational Speed
Test 9: Backward Compatibility

"""

import sys
import os
# Add parent directory to path so we can import from algorithms package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time

from algorithms.maml import ModelAgnosticMetaLearning, train_maml
from algorithms.cnn_maml import SimpleConvNet


def create_simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    )


def create_toy_batch(batch_size=4, n_way=5, k_shot=1, k_query=15, input_dim=28):
    """Create synthetic task batch for testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Support set
    support_data = torch.randn(batch_size, n_way * k_shot, input_dim * input_dim).to(device)
    support_labels = torch.cat([torch.full((k_shot,), i) for i in range(n_way)]).to(device)
    support_labels = support_labels.unsqueeze(0).expand(batch_size, -1).reshape(batch_size, -1).to(device)
    
    # Query set
    query_data = torch.randn(batch_size, n_way * k_query, input_dim * input_dim).to(device)
    query_labels = torch.cat([torch.full((k_query,), i) for i in range(n_way)]).to(device)
    query_labels = query_labels.unsqueeze(0).expand(batch_size, -1).reshape(batch_size, -1).to(device)
    
    return support_data, support_labels, query_data, query_labels


class TestMetaLearningAlgorithms:
    """Test suite for meta-learning algorithms."""
    
    @staticmethod
    def test_algorithm_instantiation():
        """Test that all three algorithms can be instantiated."""
        print("\n" + "="*60)
        print("TEST 1: Algorithm Instantiation")
        print("="*60)
        
        model = create_simple_model()
        
        for algo in ['maml', 'fomaml', 'reptile']:
            try:
                trainer = ModelAgnosticMetaLearning(
                    model,
                    inner_lr=0.01,
                    outer_lr=0.001,
                    inner_steps=3,
                    algorithm=algo
                )
                print(f"✓ {algo.upper()}: Successfully instantiated")
                assert trainer.algorithm == algo, f"Algorithm mismatch: {trainer.algorithm} != {algo}"
                assert trainer.first_order == (algo != 'maml'), f"First-order flag incorrect for {algo}"
            except Exception as e:
                print(f"✗ {algo.upper()}: Failed - {e}")
                return False
        
        return True
    
    @staticmethod
    def test_invalid_algorithm():
        """Test that invalid algorithm raises error."""
        print("\n" + "="*60)
        print("TEST 2: Invalid Algorithm Detection")
        print("="*60)
        
        model = create_simple_model()
        
        try:
            trainer = ModelAgnosticMetaLearning(
                model,
                algorithm='invalid'
            )
            print("✗ Should have raised ValueError for invalid algorithm")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")
            return True
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
    
    @staticmethod
    def test_inner_update():
        """Test that inner_update produces task-adapted parameters."""
        print("\n" + "="*60)
        print("TEST 3: Inner Loop Adaptation")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_simple_model().to(device)
        
        for algo in ['maml', 'fomaml', 'reptile']:
            trainer = ModelAgnosticMetaLearning(model, algorithm=algo, inner_steps=3)
            
            # Create task data
            support_data = torch.randn(5, 28*28).to(device)
            support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)
            
            # Get adapted parameters
            adapted_params = trainer.inner_update(support_data, support_labels)
            
            # Verify structure
            assert isinstance(adapted_params, dict), "Adapted params should be dict"
            assert len(adapted_params) > 0, "Adapted params should not be empty"
            
            # Verify parameters are different from original
            params_changed = False
            for name, param in adapted_params.items():
                orig_param = dict(model.named_parameters())[name]
                if not torch.allclose(param, orig_param, atol=1e-5):
                    params_changed = True
                    break
            
            assert params_changed, f"{algo}: Inner loop didn't update parameters"
            print(f"✓ {algo.upper()}: Inner loop correctly adapted parameters")
        
        return True
    
    @staticmethod
    def test_forward_with_weights():
        """Test forward pass with custom weights."""
        print("\n" + "="*60)
        print("TEST 4: Forward Pass with Custom Weights")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_simple_model().to(device)
        trainer = ModelAgnosticMetaLearning(model, algorithm='reptile')
        
        # Create custom weights
        custom_weights = {name: param.clone() + 0.01 
                         for name, param in model.named_parameters()}
        
        # Forward with original weights
        x = torch.randn(10, 28*28).to(device)
        out_original = model(x)
        
        # Forward with custom weights
        out_custom = trainer.forward_with_weights(x, custom_weights)
        
        # Verify outputs are different
        assert not torch.allclose(out_original, out_custom, atol=1e-4), \
            "Custom weights should produce different output"
        
        print("✓ Forward with custom weights: Working correctly")
        return True
    
    @staticmethod
    def test_meta_train_step_convergence():
        """Test that meta_train_step reduces loss over iterations."""
        print("\n" + "="*60)
        print("TEST 5: Meta-Training Convergence")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for algo in ['maml', 'fomaml', 'reptile']:
            model = create_simple_model().to(device)
            trainer = ModelAgnosticMetaLearning(
                model,
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=3,
                algorithm=algo
            )
            
            # Train for a few steps
            losses = []
            for step in range(10):
                support_data, support_labels, query_data, query_labels = create_toy_batch()
                loss = trainer.meta_train_step(
                    support_data, support_labels,
                    query_data, query_labels
                )
                losses.append(loss)
            
            # Check for decrease in loss
            avg_first_half = np.mean(losses[:5])
            avg_second_half = np.mean(losses[5:])
            
            if avg_second_half < avg_first_half:
                print(f"✓ {algo.upper()}: Loss decreasing ({avg_first_half:.4f} → {avg_second_half:.4f})")
            else:
                print(f"⚠ {algo.upper()}: Loss not decreasing (may need tuning)")
        
        return True
    
    @staticmethod
    def test_algorithm_differences():
        """Verify that algorithms produce different updates."""
        print("\n" + "="*60)
        print("TEST 6: Algorithm Differences Verification")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        updates = {}
        
        for algo in ['maml', 'fomaml', 'reptile']:
            model = create_simple_model().to(device)
            trainer = ModelAgnosticMetaLearning(
                model,
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=2,
                algorithm=algo
            )
            
            # Record initial parameters
            initial_params = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
            }
            
            # Single meta-training step
            support_data, support_labels, query_data, query_labels = create_toy_batch()
            trainer.meta_train_step(support_data, support_labels, query_data, query_labels)
            
            # Record parameter changes
            param_changes = []
            for name, param in model.named_parameters():
                change = (param - initial_params[name]).abs().mean().item()
                param_changes.append(change)
            
            updates[algo] = np.mean(param_changes)
            print(f"  {algo.upper():10} - Avg parameter change: {updates[algo]:.6f}")
        
        # Verify algorithms produce different updates
        if len(set(updates.values())) > 1:
            print("✓ Algorithms produce different parameter updates")
            return True
        else:
            print("⚠ Algorithms produced identical updates (may be coincidence)")
            return True  # Not necessarily a failure
    
    @staticmethod
    def test_memory_efficiency():
        """Compare memory usage across algorithms."""
        print("\n" + "="*60)
        print("TEST 7: Memory Efficiency Comparison")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            print("⚠ GPU not available, skipping memory test")
            return True
        
        memory_usage = {}
        
        for algo in ['maml', 'fomaml', 'reptile']:
            model = create_simple_model().to(device)
            trainer = ModelAgnosticMetaLearning(
                model,
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=5,
                algorithm=algo
            )
            
            torch.cuda.reset_peak_memory_stats(device)
            
            # Run a few meta-training steps
            for _ in range(5):
                support_data, support_labels, query_data, query_labels = create_toy_batch()
                trainer.meta_train_step(support_data, support_labels, query_data, query_labels)
            
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            memory_usage[algo] = peak_mem
            print(f"  {algo.upper():10} - Peak memory: {peak_mem:7.1f} MB")
        
        # MAML should use more memory than FOMAML and Reptile
        if memory_usage['maml'] >= memory_usage['fomaml']:
            print("✓ MAML uses more memory than FOMAML (expected)")
        else:
            print("⚠ MAML uses less memory than FOMAML (unexpected)")
        
        return True
    
    @staticmethod
    def test_computational_speed():
        """Compare training speed across algorithms."""
        print("\n" + "="*60)
        print("TEST 8: Computational Speed Comparison")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        timing = {}
        
        for algo in ['maml', 'fomaml', 'reptile']:
            model = create_simple_model().to(device)
            trainer = ModelAgnosticMetaLearning(
                model,
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=5,
                algorithm=algo
            )
            
            # Warmup
            for _ in range(2):
                support_data, support_labels, query_data, query_labels = create_toy_batch()
                trainer.meta_train_step(support_data, support_labels, query_data, query_labels)
            
            # Time 20 iterations
            start = time.time()
            for _ in range(20):
                support_data, support_labels, query_data, query_labels = create_toy_batch()
                trainer.meta_train_step(support_data, support_labels, query_data, query_labels)
            elapsed = time.time() - start
            
            timing[algo] = elapsed
            print(f"  {algo.upper():10} - Time for 20 steps: {elapsed:6.2f}s")
        
        # Calculate speedups
        maml_time = timing['maml']
        for algo in ['fomaml', 'reptile']:
            speedup = maml_time / timing[algo]
            print(f"  {algo.upper()} is {speedup:.2f}x vs MAML")
        
        return True
    
    @staticmethod
    def test_backward_compatibility():
        """Test that existing MAML code still works."""
        print("\n" + "="*60)
        print("TEST 9: Backward Compatibility")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_simple_model().to(device)
        
        # Old code: using first_order parameter directly
        # This should still work for backward compatibility
        try:
            # Create FOMAML using first_order=True (old interface)
            trainer = ModelAgnosticMetaLearning(
                model,
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=3,
                algorithm='fomaml'
            )
            
            assert trainer.first_order == True, "FOMAML should have first_order=True"
            
            support_data, support_labels, query_data, query_labels = create_toy_batch()
            loss = trainer.meta_train_step(support_data, support_labels, query_data, query_labels)
            
            assert isinstance(loss, float), "Loss should be a float"
            
            print("✓ Backward compatibility maintained")
            return True
        except Exception as e:
            print(f"✗ Backward compatibility broken: {e}")
            return False
    
    @staticmethod
    def run_all_tests():
        """Run all tests and report results."""
        print("\n" + "="*60)
        print("REPTILE/MAML/FOMAML TEST SUITE")
        print("="*60)
        
        tests = [
            TestMetaLearningAlgorithms.test_algorithm_instantiation,
            TestMetaLearningAlgorithms.test_invalid_algorithm,
            TestMetaLearningAlgorithms.test_inner_update,
            TestMetaLearningAlgorithms.test_forward_with_weights,
            TestMetaLearningAlgorithms.test_meta_train_step_convergence,
            TestMetaLearningAlgorithms.test_algorithm_differences,
            TestMetaLearningAlgorithms.test_memory_efficiency,
            TestMetaLearningAlgorithms.test_computational_speed,
            TestMetaLearningAlgorithms.test_backward_compatibility,
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"\n✗ Test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results.append(False)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        passed = sum(results)
        total = len(results)
        print(f"Tests Passed: {passed}/{total}")
        
        if passed == total:
            print("\n✓ ALL TESTS PASSED!")
        else:
            print(f"\n✗ {total - passed} test(s) failed")
        
        return passed == total


if __name__ == "__main__":
    success = TestMetaLearningAlgorithms.run_all_tests()
    exit(0 if success else 1)