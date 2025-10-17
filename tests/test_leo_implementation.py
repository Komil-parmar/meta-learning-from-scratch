"""
Quick test script to verify LEO implementation.

This script performs basic sanity checks on the LEO implementation
without requiring a full training run.
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
from algorithms.leo import (
    LEOEncoder, 
    LEODecoder, 
    LEORelationNetwork, 
    LEOClassifier,
    LatentEmbeddingOptimization
)


def test_encoder():
    """Test LEO encoder with CNN features."""
    print("Testing LEOEncoder...")
    encoder = LEOEncoder(latent_dim=64, feature_dim=2304)
    
    # Test forward pass with CNN features (not raw images)
    batch_size = 4
    features = torch.randn(batch_size, 2304)  # CNN feature dimension
    z = encoder(features)
    
    assert z.shape == (batch_size, 64), f"Expected shape ({batch_size}, 64), got {z.shape}"
    print(f"  ✓ Forward pass: {features.shape} → {z.shape}")
    
    # Test gradients
    loss = z.mean()
    loss.backward()
    assert encoder.fc.weight.grad is not None, "Gradients not computed"
    print(f"  ✓ Gradients computed")
    
    print("  ✓ LEOEncoder test passed!\n")


def test_decoder():
    """Test LEO decoder with prototype generation."""
    print("Testing LEODecoder...")
    decoder = LEODecoder(latent_dim=64, feature_dim=2304)
    
    # Test forward pass with N class latent codes
    num_classes = 5
    z = torch.randn(num_classes, 64)  # [N, latent_dim]
    prototypes = decoder(z)
    
    # Check prototype shape - should be [N, feature_dim]
    expected_shape = torch.Size([num_classes, 2304])
    assert prototypes.shape == expected_shape, f"Expected {expected_shape}, got {prototypes.shape}"
    print(f"  ✓ Generated prototypes: {list(prototypes.shape)}")
    print(f"    - One prototype vector per class [N, feature_dim]")
    
    # Test single latent code
    z_single = torch.randn(64)  # [latent_dim]
    prototype_single = decoder(z_single)
    assert prototype_single.shape == torch.Size([2304]), f"Expected [2304], got {prototype_single.shape}"
    print(f"  ✓ Single latent code → single prototype: {list(prototype_single.shape)}")
    
    # Test gradients
    loss = prototypes.sum()
    loss.backward()
    assert decoder.fc3.weight.grad is not None, "Gradients not computed"
    print(f"  ✓ Gradients computed")
    
    print("  ✓ LEODecoder test passed!\n")


def test_relation_network():
    """Test LEO relation network."""
    print("Testing LEORelationNetwork...")
    relation_net = LEORelationNetwork(latent_dim=64)
    
    # Test forward pass
    batch_size = 4
    z1 = torch.randn(batch_size, 64)
    z2 = torch.randn(batch_size, 64)
    out = relation_net(z1, z2)
    
    assert out.shape == (batch_size, 64), f"Expected shape ({batch_size}, 64), got {out.shape}"
    print(f"  ✓ Forward pass: ({z1.shape}, {z2.shape}) → {out.shape}")
    
    # Test gradients
    loss = out.mean()
    loss.backward()
    assert relation_net.fc1.weight.grad is not None, "Gradients not computed"
    print(f"  ✓ Gradients computed")
    
    print("  ✓ LEORelationNetwork test passed!\n")


def test_classifier():
    """Test LEO classifier."""
    print("Testing LEOClassifier...")
    classifier = LEOClassifier(num_classes=5)
    
    # Test feature extraction
    batch_size = 4
    x = torch.randn(batch_size, 1, 105, 105)
    features = classifier.extract_features(x)
    
    assert features.shape == (batch_size, 2304), f"Expected shape ({batch_size}, 2304), got {features.shape}"
    print(f"  ✓ Feature extraction: {x.shape} → {features.shape}")
    
    # Test default forward pass
    logits = classifier(x)
    
    assert logits.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {logits.shape}"
    print(f"  ✓ Default forward pass: {x.shape} → {logits.shape}")
    
    # Test forward pass with custom prototypes
    decoder = LEODecoder(latent_dim=64, feature_dim=2304)
    z = torch.randn(5, 64)  # [N, latent_dim] for N classes
    prototypes = decoder(z)
    
    assert prototypes.shape == (5, 2304), f"Expected prototypes shape (5, 2304), got {prototypes.shape}"
    print(f"  ✓ Generated prototypes: {list(prototypes.shape)} [N, feature_dim]")
    
    logits_custom = classifier(x, prototypes)
    assert logits_custom.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {logits_custom.shape}"
    print(f"  ✓ Custom prototype forward pass: {x.shape} → {logits_custom.shape}")
    
    # Test gradients
    loss = logits_custom.mean()
    loss.backward()
    assert decoder.fc3.weight.grad is not None, "Decoder gradients computed"
    assert classifier.conv1.weight.grad is not None, "Classifier CNN gradients computed"
    print(f"  ✓ Gradients computed for both decoder and classifier CNN")
    
    print("  ✓ LEOClassifier test passed!\n")


def test_leo_full():
    """Test full LEO algorithm."""
    print("Testing LatentEmbeddingOptimization...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    leo = LatentEmbeddingOptimization(
        num_classes=5,
        latent_dim=64,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=3
    )
    leo = leo.to(device)
    
    # Test encode_task
    print("\n  Testing encode_task...")
    support_data = torch.randn(5, 1, 105, 105).to(device)  # 5-way 1-shot
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)
    
    latent_codes = leo.encode_task(support_data, support_labels)
    assert latent_codes.shape == (5, 64), f"Expected shape (5, 64), got {latent_codes.shape}"
    print(f"    ✓ Encoded support set: {support_data.shape} → {latent_codes.shape}")
    
    # Test decode_to_prototypes
    print("\n  Testing decode_to_prototypes...")
    prototypes = leo.decode_to_prototypes(latent_codes)
    assert prototypes.shape == torch.Size([5, 2304]), f"Expected (5, 2304), got {prototypes.shape}"
    print(f"    ✓ Decoded to prototypes: {list(prototypes.shape)} [N, feature_dim]")
    
    # Test inner_update
    print("\n  Testing inner_update...")
    adapted_codes = leo.inner_update(support_data, support_labels, latent_codes)
    assert adapted_codes.shape == (5, 64), f"Expected shape (5, 64), got {adapted_codes.shape}"
    print(f"    ✓ Adapted latent codes: {latent_codes.shape} → {adapted_codes.shape}")
    
    # Test meta_train_step
    print("\n  Testing meta_train_step...")
    batch_size = 2
    support_data_batch = torch.randn(batch_size, 5, 1, 105, 105).to(device)
    support_labels_batch = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]).to(device)
    query_data_batch = torch.randn(batch_size, 75, 1, 105, 105).to(device)  # 15 per class
    query_labels_batch = torch.randint(0, 5, (batch_size, 75)).to(device)
    
    loss = leo.meta_train_step(
        support_data_batch,
        support_labels_batch,
        query_data_batch,
        query_labels_batch
    )
    
    assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
    assert loss > 0, f"Expected positive loss, got {loss}"
    print(f"    ✓ Meta-training step loss: {loss:.4f}")
    
    # Test train/eval modes
    print("\n  Testing train/eval modes...")
    leo.train()
    assert leo.encoder.training, "Encoder not in training mode"
    
    leo.eval()
    assert not leo.encoder.training, "Encoder not in eval mode"
    print(f"    ✓ Train/eval modes working")
    
    print("\n  ✓ LatentEmbeddingOptimization test passed!\n")


def test_integration():
    """Test integration with typical workflow."""
    print("Testing Integration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create LEO
    leo = LatentEmbeddingOptimization(
        num_classes=5,
        latent_dim=64,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=3
    )
    leo = leo.to(device)
    
    # Simulate a few training steps
    print("  Running 5 training iterations...")
    leo.train()
    
    for i in range(5):
        # Generate fake task batch
        support_data = torch.randn(2, 5, 1, 105, 105).to(device)
        support_labels = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]).to(device)
        query_data = torch.randn(2, 75, 1, 105, 105).to(device)
        query_labels = torch.randint(0, 5, (2, 75)).to(device)
        
        loss = leo.meta_train_step(
            support_data,
            support_labels,
            query_data,
            query_labels
        )
        
        print(f"    Step {i+1}/5: loss = {loss:.4f}")
    
    # Test evaluation
    print("\n  Testing evaluation...")
    leo.eval()
    
    # Note: inner_update requires gradients, so we don't use no_grad here
    support_data = torch.randn(5, 1, 105, 105).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)
    query_data = torch.randn(75, 1, 105, 105).to(device)
    query_labels = torch.randint(0, 5, (75,)).to(device)
    
    # Encode and adapt (requires gradients for inner loop)
    initial_codes = leo.encode_task(support_data, support_labels)
    adapted_codes = leo.inner_update(support_data, support_labels, initial_codes)
    
    # Predict (can use no_grad for inference)
    with torch.no_grad():
        adapted_prototypes = leo.decode_to_prototypes(adapted_codes)
        logits = leo.classifier(query_data, adapted_prototypes)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        print(f"    Query accuracy: {accuracy.item()*100:.2f}%")
    
    print("\n  ✓ Integration test passed!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("LEO Implementation Tests")
    print("="*60)
    print()
    
    try:
        test_encoder()
        test_decoder()
        test_relation_network()
        test_classifier()
        test_leo_full()
        test_integration()
        
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        print()
        print("The LEO implementation is working correctly!")
        print("You can now run the full training with:")
        print("  python examples/leo_on_omniglot.py")
        print()
        
    except Exception as e:
        print("="*60)
        print("❌ Test failed!")
        print("="*60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
