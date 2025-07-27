#!/usr/bin/env python3
"""
Quick test script to verify CUR4LLM is working before running full experiments.
"""

import sys
import torch
import torch.nn as nn

# Add current directory to path
sys.path.append(".")

def test_basic_cur():
    """Test basic CUR functionality"""
    print("🔍 Quick CUR4LLM Test")
    print("-" * 40)
    
    try:
        from modules.cur_linear import CURLinear
        print("✅ CURLinear import successful")
    except Exception as e:
        print(f"❌ CURLinear import failed: {e}")
        return False
    
    # Test 1: Create a simple CUR layer
    print("\n1️⃣ Testing CUR layer creation...")
    try:
        # Create components
        C = torch.randn(64, 16)  # out_features=64, k_cols=16
        U = torch.randn(16, 12)  # k_cols=16, k_rows=12  
        R = torch.randn(12, 128) # k_rows=12, in_features=128
        bias = torch.randn(64)
        
        # Create CUR layer
        cur_layer = CURLinear(C, U, R, bias=bias)
        
        # Test forward pass
        x = torch.randn(8, 128)  # batch_size=8, in_features=128
        output = cur_layer(x)
        
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output shape: (8, 64)")
        
        if output.shape == (8, 64):
            print("   ✅ CUR layer forward pass works")
        else:
            print("   ❌ Wrong output shape")
            return False
            
    except Exception as e:
        print(f"   ❌ CUR layer creation failed: {e}")
        return False
    
    # Test 2: Test from_linear conversion
    print("\n2️⃣ Testing from_linear conversion...")
    try:
        # Create original linear layer
        original = nn.Linear(128, 64, bias=True)
        
        # Add fake activation statistics
        original.scaling_diag_matrix = torch.ones(128)
        original.fisher_info = torch.ones(128)
        
        # Convert to CUR
        cur_layer = CURLinear.from_linear(
            original, 
            param_ratio=0.8,
            act_aware=True,
            alpha=0.5
        )
        
        if isinstance(cur_layer, CURLinear):
            ratio = cur_layer.get_param_ratio()
            print(f"   ✅ Conversion successful, param ratio: {ratio:.4f}")
            
            if ratio < 1.0:
                print("   ✅ Compression achieved")
            else:
                print("   ⚠️  No compression (ratio >= 1.0)")
        else:
            print("   ⚠️  Fallback to original layer")
            
    except Exception as e:
        print(f"   ❌ from_linear conversion failed: {e}")
        return False
    
    # Test 3: Test reconstruction
    print("\n3️⃣ Testing reconstruction...")
    try:
        # Test reconstruction accuracy
        x_test = torch.randn(5, 128)
        
        original_output = original(x_test)
        
        if isinstance(cur_layer, CURLinear):
            cur_output = cur_layer(x_test)
            
            error = torch.norm(original_output - cur_output) / torch.norm(original_output)
            print(f"   Reconstruction error: {error.item():.4f}")
            
            if error < 0.5:  # Reasonable threshold
                print("   ✅ Good reconstruction quality")
            else:
                print("   ⚠️  High reconstruction error")
        
    except Exception as e:
        print(f"   ❌ Reconstruction test failed: {e}")
        return False
    
    print("\n🎉 Quick test completed successfully!")
    print("✅ CUR4LLM basic functionality works")
    return True


def test_imports():
    """Test all required imports"""
    print("\n🔍 Testing imports...")
    
    imports_to_test = [
        ("modules.cur_linear", "CURLinear"),
        ("act_aware_utils", "calib_input_distribution"),
        ("sensitivity", "calib_sensitivity_ppl"),
        ("binary_search", "binary_search_truncation_rank"),
        ("datautils", "get_calib_data"),
        ("evaluate_utils", "evaluate_model"),
    ]
    
    all_good = True
    for module_name, item_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"   ✅ {module_name}.{item_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{item_name}: {e}")
            all_good = False
    
    return all_good


if __name__ == "__main__":
    print("🚀 CUR4LLM Quick Verification")
    print("=" * 50)
    
    # Test imports first
    imports_ok = test_imports()

    
    
    if not imports_ok:
        print("\n❌ Import issues detected. Fix imports first.")
        sys.exit(1)
    
    # Test basic functionality
    basic_ok = test_basic_cur()
    
    if basic_ok:
        print("\n✅ Ready to run full tests with: python tests/test_cur_correctness.py")
        print("✅ After full tests pass, try: python acur.py --model_id='facebook/opt-125m' --param_ratio_target=0.9 --n_calib_samples=8")
    else:
        print("\n❌ Basic functionality issues. Check the implementation.")
        sys.exit(1)