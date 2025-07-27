#!/usr/bin/env python3
"""
Debug what layers are being selected for compression.
The issue is likely that we're compressing critical layers that shouldn't be touched.
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add path
if '.' not in sys.path:
    sys.path.append('.')

from modules.cur_linear import CURLinear


def analyze_model_layers(model):
    """Analyze all linear layers in the model to see what should/shouldn't be compressed."""
    print(f"\n🔍 COMPLETE MODEL LAYER ANALYSIS")
    print("=" * 60)
    
    all_linear = []
    should_compress = []
    should_skip = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            
            # Current skip logic
            should_skip_current = any(skip in name for skip in ['embed', 'lm_head', 'classifier'])
            
            # Better skip logic (more comprehensive)
            should_skip_better = any(skip in name for skip in [
                'embed',           # Embedding layers
                'lm_head',         # Language model head
                'classifier',      # Classification head
                'layer_norm',      # Layer norm (these are tiny anyway)
                'layernorm',       # Alternative layer norm naming
                'norm',            # Any normalization
                'positional',      # Positional encodings
                'position'         # Position embeddings
            ])
            
            layer_info = {
                'name': name,
                'module': module,
                'parameters': params,
                'shape': f"{module.weight.shape[0]}x{module.weight.shape[1]}",
                'skip_current': should_skip_current,
                'skip_better': should_skip_better
            }
            
            all_linear.append(layer_info)
            
            if should_skip_better:
                should_skip.append(layer_info)
            else:
                should_compress.append(layer_info)
    
    print(f"📊 Layer Categories:")
    print(f"  Total Linear layers: {len(all_linear)}")
    print(f"  Should SKIP (critical): {len(should_skip)}")
    print(f"  Should COMPRESS (safe): {len(should_compress)}")
    
    print(f"\n❌ CRITICAL LAYERS (should NOT be compressed):")
    for layer in should_skip:
        print(f"  {layer['name']}: {layer['shape']} ({layer['parameters']:,} params)")
    
    print(f"\n✅ SAFE LAYERS (can be compressed):")
    for i, layer in enumerate(should_compress):
        print(f"  {layer['name']}: {layer['shape']} ({layer['parameters']:,} params)")
        if i >= 10:  # Show first 10
            print(f"  ... and {len(should_compress) - 10} more")
            break
    
    # Check what the current logic would do
    current_skip_count = sum(1 for layer in all_linear if layer['skip_current'])
    current_compress_count = len(all_linear) - current_skip_count
    
    better_skip_count = len(should_skip)
    better_compress_count = len(should_compress)
    
    print(f"\n📈 Comparison:")
    print(f"  Current logic: Skip {current_skip_count}, Compress {current_compress_count}")
    print(f"  Better logic:  Skip {better_skip_count}, Compress {better_compress_count}")
    
    if current_compress_count > better_compress_count:
        print(f"  ⚠️  Current logic compresses {current_compress_count - better_compress_count} more layers!")
        print(f"  🚨 This might be compressing critical layers!")
    
    return should_compress, should_skip


def test_conservative_compression():
    """Test compression with a very conservative approach - only compress attention layers."""
    print(f"\n🧪 CONSERVATIVE COMPRESSION TEST")
    print("=" * 50)
    
    # Load model
    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    
    # Analyze layers
    safe_layers, critical_layers = analyze_model_layers(model)
    
    # Be VERY conservative - only compress attention projection layers
    attention_layers = []
    for layer in safe_layers:
        if any(attn in layer['name'] for attn in ['k_proj', 'q_proj', 'v_proj', 'out_proj']):
            attention_layers.append(layer)
    
    print(f"\n🎯 ULTRA-CONSERVATIVE: Only attention layers ({len(attention_layers)}):")
    for layer in attention_layers[:5]:
        print(f"  {layer['name']}: {layer['shape']}")
    if len(attention_layers) > 5:
        print(f"  ... and {len(attention_layers) - 5} more")
    
    # Count original parameters manually
    original_total = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            original_total += params
    
    print(f"\n📊 BEFORE compression:")
    print(f"  Total parameters: {original_total:,}")
    
    # Apply compression ONLY to attention layers
    print(f"\n🔄 Applying CUR to attention layers only...")
    
    from datautils import get_calib_data
    from act_aware_utils import calib_input_distribution
    
    calib_loader = get_calib_data("wikitext2", tokenizer, model_id, 4, seed=42)
    calib_input_distribution(model, calib_loader, "abs_mean", use_cache=False)
    
    successful = 0
    target_ratio = 0.8
    
    for layer_info in attention_layers:
        try:
            name = layer_info['name']
            module = layer_info['module']
            
            cur_layer = CURLinear.from_linear(module, param_ratio=target_ratio, act_aware=True, alpha=0.5)
            
            if isinstance(cur_layer, CURLinear):
                # Replace layer
                parent = model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], cur_layer)
                successful += 1
                
        except Exception as e:
            print(f"  ❌ Failed {name}: {e}")
    
    print(f"✅ Successfully compressed {successful}/{len(attention_layers)} attention layers")
    
    # Count parameters after compression  
    compressed_total = 0
    cur_layers = 0
    linear_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if isinstance(module, CURLinear):
                params = (module.C.numel() + module.U.numel() + module.R.numel() + 
                         (module.bias.numel() if module.bias is not None else 0))
                cur_layers += 1
            else:
                params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                linear_layers += 1
            compressed_total += params
    
    print(f"\n📊 AFTER compression:")
    print(f"  Total parameters: {compressed_total:,}")
    print(f"  CUR layers: {cur_layers}")
    print(f"  Linear layers: {linear_layers}")
    print(f"  Compression ratio: {compressed_total / original_total:.4f}")
    print(f"  Parameter reduction: {(1 - compressed_total / original_total) * 100:.1f}%")
    
    # Test functionality
    print(f"\n🧪 Testing functionality:")
    try:
        test_input = tokenizer("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                test_input.input_ids.to(model.device), 
                max_new_tokens=8, 
                do_sample=False
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Generated: '{generated}'")
        
        # Check quality
        words = generated.split()[-6:]
        unique_words = len(set(words))
        
        if unique_words >= 3:
            print(f"  ✅ Good quality ({unique_words} unique words)")
            return True
        else:
            print(f"  ❌ Poor quality ({unique_words} unique words)")
            return False
            
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        return False


def main():
    """Main debug function."""
    print("🚨 LAYER SELECTION DEBUG")
    print("=" * 60)
    
    try:
        success = test_conservative_compression()
        
        if success:
            print(f"\n🎉 Conservative compression works!")
            print(f"💡 The issue was likely compressing critical layers")
            print(f"🔧 Solution: Be more selective about which layers to compress")
        else:
            print(f"\n❌ Even conservative compression fails")
            print(f"🔧 Deeper debugging needed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()