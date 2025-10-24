# debug_tokenizer_issue.py
import torch
import yaml
import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from construct_model import construct_model

class ConfigObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

def debug_tokenizer():
    """Debug the tokenizer dimension issue"""
    config_path = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = ConfigObject(config_dict)
    config.model = "genie_redux_guided"
    config.mode = "eval"
    
    print("🔍 Debugging tokenizer dimensions...")
    
    # Build model
    model = construct_model(config)
    
    # Check tokenizer configuration
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
        print(f"Tokenizer config:")
        print(f"  image_size: {tokenizer.image_size}")
        print(f"  patch_size: {getattr(tokenizer, 'patch_size', 'NOT SET')}")
        print(f"  dim: {getattr(tokenizer, 'dim', 'NOT SET')}")
        
        # Check the first frame embedding layer
        if hasattr(tokenizer, 'to_patch_emb_first_frame'):
            first_frame_emb = tokenizer.to_patch_emb_first_frame
            print(f"First frame embedding: {first_frame_emb}")
            
            # Check if it's a Sequential and inspect its layers
            if isinstance(first_frame_emb, torch.nn.Sequential):
                print("First frame embedding layers:")
                for i, layer in enumerate(first_frame_emb):
                    print(f"  Layer {i}: {type(layer).__name__}")
                    if hasattr(layer, 'normalized_shape'):
                        print(f"    normalized_shape: {layer.normalized_shape}")
    
    # Test with different input shapes
    print("\nTesting different input shapes:")
    
    # Test 1: Single frame (should fail)
    test_shapes = [
        (1, 3, 64, 64),      # Single frame
        (1, 2, 3, 64, 64),   # 2-frame sequence
        (1, 16, 3, 64, 64),  # 16-frame sequence
    ]
    
    for shape in test_shapes:
        print(f"\nTesting shape: {shape}")
        try:
            if len(shape) == 4:
                # Single frame
                test_input = torch.randn(*shape)
            else:
                # Video sequence
                test_input = torch.randn(*shape)
            
            # Try to get tokenizer output
            with torch.no_grad():
                # Access tokenizer directly if possible
                if hasattr(model, 'tokenizer'):
                    output = model.tokenizer(test_input, return_only_codebook_ids=True)
                    print(f"  ✅ Tokenizer output shape: {output.shape}")
                else:
                    # Try full model
                    action = torch.LongTensor([1])
                    output = model(test_input, action)
                    print(f"  ✅ Full model output shape: {output.shape}")
                    
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    debug_tokenizer()
