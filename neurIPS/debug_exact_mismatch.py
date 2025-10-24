# debug_exact_mismatch.py
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def debug_exact_mismatch():
    """Debug the exact tokenizer mismatch"""
    from construct_model import construct_model
    import yaml
    
    class ConfigObject:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
                else:
                    setattr(self, key, value)
    
    config_path = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = ConfigObject(config_dict)
    config.model = "genie_redux_guided"
    config.mode = "eval"
    
    print("🔍 Debugging exact tokenizer mismatch...")
    
    # Build model without loading weights first
    model = construct_model(config)
    
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
        
        print("Tokenizer structure:")
        print(f"  image_size: {tokenizer.image_size}")
        print(f"  patch_size: {getattr(tokenizer, 'patch_size', 'NOT SET')}")
        
        # Check the first frame embedding in detail
        if hasattr(tokenizer, 'to_patch_emb_first_frame'):
            first_emb = tokenizer.to_patch_emb_first_frame
            print(f"  First frame embedding: {first_emb}")
            
            # Test what happens with our input
            test_input = torch.randn(1, 16, 3, 64, 64)  # Our current input format
            
            print(f"  Test input shape: {test_input.shape}")
            
            # Manually step through the first frame embedding
            for i, layer in enumerate(first_emb):
                print(f"    Layer {i}: {type(layer).__name__}")
                try:
                    test_input = layer(test_input)
                    print(f"      Output shape: {test_input.shape}")
                except Exception as e:
                    print(f"      ❌ Failed at layer {i}: {e}")
                    break
        
        # Check if there are other embedding layers
        other_embeddings = [attr for attr in dir(tokenizer) if 'emb' in attr.lower() and attr != 'to_patch_emb_first_frame']
        print(f"  Other embeddings: {other_embeddings}")
        
        for emb_name in other_embeddings:
            emb_layer = getattr(tokenizer, emb_name)
            print(f"  {emb_name}: {emb_layer}")

debug_exact_mismatch()