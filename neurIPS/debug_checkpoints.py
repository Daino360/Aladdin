# debug_checkpoint.py
import torch
import sys
import os

def analyze_checkpoint(weights_path):
    """Analyze the checkpoint to understand the architecture"""
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    print("=== CHECKPOINT ANALYSIS ===")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Look for pattern in parameter shapes
    param_shapes = {}
    for name, param in state_dict.items():
        if 'maskgit.transformer' in name and param.dim() > 0:
            first_dim = param.shape[0]
            if first_dim not in param_shapes:
                param_shapes[first_dim] = []
            param_shapes[first_dim].append(name)
    
    print("\nParameter dimensions found:")
    for dim, names in sorted(param_shapes.items()):
        print(f"Dimension {dim}: {len(names)} parameters")
        if dim in [517, 519, 512, 1024]:
            sample_names = names[:3] if len(names) > 3 else names
            print(f"  Sample params: {sample_names}")
    
    # Specifically look for action-related parameters
    print("\nAction-related parameters:")
    action_params = [name for name in state_dict.keys() if 'action' in name.lower()]
    for name in action_params:
        print(f"  {name}: {state_dict[name].shape}")
    
    # Analyze the first transformer layer to understand the dimension
    first_layer_params = [name for name in state_dict.keys() if 'layers.0.0' in name]
    print(f"\nFirst transformer layer parameters ({len(first_layer_params)} total):")
    for name in first_layer_params[:10]:  # Show first 10
        print(f"  {name}: {state_dict[name].shape}")

if __name__ == "__main__":
    weights_path = "checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"
    analyze_checkpoint(weights_path)