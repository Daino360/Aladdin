# load_guided_genie_hydra.py
import torch
import numpy as np
import os
import sys
from pathlib import Path
import yaml

# Add the current directory to path to import your modules
sys.path.append(os.path.dirname(__file__))

# Alternative import
from models.construct_model import construct_model

class ConfigObject:
    """Simple config object that works like Hydra's DictConfig"""
    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
                elif isinstance(value, list):
                    # Handle lists properly
                    setattr(self, key, [ConfigObject(item) if isinstance(item, dict) else item for item in value])
                else:
                    setattr(self, key, value)
    
    def __getattr__(self, name):
        # Handle missing attributes gracefully
        return None
    
    def __contains__(self, key):
        return hasattr(self, key)

def load_hydra_config(config_path):
    """Load YAML config and handle Hydra defaults"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle Hydra defaults by merging base configs
    if 'defaults' in config_dict:
        base_config = {}
        for default in config_dict['defaults']:
            if default != '_self_':
                # Load base config file
                base_path = f"config/{default}.yaml"
                if os.path.exists(base_path):
                    with open(base_path, 'r') as f:
                        base_config.update(yaml.safe_load(f))
        
        # Merge base config with current config (current overrides base)
        merged_config = {**base_config, **config_dict}
        return ConfigObject(merged_config)
    else:
        return ConfigObject(config_dict)

def load_guided_genie_redux(config_path, weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the guided GenieRedux model with Hydra-style config
    """
    print("Loading Hydra config...")
    config = load_hydra_config(config_path)
    
    # Ensure we're using the right model type and mode
    config.model = "genie_redux_guided"
    config.mode = "eval"
    
    # Set critical paths
    if not hasattr(config, 'tokenizer_fpath') or config.tokenizer_fpath is None:
        # Set default tokenizer path
        config.tokenizer_fpath = "/home/sdainelli/aladdin/neurIPS/checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0/model.pt"
    
    print(f"Model type: {config.model}")
    print(f"Mode: {config.mode}")
    print(f"Tokenizer path: {config.tokenizer_fpath}")
    
    # Construct model
    print("Constructing Guided GenieRedux model...")
    model = construct_model(config)
    model = model.to(device)
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        # Try to find alternative locations
        possible_locations = [
            weights_path,
            f"checkpoints/{os.path.basename(weights_path)}",
            f"models/{os.path.basename(weights_path)}",
            config.eval.model_fpath if hasattr(config, 'eval') and hasattr(config.eval, 'model_fpath') else None,
        ]
        for loc in possible_locations:
            if loc and os.path.exists(loc):
                weights_path = loc
                print(f"✅ Found weights at: {weights_path}")
                break
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find model weights: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded from 'model_state_dict'")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded from 'model'")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded from 'state_dict'")
    else:
        # Try direct loading
        try:
            model.load_state_dict(checkpoint)
            print("Loaded directly from checkpoint")
        except:
            print("❌ Could not load weights - unknown format")
            raise
    
    model.eval()
    print("✅ Guided GenieRedux model loaded successfully!")
    return model, config

def preprocess_frame_for_genie(frame, target_size=64):
    """
    Preprocess frame for GenieRedux model
    - Resize to target_size
    - Convert to tensor
    - Normalize to [0, 1]
    """
    import torch
    from torchvision import transforms
    
    if isinstance(frame, np.ndarray):
        # Convert to tensor and normalize
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0
        
        frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1)  # [C, H, W]
    else:
        frame_tensor = frame
    
    # Resize if needed
    if frame_tensor.shape[1] != target_size or frame_tensor.shape[2] != target_size:
        frame_tensor = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0), 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    return frame_tensor

def test_genie_predictions(model, frames, actions, device, num_steps=None, use_open_loop=False):
    """
    Test the GenieRedux model predictions
    
    Args:
        model: Guided GenieRedux model
        frames: List of ground truth frames
        actions: List of actions
        device: torch device
        num_steps: Number of steps to predict
        use_open_loop: If True, use ground truth frames as input (open-loop)
                      If False, use model predictions (closed-loop)
    """
    model.eval()
    
    if num_steps is None:
        num_steps = min(len(frames) - 1, len(actions))
    
    predictions = []
    
    with torch.no_grad():
        # Start from first frame
        current_frame = preprocess_frame_for_genie(frames[0]).to(device).unsqueeze(0)  # [1, C, H, W]
        
        for t in range(num_steps):
            # Get action (ensure it's in valid range)
            action_val = int(actions[t])
            action = torch.LongTensor([action_val]).to(device)  # [1]
            
            print(f"Step {t+1}/{num_steps}: Action = {action_val}")
            
            try:
                # Guided GenieRedux forward pass
                predicted_frame = model(current_frame, action)
                
                # Convert prediction to numpy for storage
                predicted_np = predicted_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                predicted_np = (np.clip(predicted_np, 0, 1) * 255).astype(np.uint8)
                
                predictions.append(predicted_np)
                
                # Decide what to use as next input
                if use_open_loop and t + 1 < len(frames):
                    # Open-loop: use ground truth frame
                    current_frame = preprocess_frame_for_genie(frames[t + 1]).unsqueeze(0).to(device)
                else:
                    # Closed-loop: use model prediction
                    current_frame = preprocess_frame_for_genie(predicted_np).unsqueeze(0).to(device)
                    
            except Exception as e:
                print(f"❌ Error at step {t}: {e}")
                break
    
    return predictions