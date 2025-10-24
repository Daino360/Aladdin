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
        config.tokenizer_fpath = "/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0/model.pt"
    
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
    Fixed version with correct action tensor shape
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for t in range(min(num_steps, len(actions))):
            print(f"  Step {t+1}/{num_steps}: Action = {actions[t]}")
            
            try:
                # Create a single frame with proper format
                frame = frames[t]
                
                # Preprocess frame
                if isinstance(frame, np.ndarray):
                    if frame.dtype == np.uint8:
                        frame = frame.astype(np.float32) / 255.0
                    frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1)  # [C, H, W]
                else:
                    frame_tensor = frame
                
                # CORRECT FORMAT: [batch, channels, temporal, height, width]
                video_input = frame_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, 3, 1, 64, 64]
                
                # FIX: Action tensor needs proper shape for dynamics
                # The dynamics expects actions with shape that can be unsqueezed
                action = torch.LongTensor([[actions[t]]]).to(device)  # [1, 1] instead of [1]
                
                print(f"    Input shape: {video_input.shape}")
                print(f"    Action shape: {action.shape}")
                
                # Forward pass
                output = model(video_input, action)
                print(f"    Output shape: {output.shape}")
                
                # Extract prediction
                if output.dim() == 5:
                    if output.shape[1] == 3:  # channels first: [B, C, T, H, W]
                        predicted_frame = output[0, :, -1]  # Last frame
                    else:  # temporal first: [B, T, C, H, W] 
                        predicted_frame = output[0, -1]  # Last frame
                else:
                    predicted_frame = output[0]
                
                # Convert to numpy for visualization
                predicted_np = predicted_frame.permute(1, 2, 0).cpu().numpy()
                predicted_np = (np.clip(predicted_np, 0, 1) * 255).astype(np.uint8)
                
                predictions.append(predicted_np)
                print(f"    ✅ Prediction successful")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                
                # Try different action formats
                try:
                    print("    Trying different action format...")
                    frame = frames[t]
                    if isinstance(frame, np.ndarray):
                        if frame.dtype == np.uint8:
                            frame = frame.astype(np.float32) / 255.0
                        frame_tensor = torch.FloatTensor(frame).permute(2, 0, 1)
                    else:
                        frame_tensor = frame
                    
                    video_input = frame_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, 3, 1, 64, 64]
                    
                    # Try action with more dimensions
                    action = torch.LongTensor([actions[t]]).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 1]
                    
                    print(f"    Alternative action shape: {action.shape}")
                    output = model(video_input, action)
                    print(f"    Output shape: {output.shape}")
                    
                    if output.dim() == 5:
                        predicted_frame = output[0, :, -1] if output.shape[1] == 3 else output[0, -1]
                    else:
                        predicted_frame = output[0]
                    
                    predicted_np = predicted_frame.permute(1, 2, 0).cpu().numpy()
                    predicted_np = (np.clip(predicted_np, 0, 1) * 255).astype(np.uint8)
                    
                    predictions.append(predicted_np)
                    print(f"    ✅ Alternative action format worked!")
                    
                except Exception as e2:
                    print(f"    ❌ Alternative also failed: {e2}")
                    break
    
    return predictions