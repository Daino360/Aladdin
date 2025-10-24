# simple_genie_test.py
import torch
import numpy as np
import os
import sys
import yaml

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

def load_config(config_path):
    with open(config_path, 'r') as f:
        return ConfigObject(yaml.safe_load(f))

def test_single_prediction():
    """Test a single prediction with detailed debugging"""
    print("🧪 Testing single prediction...")
    
    config_path = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
    weights_path = "/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    config = load_config(config_path)
    config.model = "genie_redux_guided"
    config.mode = "eval"
    
    print("Building model...")
    model = construct_model(config)
    
    print("Loading weights...")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded")
    
    # Create test data
    print("\nCreating test data...")
    
    # Test with a simple frame (all zeros)
    test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    test_action = 1
    
    print(f"Test frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
    print(f"Test action: {test_action}")
    
    # Preprocess
    frame_tensor = torch.FloatTensor(test_frame).permute(2, 0, 1) / 255.0
    frame_tensor = frame_tensor.unsqueeze(0).to(device)  # [1, 3, 64, 64]
    action_tensor = torch.LongTensor([test_action]).to(device)  # [1]
    
    print(f"Frame tensor shape: {frame_tensor.shape}")
    print(f"Action tensor shape: {action_tensor.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            output = model(frame_tensor, action_tensor)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Convert back
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (np.clip(output_np, 0, 1) * 255).astype(np.uint8)
            print(f"Converted output shape: {output_np.shape}, dtype: {output_np.dtype}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_with_real_frame():
    """Test with a real frame from your data"""
    print("\n🧪 Testing with real frame...")
    
    # Load one frame from your data
    data_path = "/home/sdainelli/Aladdin/neurIPS/data_generation/external/coinrun/ground_truth_data/ground_truth_20251024_014408.npz"  # UPDATE THIS
    data = np.load(data_path, allow_pickle=True)
    ground_truth_data = data['ground_truth_data']
    
    if len(ground_truth_data) > 0:
        first_episode = ground_truth_data[0]
        real_frame = first_episode['frames'][0]
        real_action = first_episode['actions'][0]
        
        print(f"Real frame shape: {real_frame.shape}, dtype: {real_frame.dtype}")
        print(f"Real action: {real_action}")
        
        # Load model (same as above)
        config_path = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
        weights_path = "/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        config = load_config(config_path)
        config.model = "genie_redux_guided"
        config.mode = "eval"
        
        model = construct_model(config)
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        
        # Preprocess real frame
        frame_tensor = torch.FloatTensor(real_frame).permute(2, 0, 1) / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        action_tensor = torch.LongTensor([real_action]).to(device)
        
        print(f"Processed frame shape: {frame_tensor.shape}")
        print(f"Processed action shape: {action_tensor.shape}")
        
        try:
            with torch.no_grad():
                output = model(frame_tensor, action_tensor)
                print(f"✅ Real frame forward pass successful!")
                print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Real frame forward pass failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_single_prediction()
    test_with_real_frame()