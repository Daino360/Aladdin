# check_setup.py
import os

def check_paths():
    paths = {
        "Config": "/home/sdainelli/aladdin/neurIPS/configs/config/guided_genie_config.yaml",
        "Model weights": "/home/sdainelli/aladdin/GenieRedux/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt",
        "Ground truth data": "/home/sdainelli/aladdin/neurIPS/data_generation/external/coinrun/ground_truth_data/ground_truth_20251023_173319.npz",
        "Tokenizer weights": "/home/sdainelli/aladdin/GenieRedux/checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0/model.pt"
    }
    
    print("🔍 Checking required files:")
    for desc, path in paths.items():
        exists = os.path.exists(path)
        print(f"   {'✅' if exists else '❌'} {desc}: {path}")
        
        if not exists and desc == "Model weights":
            # Suggest alternative locations
            print("   💡 Try finding your model weights in:")
            possible_locs = [
                "checkpoints/",
                "models/", 
                "outputs/",
                "experiments/",
                "../checkpoints/",
            ]
            for loc in possible_locs:
                if os.path.exists(loc):
                    print(f"      - {loc}")

check_paths()