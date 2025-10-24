# check_tokenizer_config.py
import torch
import yaml

def check_tokenizer_config():
    """Check the tokenizer configuration from the YAML"""
    config_path = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    print("Tokenizer configuration from YAML:")
    if 'tokenizer' in config_dict:
        tokenizer_config = config_dict['tokenizer']
        for key, value in tokenizer_config.items():
            print(f"  {key}: {value}")
    else:
        print("  No tokenizer section found in config")
    
    # Check dynamics config too
    print("\nDynamics configuration:")
    if 'dynamics' in config_dict:
        dynamics_config = config_dict['dynamics']
        for key, value in dynamics_config.items():
            print(f"  {key}: {value}")

check_tokenizer_config()