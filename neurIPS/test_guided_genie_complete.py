# test_guided_genie_complete.py
import torch
import numpy as np
#import matplotlib.pyplot as plt
import os
from pathlib import Path
from load_guided_genie_hydra import load_guided_genie_redux, test_genie_predictions

def load_ground_truth_data(data_path):
    """Load your collected frames and actions"""
    data = np.load(data_path, allow_pickle=True)
    ground_truth_data = data['ground_truth_data']
    print(f"Loaded {len(ground_truth_data)} episodes")
    return ground_truth_data

def visualize_comparison(real_frames, predicted_frames, episode_idx=0, save_path=None):
    """Create side-by-side comparison"""
    num_comparisons = min(8, len(predicted_frames))
    
    #fig, axes = plt.subplots(3, num_comparisons, figsize=(20, 9))
    if num_comparisons == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_comparisons):
        # Real frame at time t (input)
        axes[0, i].imshow(real_frames[i])
        axes[0, i].set_title(f"Input Frame {i}")
        axes[0, i].axis('off')
        
        # Real frame at time t+1 (ground truth)
        axes[1, i].imshow(real_frames[i + 1])
        axes[1, i].set_title(f"Real Frame {i + 1}")
        axes[1, i].axis('off')
        
        # Predicted frame at time t+1
        axes[2, i].imshow(predicted_frames[i])
        axes[2, i].set_title(f"Predicted Frame {i + 1}")
        axes[2, i].axis('off')
    
    #plt.suptitle(f"Guided GenieRedux - Episode {episode_idx + 1}\n(Input → Real Next → Predicted Next)", fontsize=16)
    #plt.tight_layout()
    
    if save_path:
        #plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    #plt.show()

def calculate_metrics(real_frames, predicted_frames):
    """Calculate prediction quality metrics"""
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        has_skimage = True
    except ImportError:
        print("⚠️  scikit-image not available, using basic MSE only")
        has_skimage = False
    
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    
    for i, (pred, real) in enumerate(zip(predicted_frames, real_frames[1:1+len(predicted_frames)])):
        # MSE
        mse = np.mean((pred.astype(float) - real.astype(float)) ** 2)
        mse_scores.append(mse)
        
        if has_skimage:
            # PSNR
            psnr_val = psnr(real, pred, data_range=255)
            psnr_scores.append(psnr_val)
            
            # SSIM (convert to grayscale if needed)
            if len(real.shape) == 3:
                real_gray = np.mean(real, axis=2)
                pred_gray = np.mean(pred, axis=2)
            else:
                real_gray = real
                pred_gray = pred
            ssim_val = ssim(real_gray, pred_gray, data_range=255)
            ssim_scores.append(ssim_val)
        
        if i < 3:  # Print first few metrics
            if has_skimage:
                print(f"Frame {i+1}: MSE={mse:.4f}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")
            else:
                print(f"Frame {i+1}: MSE={mse:.4f}")
    
    results = {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
    }
    
    if has_skimage:
        results.update({
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
        })
    
    return results

def main():
    # PATHS - UPDATE THESE!
    config_path = "/home/sdainelli/Aladdin/neurIPS/configs/config/guided_genie_config.yaml"
    # config_path = "/home/sdainelli/Aladdin/GenieRedux/configs/config/guided_genie_config.yaml"
    weights_path = "/home/sdainelli/Aladdin/neurIPS/checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"  # Update this
    data_path = "/home/sdainelli/Aladdin/neurIPS/data_generation/external/coinrun/ground_truth_data/ground_truth_20251024_014408.npz"  # Update this
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_guided_genie_redux(config_path, weights_path, device)
    
    # Load your collected data
    ground_truth_data = load_ground_truth_data(data_path)
    
    # Test on first few episodes
    for episode_idx in range(min(2, len(ground_truth_data))):
        print(f"\n{'='*60}")
        print(f"TESTING EPISODE {episode_idx + 1}")
        print(f"{'='*60}")
        
        episode = ground_truth_data[episode_idx]
        frames = episode['frames']
        actions = episode['actions']
        
        print(f"Episode stats:")
        print(f"  - Frames: {len(frames)}")
        print(f"  - Actions: {len(actions)}")
        print(f"  - Frame shape: {frames[0].shape}")
        print(f"  - Action range: {np.min(actions)} to {np.max(actions)}")
        print(f"  - Unique actions: {np.unique(actions)}")
        
        # Test both open-loop and closed-loop predictions
        for loop_type in ['open_loop', 'closed_loop']:
            print(f"\n--- {loop_type.upper()} PREDICTIONS ---")
            
            use_open_loop = (loop_type == 'open_loop')
            num_steps = min(15, len(actions))  # Test first 15 steps
            
            predictions = test_genie_predictions(
                model, frames, actions, device, 
                num_steps=num_steps, 
                use_open_loop=use_open_loop
            )
            
            if len(predictions) > 0:
                # Calculate metrics
                metrics = calculate_metrics(frames, predictions)
                
                print(f"\n{loop_type.upper()} Metrics:")
                print(f"  MSE:    {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
                if 'psnr_mean' in metrics:
                    print(f"  PSNR:   {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
                    print(f"  SSIM:   {metrics['ssim_mean']:.3f} ± {metrics['ssim_std']:.3f}")
                
                # Visualize
                visualize_comparison(
                    frames, predictions, episode_idx,
                    save_path=f"genie_{loop_type}_episode_{episode_idx+1}.png"
                )
            else:
                print("❌ No predictions generated")

if __name__ == "__main__":
    main()