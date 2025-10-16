import os
from huggingface_hub import hf_hub_download

def download_model(model_name: str, target_dpath: str = "checkpoints") -> str:
    # Replace with your repository and file details
    repo_id = "INSAIT-Institute/GenieRedux"  # Your Hugging Face repo ID
    filename = f"{model_name}.pt"
    save_dpath = f"{target_dpath}/{model_name}"  # File to download
    #make directories of filename
    os.makedirs(os.path.dirname(save_dpath), exist_ok=True)
    
    # Download the file
    downloaded_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=save_dpath)
    
    # Rename the downloaded file
    new_file_path = os.path.join(save_dpath, f"{model_name}_downloaded.pt")
    os.rename(f"{save_dpath}/{filename}", f"{save_dpath}/model.pt")
    downloaded_file_path = new_file_path

    print(f"File downloaded to: {downloaded_file_path}")
    return downloaded_file_path

download_model("GenieRedux_CoinRun_250mln_v1.0")
download_model("GenieRedux_Guided_CoinRun_80mln_v1.0")
download_model("GenieRedux_Tokenizer_CoinRun_100mln_v1.0")
