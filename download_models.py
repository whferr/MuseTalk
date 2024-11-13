import os
import time
from huggingface_hub import snapshot_download
import requests
import gdown

def download_models():
    ProjectDir = os.path.abspath(os.path.dirname(__file__))
    CheckpointsDir = os.path.join(ProjectDir, "models")
    
    if not os.path.exists(CheckpointsDir):
        os.makedirs(CheckpointsDir)
        print("Downloading checkpoints...")
        
        # Download MuseTalk models
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            local_dir_use_symlinks=False
        )
        
        # Download VAE
        os.makedirs(f"{CheckpointsDir}/sd-vae-ft-mse/", exist_ok=True)
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=f"{CheckpointsDir}/sd-vae-ft-mse",
            local_dir_use_symlinks=False
        )
        
        # Download DWPose
        os.makedirs(f"{CheckpointsDir}/dwpose/", exist_ok=True)
        snapshot_download(
            repo_id="yzd-v/DWPose",
            local_dir=f"{CheckpointsDir}/dwpose",
            local_dir_use_symlinks=False
        )
        
        # Download Whisper
        os.makedirs(f"{CheckpointsDir}/whisper/", exist_ok=True)
        whisper_url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        response = requests.get(whisper_url)
        if response.status_code == 200:
            with open(f"{CheckpointsDir}/whisper/tiny.pt", "wb") as f:
                f.write(response.content)
                
        # Download Face Parse
        os.makedirs(f"{CheckpointsDir}/face-parse-bisent/", exist_ok=True)
        gdown.download(
            "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812",
            f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
        )
        
        # Download ResNet
        resnet_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(resnet_url)
        if response.status_code == 200:
            with open(f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth", "wb") as f:
                f.write(response.content)

if __name__ == "__main__":
    download_models()
