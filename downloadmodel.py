import os
import subprocess
import shutil
from huggingface_hub import snapshot_download

def download_file(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, os.path.basename(url))
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        subprocess.run(["wget", "-q", "-O", dest_path, url], check=True)
    else:
        print(f"File {dest_path} already exists, skipping download.")

# Step 1: Download ControlNet models
controlnet_folder = "ComfyUI/models/controlnet"
controlnet_urls = [
    "https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/aes_stage2_control_net/aes_stage2_control.safetensors",
    "https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/sim_stage1_control_net/sim_stage1_control_net.safetensors"
]

for url in controlnet_urls:
    download_file(url, controlnet_folder)

# Step 2: Download Image Projection models
infiniteyou_folder = "InfiniteYou"
infiniteyou_urls = [
    "https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/aes_stage2_control_net/aes_stage2_img_proj.bin",
    "https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/sim_stage1_control_net/sim_stage1_img_proj.bin"
]

for url in infiniteyou_urls:
    download_file(url, infiniteyou_folder)


# Step 3: Download InsightFace model
os.makedirs("ComfyUI/models/insightface/", exist_ok=True)

snapshot_download(
    repo_id="vuongminhkhoi4/antelopev2",
    cache_dir="models",
    repo_type="model",
    local_dir="ComfyUI/models/insightface",
)

if os.path.exists("ComfyUI/models/insightface") and os.path.isdir("ComfyUI/models/insightface"):
    shutil.rmtree("ComfyUI/models/insightface")

print("All files downloaded successfully.")