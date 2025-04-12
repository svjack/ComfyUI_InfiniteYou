# 🚀 ComfyUI_InfiniteYou
An implementation of **InfiniteYou** for **ComfyUI**

Native support for [InfiniteYou](https://github.com/bytedance/InfiniteYou) in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed by the ZenAI team.  


**✨ Support further development by starring the project! ✨**


![teaser](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/teaser.jpg)

---

## 🔥 News
- **[03/2025]** 🔥 Integrate Face Swap feature
- **[03/2025]** 🔥 Integrate Face Combine feature to predict future children
- **[03/2025]** 🔥 Code updated and released as the first version.

---

#### vim run_infiniteyou.py
```python
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    noise = RandomNoise(442)
    control_net = ControlNetLoader('sim_stage1_control_net.safetensors')
    model = UNETLoader('flux1-dev.safetensors', 'default')
    clip = DualCLIPLoader('t5xxl_fp16.safetensors', 'clip_l.safetensors', 'flux', 'default')
    clip_text_encode_positive_prompt_conditioning = CLIPTextEncode('sweet boy in office, front view', clip)
    clip_text_encode_positive_prompt_conditioning = FluxGuidance(clip_text_encode_positive_prompt_conditioning, 2.5)
    clip_text_encode_positive_prompt_conditioning2 = CLIPTextEncode('', clip)
    image, _ = LoadImage('xiang (2).jpg')
    latent = EmptySD3LatentImage(1024, 1024, 1)
    vae = VAELoader('ae.safetensors')
    model2, positive, _, latent = InfiniteYouApply(control_net, model, clip_text_encode_positive_prompt_conditioning, clip_text_encode_positive_prompt_conditioning2, image, latent, 'sim_stage1_img_proj.bin', 1.0, 0.000000000000000, 1, vae, 1)
    guider = BasicGuider(model2, positive)
    sampler = KSamplerSelect('euler')
    sigmas = BasicScheduler(model, 'simple', 30, 1)
    latent, _ = SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent)
    image2 = VAEDecode(latent, vae)
    SaveImage(image2, 'ComfyUI')
```

#### vim run_xiang_infiniteyou.py
```python
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    control_net = ControlNetLoader('aes_stage2_control_net/aes_stage2_control.safetensors')
    model = UNETLoader('flux1-dev.safetensors', 'default')
    clip = DualCLIPLoader('t5xxl_fp16.safetensors', 'clip_l.safetensors', 'flux', 'default')
    clip_text_encode_positive_prompt_conditioning = CLIPTextEncode('a sweet boy, 25 years old, clean the floor with a mop', clip)
    clip_text_encode_positive_prompt_conditioning = FluxGuidance(clip_text_encode_positive_prompt_conditioning, 3.5)
    clip_text_encode_positive_prompt_conditioning2 = CLIPTextEncode('', clip)
    image, _ = LoadImage('xiang (2).jpg')
    image2, _ = LoadImage('IMG_202305031758340.JPG')
    latent = EmptySD3LatentImage(1024, 1024, 1)
    vae = VAELoader('ae.safetensors')
    model, positive, negative, latent = FaceCombine(control_net, model, clip_text_encode_positive_prompt_conditioning, clip_text_encode_positive_prompt_conditioning2, image, image2, latent, 'aes_stage2_control_net/aes_stage2_img_proj.bin', 1.0000000000000002, 1, 0, 1, vae, 0.7000000000000002)
    latent = KSampler(model, 42, 30, 1, 'euler', 'simple', positive, negative, latent, 1)
    image3 = VAEDecode(latent, vae)
    SaveImage(image3, 'ComfyUI')
```

```python
import os
import time
import pandas as pd
import subprocess
from pathlib import Path
from itertools import product

# Configuration
SEEDS = [2, 4, 42, 442, 224]
IMAGE_PATHS = ['xiang (2).jpg', 'xiang_flipped.jpg']
OUTPUT_DIR = 'ComfyUI/output'
PROMPT_TEMPLATE = 'a sweet boy, 25 years old, {}'
CSV_PATH = 'en_action.csv'
PYTHON_PATH = '/environment/miniconda3/bin/python'

def get_latest_output_count():
    """Return the number of PNG files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.png')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new PNG file appears in the output directory"""
    timeout = 60  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def generate_script(image_path, seed, action):
    """Generate the ComfyUI script with the given parameters"""
    prompt = PROMPT_TEMPLATE.format(action)
    
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    control_net = ControlNetLoader('aes_stage2_control_net/aes_stage2_control.safetensors')
    model = UNETLoader('flux1-dev.safetensors', 'default')
    clip = DualCLIPLoader('t5xxl_fp16.safetensors', 'clip_l.safetensors', 'flux', 'default')
    clip_text_encode_positive_prompt_conditioning = CLIPTextEncode('{prompt}', clip)
    clip_text_encode_positive_prompt_conditioning = FluxGuidance(clip_text_encode_positive_prompt_conditioning, 3.5)
    clip_text_encode_positive_prompt_conditioning2 = CLIPTextEncode('', clip)
    image, _ = LoadImage('{image_path}')
    image2, _ = LoadImage('IMG_202305031758340.JPG')
    latent = EmptySD3LatentImage(1024, 1024, 1)
    vae = VAELoader('ae.safetensors')
    model, positive, negative, latent = FaceCombine(control_net, model, clip_text_encode_positive_prompt_conditioning, clip_text_encode_positive_prompt_conditioning2, image, image2, latent, 'aes_stage2_control_net/aes_stage2_img_proj.bin', 1.0000000000000002, 1, 0, 1, vae, 0.7000000000000002)
    latent = KSampler(model, {seed}, 30, 1, 'euler', 'simple', positive, negative, latent, 1)
    image3 = VAEDecode(latent, vae)
    SaveImage(image3, 'ComfyUI')
"""
    return script_content

def main():
    # Load actions from CSV
    try:
        actions = pd.read_csv(CSV_PATH)["en_action"].tolist()
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate all combinations of seeds and image paths
    seed_image_combinations = list(product(SEEDS, IMAGE_PATHS))
    
    # Main generation loop
    for action in actions:
        for seed, image_path in seed_image_combinations:
            # Generate script
            script = generate_script(image_path, seed, action)
            
            # Write script to file
            with open('run_xiang_infiniteyou.py', 'w') as f:
                f.write(script)
            
            # Get current output count before running
            initial_count = get_latest_output_count()
            
            # Run the script
            print(f"Generating image with action: {action}, seed: {seed}, image: {image_path}")
            subprocess.run([PYTHON_PATH, 'run_xiang_infiniteyou.py'])
            
            # Wait for new output
            if not wait_for_new_output(initial_count):
                print("Timeout waiting for new output. Continuing to next generation.")
                continue

if __name__ == "__main__":
    main()
```



## 📜 Introduction  

🚀 **InfiniteYou** is a **SOTA zero-shot identity preservation** model by **ByteDance**, built on **FLUX**. This repo brings it to **ComfyUI**, powered by **ZenAI**.  

🔗 **References:**  
📄 [Paper](https://arxiv.org/abs/2503.16418) | 💾 [Official Repo](https://github.com/bytedance/InfiniteYou)  

💡 Inspired by **ZenID** 🔗 [ZenID Repo](https://github.com/vuongminh1907/ComfyUI_ZenID)  

🔥 Stay tuned for updates!  

---

## 🏆 Model Zoo

The main author has released two versions of the model, each tailored for a specific purpose:  

- 🔹 **sim_stage1** – Prioritizes **higher identity similarity** for more accurate face preservation.  
- 🎨 **aes_stage2** – Focuses on **better text-image alignment** and enhanced **aesthetics**.  

To ensure seamless integration with **ComfyUI**, we have converted the model to the **safetensors** format. 


**Download the model on Hugging Face:**  
👉 [ComfyUI_InfiniteYou](https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou) 
---

## 🛠️ Workflow
### **Zero-Shot Task**
![Musk](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/musk.png)

### **FaceCombine Task**
![Children](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/face_combine_workflow.png)

### **FaceSwap Task**
![FaceSwap](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/face_swap.jpg)
---

## 📦 Installation

### Step 1: Clone the Repo
```bash
cd custom_nodes
git clone https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou
```
### Step 2: Install Requirements
```bash
cd ComfyUI_InfiniteYou
pip install -r requirements.txt
```

### Step 3: Download ControlNet Models
Place the ControlNet Models in the `ComfyUI/models/controlnet` directory.
```bash
cd ../../models/controlnet
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/aes_stage2_control_net/aes_stage2_control.safetensors
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/sim_stage1_control_net/sim_stage1_control_net.safetensors
cd ..
```
### Step 4: Download Image Projection Files
Place the Image Projection files in the `ComfyUI/models/InfiniteYou` directory.
```bash
mkdir InfiniteYou
cd InfiniteYou
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/aes_stage2_control_net/aes_stage2_img_proj.bin
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/sim_stage1_control_net/sim_stage1_img_proj.bin
cd ../..
```
### Step 5: Download InsightFace model
The InsightFace model is **antelopev2** (not the classic buffalo_l). Download the models (for example from [here](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing) or [here](https://huggingface.co/MonsterMMORPG/tools/tree/main)), unzip and place them in the `ComfyUI/models/insightface/models/antelopev2` directory.

Alternatively, you can quickly download all **models** using the following command:
```
# make sure you are in the ComfyUI directory
cd custom_nodes/
git clone https://github.com/ZenAI-Vietnam/ComfyUI_InfiniteYou
python ComfyUI_InfiniteYou/downloadmodel.py
pip install -r ComfyUI_InfiniteYou/requirements.txt
```

## 🧭 Usage

🔹 For `aes_stage2`: Try file `aes_stages2.json` in `workflows`

🔹 For `sim_stage1`: Try file `sim_stages1.json` in `workflows`

🔹 For `Face Combine` to predict your future children: Try file `face_combine.json` in `workflows`

🔹 For `Face Swap` : Try file `face_swap.json` in `workflows`



## 📞 Contact for Work 🌟
This implementation of InfiniteYou is brought to you by the ZenAI Team.

<img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenai.png" width="400" />

If you need more polished and enhanced version, please contact us through:  
- 📱 **Facebook Page**: [ZenAI](https://web.facebook.com/zenai.vn)  
- ☎️ **Phone**: 0971912713 Miss. Chi 

## 📖 Citation

```bibtex
@article{jiang2025infiniteyou,
  title={{InfiniteYou}: Flexible Photo Recrafting While Preserving Your Identity},
  author={Jiang, Liming and Yan, Qing and Jia, Yumin and Liu, Zichuan and Kang, Hao and Lu, Xin},
  journal={arXiv preprint},
  volume={arXiv:2503.16418},
  year={2025}
}
```
