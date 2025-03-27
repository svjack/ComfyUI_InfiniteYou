# 🚀 ComfyUI_InfiniteYou
An implementation of **InfiniteYou** for **ComfyUI**

Native support for [InfiniteYou](https://github.com/bytedance/InfiniteYou) in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed by the ZenAI team.  


**✨ Support further development by starring the project! ✨**


![teaser](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/teaser.jpg)

---

## 🔥 News
- **[03/2025]** 🔥 Code updated and released as the first version.

---

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
![Musk](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/musk.png)
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
```bash
cd ../../models/controlnet
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/aes_stage2_control_net/aes_stage2_control.safetensors
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/sim_stage1_control_net/sim_stage1_control_net.safetensors
cd ..
```
### Step 4: Download Image Projection Files
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
python custom_nodes/ComfyUI_InfiniteYou/downloadmodel.py
```

## 🧭 Usage

🔹 For `aes_stage2`: Try file `aes_stages2.json` in `workflows`

🔹 For `sim_stage1`: Try file `sim_stages1.json` in `workflows`


## 📞 Contact for Work 🌟
This implementation of InfiniteYou is brought to you by the ZenAI Team.

<img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenai.png" width="400" />

If you need more polished and enhanced version, please contact us through:  
- 📱 **Facebook Page**: [ZenAI](https://web.facebook.com/zenai.vn)  
- ☎️ **Phone**: 0971912713 Miss. Chi  
