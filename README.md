# 🚀 ComfyUI_InfiniteYou
An implementation of **InfiniteYou** for **ComfyUI**

Native support for InfiniteYou in ComfyUI, designed by the ZenAI team.  
**✨ Support further development by starring the project! ✨**

---

## 🔥 News
- **[03/2025]** 🔥 Code updated and released as the first version.

---

## 📜 Introduction

**InfiniteYou** is a state-of-the-art (SOTA) 0-shot identity preservation model developed by ByteDance, based on **FLUX** models. This implementation brings InfiniteYou to **ComfyUI** by the **ZenAI team**.

For detailed reference to the technique and the original code, check the links below:
- **Research Paper**: [InfiniteYou Paper](https://arxiv.org/abs/2503.16418)
- **Original Repo**: [ByteDance InfiniteYou Repo](https://github.com/bytedance/InfiniteYou)

Special thanks to the concept and inspiration from **ZenID**: [ZenID Repo](https://github.com/vuongminh1907/ComfyUI_ZenID), which enabled us to complete this repo earlier.

---

## 🏆 Model Zoo

The main author has released two versions of the model: **aes_stage2** and **sim_stage1**, each with distinct purposes:
- **sim_stage1**: Optimized for higher ID similarity.
- **aes_stage2**: Optimized for better text-image alignment and aesthetics.

In addition to the Hugging Face public models, we have updated the checkpoint format to **safetensors**, which is required for ComfyUI usage.

- **Hugging Face Model Link**: [ComfyUI_InfiniteYou](https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou)
![Trump](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/workflow_example.png)
![Musk](https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou/blob/main/assets/musk.png)
---

## 🛠️ Workflow

---

## 📦 Installation

### Step 1: Download ControlNet Models
```bash
cd ComfyUI/models/controlnet
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/aes_stage2_control_net/aes_stage2_control.safetensors
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/sim_stage1_control_net/sim_stage1_control_net.safetensors
cd ..
```
### Step 2: Download Image Projection Files
```bash
mkdir InfiniteYou
cd InfiniteYou
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/aes_stage2_control_net/aes_stage2_img_proj.bin
wget https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/resolve/main/sim_stage1_control_net/sim_stage1_img_proj.bin
cd ../..
```
### Step 3: Clone the Repo
```bash
cd custom_nodes
git clone https://github.com/ZenAI-Comfy/ComfyUI_InfiniteYou
```
### Step 4: Install Requirements
```bash
cd ComfyUI_InfiniteYou
pip install -r requirements.txt
```
Alternatively, you can quickly download the model using the following command:
```
python ComfyUI/custom_nodes/ComfyUI_InfiniteYou/downloadmodel.py
```

## 🧭 Usage

- If you're using **aes\_stage2**, make sure to load both the **ControlNet model** (`aes_stage2_control.safetensors`) and the **Image Projection** (`aes_stage2_img_proj.bin`) at the same time

- Conversely, if you're using **sim_stage1**, select the corresponding models.

## 📞 Contact for Work 🌟
This implementation of InfiniteYou is brought to you by the ZenAI Team.

<img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenai.png" width="400" />

If you need more polished and enhanced version, please contact us through:  
- 📱 **Facebook Page**: [ZenAI](https://web.facebook.com/zenai.vn)  
- ☎️ **Phone**: 0971912713 Miss. Chi  
