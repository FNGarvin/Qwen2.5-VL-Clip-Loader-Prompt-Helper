# ComfyUI Qwen2.5-VL Loader & Prompt Helper

A unified custom node for ComfyUI that integrates the **Qwen2.5-VL** Vision-Language Model. 

This node serves two powerful purposes simultaneously:
1. **Smart Prompt Engine:** It uses the LLM to generate rich image descriptions from input images, expand simple text prompts, or blend both into a cohesive concept.
2. **Diffusion Conditioning:** It wraps the model to act as a fully compatible Text Encoder (CLIP), allowing you to pipe the generated descriptions directly into Qwen-Image or other compatible diffusion models without loading a second encoder.

## Features

- **Single-Node Solution:** Handles model loading, vision analysis, text generation, and conditioning encoding in one place.
- **Dual-Use Architecture:** Uses the same VRAM-loaded model for both high-level reasoning (Chat/Description) and low-level embedding generation.
- **Smart Modes:**
  - **Image Only:** Describes the input image in extreme detail.
  - **Text Only:** Expands short prompts into rich, descriptive prompts.
  - **Hybrid (Image + Text):** Conceptually blends the visual elements of an input image with a text concept.

## Installation

### 1. Install the Node
Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/FNGarvin/Qwen2.5-VL-Clip-Loader-Prompt-Helper.git
```

### 2. Install Python Requirements
This node relies on specific libraries to handle the Qwen architecture and vision processing. You must install these in your ComfyUI python environment:

```bash
pip install transformers qwen-vl-utils accelerate bitsandbytes
```
*Note: `bitsandbytes` is required if you are using 4-bit or 8-bit quantized models (highly recommended for VRAM efficiency).*  
*Note: It's a good idea to ensure you're running recent `transformers`* 

## Model Setup (Critical!)

This node uses the official Hugging Face `transformers` loading method. **You cannot simply download a single `.safetensors` file.**

You must download the **entire model repository** (all files, including `config.json`, `tokenizer.json`, `preprocessor_config.json`, etc.) and place the folder inside your `models/text_encoders/` directory.

**Correct Folder Structure:**
```
ComfyUI/
  models/
    text_encoders/
      Qwen2.5-VL-7B-Instruct-bnb-4bit/  <-- Select this folder in the node
        ????????? config.json
        ????????? model.safetensors
        ????????? tokenizer.json
        ????????? preprocessor_config.json
        ????????? ... (other repo files)
```

## Usage

1. **Add Node:** Search for `Qwen2.5 VL Clip Loader / Prompt Helper`.
2. **Select Model:** Choose your Qwen2.5 VL model folder from the dropdown.
3. **Connect Inputs:**
   - **Image (Optional):** Connect an image to have the model describe or blend it.
   - **Text (Optional):** Enter a concept or simple prompt.
4. **Connect Outputs:**
   - **CLIP:** Connect this to your `CLIP Text Encode` node (or any node expecting a CLIP model).
   - **String:** Connect this to a `Show Text` node or directly into the text widget of your encoder to see the generated description.

## Disclaimer & Legal

This software is an unofficial community contribution and is **not** affiliated with, endorsed by, or sponsored by Alibaba Cloud, the Qwen team, or ComfyUI. 

- **Qwen** and **Alibaba Cloud** are trademarks of Alibaba Group.
- **ComfyUI** is the property of its respective owners.

This project claims no ownership over the model weights or the trademarks associated with the underlying technologies. Users are responsible for adhering to the specific licenses of the models they download and use.
