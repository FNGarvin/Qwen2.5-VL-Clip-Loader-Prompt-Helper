"""
Author: FNGarvin
License: MIT License
Copyright (c) 2025 FNGarvin

Description:
Core logic for the Qwen2.5 VL Clip Loader / Prompt Helper.

This node:
1. Loads the Qwen2.5-VL model using the official transformers API.
2. Wraps it in a custom 'QwenCLIPWrapper' compatible with standard ComfyUI CLIP nodes.
3. Fixes 'FloatTensor' crashes by forcing Long/Int inputs.
4. Fixes 'IndexError' crashes by preventing empty token sequences.
5. Fixes 'Glitchy Image' output by enforcing a safe token limit (1024) for conditioning.
"""

import os
import torch
import gc
import numpy as np
from PIL import Image
import folder_paths
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Global cache to prevent reloading the heavy model on every execution
_MODEL_CACHE = {
    "path": None,
    "device": None,
    "wrapper": None
}

class QwenCLIPWrapper:
    """
    Wraps the Qwen2.5-VL model to mimic the interface expected by
    standard ComfyUI 'CLIP Text Encode' nodes.
    """
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.cond_stage_model = model
        self.patcher = self 
        self.layer_idx = None 

    def tokenize(self, text, return_word_ids=False):
        """
        Tokenizes text using the Qwen processor.
        Returns a clean dictionary of LongTensors.
        """
        if isinstance(text, str):
            text = [text]
            
        # Use processor to tokenize
        # CRITICAL FIX: Limit max_length to 1024. 
        # 2048+ tokens can overflow the Diffusion Model's context window, causing glitchy/static images.
        encoding = self.processor(
            text=text, 
            images=None, 
            videos=None, 
            padding="longest",
            truncation=True,
            max_length=1024, 
            return_tensors="pt"
        )
        
        # Explicitly cast to Long (Int64) to prevent "Expected Long but got Float" errors
        input_ids = encoding["input_ids"].long().to(self.device)
        attention_mask = encoding["attention_mask"].long().to(self.device)
        
        # Safety Guard: If tokenization somehow yields empty tensor (e.g. empty string)
        if input_ids.numel() == 0:
            print("[Qwen2.5-VL] Warning: Empty text input. Injecting padding token.")
            pad_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
            input_ids = torch.tensor([[pad_id]], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([[1]], dtype=torch.long, device=self.device)

        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
            
        return tokens

    def encode_from_tokens(self, tokens, return_pooled=False):
        """
        Runs the model forward pass to get embeddings.
        """
        # Ensure we are using the tensors from our clean dictionary
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Extract last hidden state [Batch, Seq_Len, Hidden_Size]
            last_hidden_state = outputs.hidden_states[-1]
            
            # Return None for pooled output (Qwen doesn't use standard CLIP pooling)
            return last_hidden_state, None

    def encode_from_tokens_scheduled(self, tokens):
        """
        Required by CLIPTextEncode node.
        Performs encoding and packages the result into the standard ComfyUI 
        conditioning format.
        """
        cond, pooled = self.encode_from_tokens(tokens)
        return [[cond, {"pooled_output": pooled}]]

class Qwen25_VL_Loader_PromptHelper:
    @classmethod
    def INPUT_TYPES(s):
        # Scan standard ComfyUI text_encoders folder
        model_list = folder_paths.get_filename_list("text_encoders")
        
        return {
            "required": {
                "model_name": (model_list,),
                "device": (["cuda", "cpu"],),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "text_input": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "generated_text")
    FUNCTION = "run"
    CATEGORY = "Qwen-Image"

    def _load_model(self, model_name, device):
        """Internal helper to load or retrieve cached model."""
        global _MODEL_CACHE
        
        # Get full path to the selected file
        model_path = folder_paths.get_full_path("text_encoders", model_name)
        
        # Directory Fix: Ensure we point to the folder if a file was selected
        if os.path.isfile(model_path):
            model_path = os.path.dirname(model_path)
        
        # Check Cache
        if _MODEL_CACHE["wrapper"] is not None:
            if _MODEL_CACHE["path"] == model_path and _MODEL_CACHE["device"] == device:
                return _MODEL_CACHE["wrapper"]
            else:
                print(f"[Qwen2.5-VL] Unloading previous model...")
                del _MODEL_CACHE["wrapper"]
                _MODEL_CACHE["wrapper"] = None
                gc.collect()
                torch.cuda.empty_cache()

        print(f"[Qwen2.5-VL] Loading new model from: {model_path}")
        
        try:
            # Official Loading Logic (relying on config.json for quantization)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            
            processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                local_files_only=True
            )
            
            wrapper = QwenCLIPWrapper(model, processor, device)
            
            # Update Cache
            _MODEL_CACHE["path"] = model_path
            _MODEL_CACHE["device"] = device
            _MODEL_CACHE["wrapper"] = wrapper
            
            return wrapper
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen2.5-VL: {e}")

    def run(self, model_name, device, image_input=None, text_input=None):
        # 1. Load Model (Cached)
        wrapper = self._load_model(model_name, device)
        model = wrapper.model
        processor = wrapper.processor
        
        # 2. Input Validation & Setup
        has_image = image_input is not None
        has_text = text_input is not None and text_input.strip() != ""
        
        if not has_image and not has_text:
            raise ValueError("Qwen2.5-VL Error: You must provide an Image OR a Text Prompt.")

        # 3. Process Image
        pil_image = None
        if has_image:
            # ComfyUI Image Tensor [Batch, H, W, C] -> Take First -> PIL
            img_np = (image_input[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

        # 4. Construct Instruction (Hidden System Prompt)
        system_instruction = (
            "You are a creative assistant designed to generate detailed visual descriptions for image generation. "
            "Output ONLY the description. Do not provide conversational filler, introductions, or explanations."
        )
        
        messages = [{"role": "system", "content": system_instruction}]
        user_content = []

        # Logic Branching for Instruction
        if has_image and has_text:
            # HYBRID MODE
            user_content.append({"type": "image", "image": pil_image})
            instruction = (
                f"Study this image and the following concept: '{text_input}'. "
                "Write a detailed visual description that blends the visual elements of the image "
                "with the concepts in the text equally."
            )
            user_content.append({"type": "text", "text": instruction})
            print(f"[Qwen2.5-VL] Mode: Hybrid (Image + '{text_input}')")

        elif has_image:
            # IMAGE DESCRIBER MODE
            user_content.append({"type": "image", "image": pil_image})
            user_content.append({"type": "text", "text": "Describe this image in extreme detail."})
            print("[Qwen2.5-VL] Mode: Image Description")

        elif has_text:
            # TEXT EXPANSION MODE
            instruction = (
                f"Expand the following prompt into a rich, detailed visual description: '{text_input}'"
            )
            user_content.append({"type": "text", "text": instruction})
            print(f"[Qwen2.5-VL] Mode: Prompt Expansion ('{text_input}')")

        messages.append({"role": "user", "content": user_content})

        # 5. Generation
        text_fmt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_fmt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Fixed settings: High tokens, no sampling noise
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=2048
            )
        
        # Decode and trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        final_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[Qwen2.5-VL] Generated: {final_text[:100]}...")

        # 6. Return
        # We return the Wrapper (as CLIP) so downstream nodes can encode positive/negative prompts.
        # We return final_text so the user can feed it into the positive prompt encoder.
        return (wrapper, final_text)

#END OF nodes.py
