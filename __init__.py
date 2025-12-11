"""
Author: FNGarvin
License: MIT License
Copyright (c) 2025 FNGarvin

Description:
Initialization file for the Qwen2.5 VL Clip Loader / Prompt Helper.
"""

print("loading Qwen2.5 VL Clip Loader / Prompt Helper")

from .nodes import Qwen25_VL_Loader_PromptHelper

NODE_CLASS_MAPPINGS = {
    "Qwen25_VL_Loader_PromptHelper": Qwen25_VL_Loader_PromptHelper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen25_VL_Loader_PromptHelper": "Qwen2.5 VL Clip Loader / Prompt Helper"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

#END OF __init__.py
