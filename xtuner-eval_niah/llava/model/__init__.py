import os

AVAILABLE_MODELS = {
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_qwen_flash": "LlavaQwenForCausalLM_Flash, LlavaQwenConfig_Flash"
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")
