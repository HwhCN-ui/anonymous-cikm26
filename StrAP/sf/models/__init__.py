from .vit_lora import ViTLoRAClissifier
def get_model(model_name: str, global_lora_path: str = None):
    name = (model_name or "").lower()
    if name in ["vit", "vit_lora", "vit-lora"]:
        if global_lora_path:
            return ViTLoRAClissifier.from_pretrained(global_lora_path)
        return ViTLoRAClissifier()
    raise ValueError(f"Unknown model_name: {model_name!r}")
