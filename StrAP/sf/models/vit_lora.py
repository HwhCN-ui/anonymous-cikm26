import torch.nn as nn
from transformers import ViTForImageClassification, AutoImageProcessor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from sf.config.base_config import vit_lora as lora_cfg

class ViTLoRAClissifier(nn.Module):
    def __init__(self):
        super().__init__()
        num_labels = int(lora_cfg.get("num_labels", 200))
        vit_name = lora_cfg.get("vit_pre_model_name", "google/vit-base-patch16-224")
        
        self.base_model = ViTForImageClassification.from_pretrained(
            vit_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        ) 
        for p in self.base_model.parameters():
            p.requires_grad = False
            
        self.lora_cfg = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            modules_to_save=None, 
            task_type=TaskType.SEQ_CLS
        )
        
        self.lora_model = get_peft_model(self.base_model, self.lora_cfg)
        self.processor = AutoImageProcessor.from_pretrained(vit_name)
        
        try:
            self.lora_model.print_trainable_parameters()
        except:
            pass
        
    def forward(self, pixel_values=None, labels=None, return_loss=True):
        outputs = self.lora_model(
            pixel_values=pixel_values,
            labels=labels if return_loss else None,
            return_dict=True,
        )
        
        return {
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "logits": outputs.logits if hasattr(outputs, "logits") else outputs[0],
        }

    def state_dict(self, *args, **kwargs):
        return self.lora_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.lora_model.load_state_dict(state_dict, strict=strict)
    
    def save_lora(self, save_path):
        self.lora_model.save_pretrained(save_path)
    
    
    @classmethod
    def from_pretrained(cls, lora_path):
        model = cls()
        model.lora_model = PeftModel.from_pretrained(
            model.base_model, lora_path, is_trainable=True
        )
        return model  