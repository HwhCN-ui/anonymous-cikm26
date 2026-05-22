from datetime import datetime
import os


date_str = datetime.now().strftime("%Y%m%d_%H%M")

train = {
    "device": "cuda:0",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
}

federated = {
    "num_clients": 100,
    "clients_per_round": 30,
    "num_rounds": 100,
    "local_epochs": 1,
    "batch_size": 64,
    "num_workers": 16,
    "prefetch_factor": 16,
    "accumulation_steps": 1,
    "seed": 42,
}

log = {
    "log_dir": os.path.join("./logs", date_str),
    "log_name": "strap.log",
}

tensorboard = {
    "log_dir": os.path.join("./tb_logs", date_str),
    "log_freq": 1,
}

lora = {
    "global_lora_path": os.path.join("./models", date_str, "global_lora", "round_0"),
}

vit_lora = {
    "vit_pre_model_name": "./datasets/google/vit-base-patch16-224",
    "num_labels": 200,

    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,

    "target_modules": ["query", "key", "value"],  
}

data = {
    "name": "tiny_imagenet",
    "image_size": 224,
    "num_classes": 200,
}