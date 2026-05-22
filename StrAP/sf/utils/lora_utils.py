import torch

def is_lora_key(k: str) -> bool:
    return ("lora_A" in k) or ("lora_B" in k)

def extract_lora_state(model, device="cpu"):
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if is_lora_key(k):
            t = v.detach()
            if device is not None:
                t = t.to(device)
            out[k] = t.clone()
    return out

def extract_lora_update(global_model, local_model, device="cpu"):
    g = extract_lora_state(global_model, device=device)
    l = extract_lora_state(local_model, device=device)

    update = {}
    for k, lv in l.items():
        gv = g.get(k, None)
        if gv is None:
            update[k] = lv.clone()
        else:
            gv = gv.to(device=lv.device, dtype=lv.dtype)
            update[k] = (lv - gv).clone()
    return update


def apply_lora_update(global_model, update):
    g = global_model.lora_model.state_dict()
    for k, dv in update.items():
        if k in g:
            g[k] = g[k] + dv.to(g[k].dtype).to(g[k].device)
        else:
            pass
    global_model.lora_model.load_state_dict(g, strict=False)
    return global_model

def compute_lora_norm(model):
    norm = 0.0
    for k, v in model.state_dict().items():
        if is_lora_key(k):
            norm += torch.norm(v.float(), p="fro").item()
    return norm


def iter_lora_pairs(state_dict):
    keys = list(state_dict.keys())
    for k in keys:
        if "lora_A" in k:
            b = k.replace("lora_A", "lora_B")
            if b in state_dict:
                yield k, b