import torch
import torch.nn.functional as F
from tqdm import tqdm
from sf.data import get_dataset
from sf.config.base_config import federated as fl_cfg
from sf.config.attack_config import backdoor as bd_cfg


def _make_tact_patch(channels: int, size: int, device: torch.device):
    seed = int(bd_cfg.get("tact_seed", 0))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    patch = torch.rand((1, channels, size, size), generator=gen, device=device, dtype=torch.float32)
    return patch


def _make_blended_pattern(channels: int, h: int, w: int, device: torch.device):
    seed = int(bd_cfg.get("blended_seed", 0))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pat = torch.rand((1, channels, h, w), generator=gen, device=device, dtype=torch.float32)
    return pat


def _get_trigger_position(h: int, w: int, size: int):
    pos = bd_cfg.get("trigger_position", "bottom_right")
    if pos == "top_left":
        return 0, 0
    if pos == "random":
        y0 = int(torch.randint(low=0, high=max(1, h - size + 1), size=(1,)).item())
        x0 = int(torch.randint(low=0, high=max(1, w - size + 1), size=(1,)).item())
        return y0, x0
    return h - size, w - size


def _apply_trigger_tensor(x: torch.Tensor,trigger_type: str):
    ttype = str(trigger_type).lower().strip()
    if ttype == "none":
        raise ValueError("trigger_type cannot be 'none' when applying trigger.")
    
    size = int(bd_cfg.get("trigger_size", 4))

    b, c, h, w = x.shape
    size = max(1, min(size, h, w))

    if ttype == "blended":
        alpha = float(bd_cfg.get("blended_alpha", 0.2))
        alpha = max(0.0, min(1.0, alpha))
        pat = _make_blended_pattern(channels=c, h=h, w=w, device=x.device)
        pat = pat.expand(b, -1, -1, -1)
        return (1.0 - alpha) * x + alpha * pat

    y0, x0 = _get_trigger_position(h, w, size)

    if ttype == "tact":
        patch = _make_tact_patch(channels=c, size=size, device=x.device)
        if patch.shape[-1] != size:
            patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)
        patch = patch.expand(b, -1, -1, -1)
        out = x.clone()
        out[:, :, y0:y0 + size, x0:x0 + size] = patch
        return out
    val = float(bd_cfg.get("trigger_value", 1.0))
    out = x.clone()
    out[:, :, y0:y0 + size, x0:x0 + size] = val
    return out


@torch.no_grad()
def evaluate_cls(model, data_name, eval_data_path, device, topk=(1,)):
    ds = get_dataset(eval_data_path, model_name="vit", client_id=0, data_name=data_name, is_server=True)
    loader = ds.get_loader(batch_size=fl_cfg["batch_size"], shuffle=False)

    model.eval()
    model.to(device)

    correct1 = 0
    total = 0

    for batch in tqdm(loader, desc="Eval Clean"):
        x = batch["pixel_values"].to(device)
        y = torch.tensor(batch["labels"]).to(device)

        out = model(pixel_values=x, labels=None, return_loss=False)
        logits = out["logits"]
        pred = torch.argmax(logits, dim=1)

        correct1 += (pred == y).sum().item()
        total += y.size(0)

    top1 = correct1 / max(1, total)
    return {"top1": top1}


@torch.no_grad()
def evaluate_backdoor_asr(model, data_name,trigger_type, eval_data_path, device):
    target = int(bd_cfg.get("target_label", 0))

    ds = get_dataset(eval_data_path, model_name="vit", client_id=0, data_name=data_name, is_server=True)
    loader = ds.get_loader(batch_size=fl_cfg["batch_size"], shuffle=False)

    model.eval()
    model.to(device)

    total = 0
    success = 0

    for batch in tqdm(loader, desc="Eval Backdoor"):
        x = batch["pixel_values"]
        y = torch.tensor(batch["labels"])

        mask = (y != target)
        if mask.sum().item() == 0:
            continue

        x = x[mask].to(device)
        y = y[mask].to(device)

        x = _apply_trigger_tensor(x,trigger_type)

        out = model(pixel_values=x, labels=None, return_loss=False)
        logits = out["logits"]
        pred = torch.argmax(logits, dim=1)

        success += (pred == target).sum().item()
        total += pred.size(0)

    return success / max(1, total)
