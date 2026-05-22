import torch
from typing import Dict, List, Tuple, Optional
from sf.config.base_config import train as train_cfg

device = torch.device(train_cfg["device"])

def _to_device(client_updates, device: torch.device=device):
    return [{k: v.to(device) for k, v in u.items()} for u in client_updates]

def _keys(client_updates):
    return list(client_updates[0].keys())

def _flatten_update(update: Dict) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in update.values()]).float()

def _return_metrics(total_time: float,mode: str=None) -> Dict[str, float]:
    return {
        "mode": mode,
        "stage1_iter": 0.0,
        "stage2_max_sigma": 0.0,
        "stage1_time": 0.0,
        "stage2_time": 0.0,
        "total_time": total_time
    }

def _flat(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in update.values()]).float()

def _mean_updates(client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    outputs = {}
    for key in client_updates[0].keys():
        outputs[key] = torch.mean(torch.stack([update[key] for update in client_updates], dim=0), dim=0)
    return outputs

def _weight_sum_updates(client_updates: List[Dict[str, torch.Tensor]], weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    num_clients = len(client_updates)
    outputs = {}
    for key in client_updates[0].keys():
        stacked_updates = torch.stack([update[key] for update in client_updates], dim=0)
        w = weights.to(stacked_updates.device, dtype=stacked_updates.dtype).view(
            (num_clients,) + (1,) * (stacked_updates.dim() - 1)
        )       
        outputs[key] = torch.sum(stacked_updates * w, dim=0)

    return outputs


__all__ = [
    "_to_device",
    "_keys",
    "_flatten_update",
    "_return_metrics",
    "_flat",
    "_mean_updates",
    "_weight_sum_updates",
]