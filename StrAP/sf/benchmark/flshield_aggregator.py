import time
import math
from typing import List, Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPProcessor

from sf.models import get_model
from sf.utils.lora_utils import apply_lora_update
from sf.data import get_dataset
from sf.config.base_config import lora as lora_cfg, vit_lora as vit_cfg
from sf.config.base_config import federated as fl_cfg

from .__init__ import (
    _to_device,
    _keys,
    _flat,
    _mean_updates,
    device as default_device,
    _return_metrics,
)
try:
    from transformers import AutoImageProcessor 
except Exception:
    AutoImageProcessor = None


class FLShieldAggregator:
    def __init__(
        self,
        device: torch.device = default_device,
        global_lora_path: Optional[str] = None,
        federated_data_path: Optional[str] = None,
        model_name: Optional[str] = None,
        data_name: Optional[str] = None,
        keep_ratio: float = 0.5,
        clip_coef: float = 10.0,
        val_batch_size: int = 8,
        max_val_batches: Optional[int] = 10,
        eps: float = 1e-12,
    ):
        assert model_name is not None, "FLShieldAggregator: model_name cannot be None"

        self.device = device
        self.data_name=data_name
        self.global_lora_path = global_lora_path
        self.federated_data_path = federated_data_path
        self.model_name = model_name

        self.keep_ratio = keep_ratio
        self.clip_coef = clip_coef
        self.val_batch_size = val_batch_size
        self.max_val_batches = max_val_batches
        self.eps = eps

    def _make_metrics(self, elapsed: float, **extra) -> Dict:
        m = _return_metrics(elapsed, mode="flshield")
        m.update(extra)
        return m

    def _unflatten_like(self, vec: torch.Tensor, ref_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}
        idx = 0
        for k, v in ref_update.items():
            numel = v.numel()
            slice_ = vec[idx: idx + numel].view_as(v)
            outputs[k] = slice_.clone().to(self.device)
            idx += numel
        return outputs

    def _build_val_loader(self, federated_data_path: str) -> DataLoader:
        name = (self.model_name or "").lower()
        if name.startswith("clip"):
            processor = CLIPProcessor.from_pretrained(lora_cfg["pre_model_name"])
        else:
            processor = AutoImageProcessor.from_pretrained(
                vit_cfg["vit_pre_model_name"]
            )

        ds = get_dataset(federated_data_path, model_name=self.model_name, client_id=0, data_name=self.data_name, is_server=True)
        loader = ds.get_loader(batch_size=fl_cfg["batch_size"], shuffle=False)
        return loader

    @torch.no_grad()
    def _avg_loss_on_loader(
        self,
        model,
        loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> float:
        model.eval()
        total_loss = 0.0
        steps = 0
        for _, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(self.device)

            if "labels" in batch:
                labels = batch["labels"].to(self.device).long()
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                    return_loss=True,
                )
            else:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True,
                )

            loss = outputs["loss"]
            total_loss += float(loss.item())
            steps += 1
            if max_batches is not None and steps >= max_batches:
                break
        return total_loss / max(1, steps)

    def _apply_update_to_global(self, base_lora_path: str, update_dict: Dict[str, torch.Tensor]):
        model = get_model(model_name=self.model_name, global_lora_path=base_lora_path)
        model.to(self.device)
        model = apply_lora_update(model, update_dict)
        model.to(self.device)
        return model

    def _generate_representatives_bijective(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Set[int]]]:
        m = len(client_updates)
        vecs = torch.stack([_flat(u) for u in client_updates], dim=0).float().to(self.device) 
        normed = F.normalize(vecs, p=2, dim=1)                                               
        cos_mat = torch.mm(normed, normed.t())                                                

        relu_cos = torch.clamp(cos_mat, min=0.0)
        row_sums = relu_cos.sum(dim=1, keepdim=True) + self.eps
        weights = relu_cos / row_sums                                                        

        rep_vecs = torch.mm(weights, vecs)                                                   

        rep_to_clients: List[Set[int]] = []
        for i in range(m):
            contrib_idx = torch.nonzero(weights[i] > 0, as_tuple=False).view(-1).tolist()
            rep_to_clients.append(set(contrib_idx))

        template = client_updates[0]
        rep_updates: List[Dict[str, torch.Tensor]] = []
        for i in range(m):
            rep_updates.append(self._unflatten_like(rep_vecs[i], template))

        return rep_updates, rep_to_clients

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        client_ids: Optional[List[int]] = None,
        is_init_benign: bool = False,
        clean_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        current_round: Optional[int] = None,
        global_lora_path: Optional[str] = None,
        federated_data_path: Optional[str] = None,
    ):
        start_time = time.perf_counter()

        glp = global_lora_path or self.global_lora_path
        fdp = federated_data_path or self.federated_data_path
        assert glp is not None, "FLShieldAggregator.aggregate: global_lora_path is None"
        assert fdp is not None, "FLShieldAggregator.aggregate: federated_data_path is None"

        client_updates = _to_device(client_updates=client_updates, device=self.device)

        rep_updates, rep_to_clients = self._generate_representatives_bijective(client_updates)

        val_loader = self._build_val_loader(fdp)
        global_model_before = get_model(model_name=self.model_name, global_lora_path=glp)
        global_model_before.to(self.device)
        base_loss = self._avg_loss_on_loader(
            global_model_before, val_loader, max_batches=self.max_val_batches
        )

        scores = []
        for rep_upd in rep_updates:
            rep_model = self._apply_update_to_global(glp, rep_upd)
            rep_loss = self._avg_loss_on_loader(
                rep_model, val_loader, max_batches=self.max_val_batches
            )
            score = rep_loss - base_loss
            scores.append(score)

        num_reps = len(rep_updates)
        k_keep = max(1, int(math.ceil(self.keep_ratio * num_reps)))
        order = sorted(range(num_reps), key=lambda idx: scores[idx])  
        chosen_rep_idx = order[:k_keep]

        chosen_client_indices: Set[int] = set()
        for ridx in chosen_rep_idx:
            chosen_client_indices |= rep_to_clients[ridx]
        if len(chosen_client_indices) == 0:
            chosen_client_indices = set(range(len(client_updates)))
        chosen_client_indices = sorted(list(chosen_client_indices))

        chosen_updates = [client_updates[i] for i in chosen_client_indices]
        norms = torch.tensor(
            [_flat(u).norm().item() for u in chosen_updates],
            dtype=torch.float32,
            device=self.device,
        )
        if norms.numel() == 0:
            norms = torch.tensor([1.0], device=self.device)

        median_norm = torch.median(norms).item()
        clip_bound = self.clip_coef * median_norm + self.eps

        clipped_updates: List[Dict[str, torch.Tensor]] = []
        for u, n in zip(chosen_updates, norms):
            n_val = n.item() + self.eps
            scale = min(1.0, clip_bound / n_val)
            u_clip = {k: (v * scale) for k, v in u.items()}
            clipped_updates.append(u_clip)

        aggregated_update = _mean_updates(clipped_updates)

        end_time = time.perf_counter()
        avg_score = float(sum(scores) / max(1, len(scores)))
        metrics = self._make_metrics(
            end_time - start_time,
            avg_score=avg_score,
            clip_bound=clip_bound,
            num_selected_clients=len(chosen_client_indices),
            keep_ratio=self.keep_ratio,
        )

        return aggregated_update, metrics