#!/usr/bin/env python3

import time
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from torch.optim import AdamW

from sf.models import get_model
from sf.data import get_dataset
from sf.utils.lora_utils import extract_lora_update
from sf.config.base_config import lora as lora_cfg, vit_lora as vit_cfg
from .__init__ import _to_device, _keys, _flat, _mean_updates, device, _return_metrics
from sf.config.base_config import federated as fl_cfg

from transformers import CLIPProcessor
try:
    from transformers import AutoImageProcessor 
except Exception:
    AutoImageProcessor = None

class FLTrustAggregator:
    def __init__(self,
                 device: torch.device = device,
                 global_lora_path: Optional[str] = None,
                 federated_data_path: Optional[str] = None,
                 model_name: str = None,
                 data_name: str = None,
                 root_steps: int = 100,
                 root_lr: float = 1e-4,
                 root_weight_decay: float = 1e-4,
                 root_batch_size: int = 128,
                 clip_coef: float = 1.0,
                 recompute_root_every: int = 1,
                 eps: float = 1e-12):
        self.device = device
        self.data_name = data_name
        self.global_lora_path = global_lora_path
        self.federated_data_path = federated_data_path
        assert model_name is not None, "FLTrustAggregator: model_name cannot be None"
        self.model_name = model_name
        self.task = self._task_from_model(self.model_name)

        self.root_steps = int(root_steps)
        self.root_lr = float(root_lr)
        self.root_weight_decay = float(root_weight_decay)
        self.root_batch_size = int(root_batch_size)
        self.clip_coef = float(clip_coef)
        self.recompute_root_every = int(recompute_root_every)
        self.eps = float(eps)

        self._cached_root_update: Optional[Dict[str, torch.Tensor]] = None
        self._cached_round_idx: Optional[int] = None

    @staticmethod
    def _task_from_model(model_name: str) -> str:
        u = (model_name or "").upper()
        if "CLIP" in u: return "clip"
        if "VIT"  in u: return "vit"
        raise ValueError(f"FLTrustAggregator: unknown model_name '{model_name}'")

    def _make_metrics(self, elapsed: float, **extra) -> Dict:
        m = _return_metrics(elapsed, mode="fltrust")
        m.update(extra)
        return m

    def _build_processor(self):
        if self.task == "clip":
            return CLIPProcessor.from_pretrained(lora_cfg["pre_model_name"])
        if AutoImageProcessor is None:
            raise RuntimeError("AutoImageProcessor not available; install transformers>=4.26 or provide a custom processor.")
        return AutoImageProcessor.from_pretrained(vit_cfg["vit_pre_model_name"])

    @torch.no_grad()
    def _clone_model(self, global_lora_path: str):
        m = get_model(model_name=self.model_name, global_lora_path=global_lora_path)
        m.to(self.device)
        return m

    def _compute_root_update(self, global_lora_path: str, federated_data_path: str):
        model_before = self._clone_model(global_lora_path)
        model_before.eval()
        model_after  = self._clone_model(global_lora_path)
        model_after.train()

        processor = self._build_processor()
        ds = get_dataset(federated_data_path, model_name=self.model_name, client_id=0, data_name=self.data_name, is_server=True)
        loader = ds.get_loader(batch_size=fl_cfg["batch_size"], shuffle=False)
        optim  = AdamW(model_after.parameters(), lr=self.root_lr, weight_decay=self.root_weight_decay)

        steps = 0
        for batch in tqdm(loader, desc="FLTrustAggregator: root update"):
            optim.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(self.device)

            if self.task == "clip":
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = model_after(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            else:
                labels = batch["labels"].to(self.device)
                outputs = model_after(pixel_values=pixel_values, labels=labels, return_loss=True)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            loss.backward()
            optim.step()
            steps += 1
            if steps >= self.root_steps:
                break

        root_update = extract_lora_update(model_before, model_after)
        return {k: v.detach().to(self.device) for k, v in root_update.items()}

    def aggregate(self,
                  client_updates: List[Dict[str, torch.Tensor]],
                  client_sizes: List[int],
                  client_ids: Optional[List[int]] = None,
                  is_init_benign: bool = False,
                  clean_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
                  current_round: Optional[int] = None,
                  global_lora_path: Optional[str] = None,
                  federated_data_path: Optional[str] = None):
        start_time = time.perf_counter()
        client_updates = _to_device(client_updates=client_updates, device=self.device)

        glp = global_lora_path or self.global_lora_path
        fdp = federated_data_path or self.federated_data_path
        assert glp is not None, "FLTrustAggregator: global_lora_path is None"
        assert fdp is not None, "FLTrustAggregator: federated_data_path is None"

        need_recompute = (
            self._cached_root_update is None
            or self._cached_round_idx is None
            or self.recompute_root_every == 1
            or (current_round is not None and
                ((current_round - (self._cached_round_idx or 0)) % max(1, self.recompute_root_every) == 0))
        )
        if need_recompute:
            r = self._compute_root_update(glp, fdp)
            self._cached_root_update = r
            self._cached_round_idx = current_round if current_round is not None else 0
        else:
            r = self._cached_root_update

        r_vec = _flat(r)
        r_norm = r_vec.norm().item() + self.eps
        clip_bound = self.clip_coef * r_norm

        scores, clipped_updates = [], []
        for u in client_updates:
            u_vec = _flat(u)
            u_norm = u_vec.norm().item() + self.eps
            cos_ur = torch.nn.functional.cosine_similarity(u_vec.unsqueeze(0), r_vec.unsqueeze(0)).item()
            si = max(0.0, float(cos_ur))
            scale = min(1.0, clip_bound / u_norm)
            u_clip = {k: v * scale for k, v in u.items()}
            scores.append(si)
            clipped_updates.append(u_clip)

        sum_s = sum(scores)
        if sum_s <= self.eps:
            outputs = _mean_updates(clipped_updates)
            elapsed = time.perf_counter() - start_time
            metrics = self._make_metrics(elapsed, r_norm=r_norm, avg_cos=0.0, avg_score=0.0, used_cache=not need_recompute)
            return outputs, metrics

        weights = [s / sum_s for s in scores]
        outputs = {}
        for key in _keys(clipped_updates):
            stacked = torch.stack([u[key] * weights[i] for i, u in enumerate(clipped_updates)], dim=0)
            outputs[key] = stacked.sum(dim=0)

        elapsed = time.perf_counter() - start_time
        metrics = self._make_metrics(elapsed, r_norm=r_norm, avg_cos=sum(scores) / max(1.0, len(scores)),
                                     avg_scores=sum_s, used_cache=not need_recompute)
        return outputs, metrics