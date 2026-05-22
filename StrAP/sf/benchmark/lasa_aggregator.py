import time
from typing import Dict, List, Optional, Tuple

import torch

from .__init__ import (
    _to_device,
    _keys,
    device as _default_device,
    _return_metrics,
)


class LASAAggregator:
    def __init__(
        self,
        device: torch.device = _default_device,
        lambda_m: float = 1.0,
        lambda_d: float = 1.0,
        keep_ratio: float = 0.7,
        eps: float = 1e-12,
    ):
        self.device = device
        self.lambda_m = float(lambda_m)
        self.lambda_d = float(lambda_d)
        self.keep_ratio = float(keep_ratio)
        self.eps = float(eps)

    def _pdp(self, x: torch.Tensor) -> float:
        signs = torch.sign(x)
        non_zeros = torch.sum(torch.abs(signs))
        if non_zeros < self.eps:
            return 0.5
        
        pdp_val = 0.5 * (1.0 + torch.sum(signs) / non_zeros)
        return float(pdp_val.item())

    def _mz_score(self, vals: List[float]) -> List[float]:
        if not vals:
            return []
        
        t_vals = torch.tensor(vals, dtype=torch.float32, device=self.device)
        med = torch.median(t_vals)
        std = torch.std(t_vals, unbiased=False)
        
        if std < self.eps:
            return [0.0 for _ in vals]
            
        scores = (t_vals - med) / (std + self.eps)
        return scores.tolist()

    def _sparsify(self, update: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
        if self.keep_ratio >= 1.0:
            return update
        
        flat_u = torch.cat([update[k].reshape(-1) for k in keys], dim=0)
        d = flat_u.numel()
        k = max(1, int(d * self.keep_ratio))
        
        _, topk_idx = torch.topk(torch.abs(flat_u), k)
        mask = torch.zeros_like(flat_u)
        mask[topk_idx] = 1.0
        
        sparsified_update = {}
        offset = 0
        for k_name in keys:
            shape = update[k_name].shape
            numel = update[k_name].numel()
            layer_mask = mask[offset : offset + numel].view(shape)
            sparsified_update[k_name] = update[k_name] * layer_mask
            offset += numel
            
        return sparsified_update

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        client_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        
        t_start = time.perf_counter()
        n = len(client_updates)
        
        if n == 0:
            return {}, _return_metrics(0.0, mode="lasa")

        updates = _to_device(client_updates=client_updates, device=self.device)
        keys = _keys(updates)
        
        sparsified_updates = []
        for i in range(n):
            sparsified_updates.append(self._sparsify(updates[i], keys))
            
        agg_update: Dict[str, torch.Tensor] = {}
        total_admitted_layers = 0
        
        for k_name in keys:
            layer_tensors = [su[k_name] for su in sparsified_updates]
            
            omega_l = []
            rho_l = []
            
            for i in range(n):
                layer_flat = layer_tensors[i].reshape(-1)
                omega_l.append(float(torch.norm(layer_flat).item()))
                rho_l.append(self._pdp(layer_flat))
                
            lambda_m_l = self._mz_score(omega_l)
            lambda_d_l = self._mz_score(rho_l)
            
            S_l = []
            for i in range(n):
                if abs(lambda_m_l[i]) <= self.lambda_m and abs(lambda_d_l[i]) <= self.lambda_d:
                    S_l.append(i)
                    
            if len(S_l) == 0:
                S_l = list(range(n))
                
            total_admitted_layers += len(S_l)
            
            stacked_admitted = torch.stack([layer_tensors[i] for i in S_l], dim=0)
            agg_update[k_name] = stacked_admitted.mean(dim=0)
            
        t_end = time.perf_counter()
        avg_admitted_per_layer = total_admitted_layers / max(1, len(keys))
        
        metrics = _return_metrics(t_end - t_start, mode="lasa")
        metrics.update({
            "name": "lasa",
            "n_total": int(n),
            "avg_admitted_per_layer": float(round(avg_admitted_per_layer, 2)),
            "keep_ratio": float(self.keep_ratio),
            "lambda_m": float(self.lambda_m),
            "lambda_d": float(self.lambda_d),
        })
        
        return agg_update, metrics