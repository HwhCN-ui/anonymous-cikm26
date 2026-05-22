import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

try:
    from .__init__ import _to_device, _keys, _flat, _mean_updates, device, _return_metrics
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_device(client_updates: List[Dict[str, torch.Tensor]], device: torch.device):
        out = []
        for u in client_updates:
            out.append({k: v.detach().to(device) for k, v in u.items()})
        return out

    def _keys(updates: List[Dict[str, torch.Tensor]]):
        if not updates:
            return []
        return sorted(updates[0].keys())

    def _flat(u: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = _keys([u])
        return torch.cat([u[k].reshape(-1) for k in keys], dim=0)

    def _mean_updates(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not updates:
            return {}
        out = {}
        for k in _keys(updates):
            out[k] = torch.stack([u[k].float() for u in updates], dim=0).mean(dim=0)
        return out

    def _return_metrics(elapsed: float, mode: str = "feddlad") -> dict:
        return {"name": mode, "total_time": float(elapsed)}


class FedDLADAggregator:
    def __init__(
        self,
        device: torch.device = device,
        iqr_multiplier: float = 1.5,
        pardon_threshold: float = 0.8,
        eps: float = 1e-12
    ):
        self.device = device
        self.iqr_multiplier = float(iqr_multiplier)
        self.pardon_threshold = float(pardon_threshold)
        self.eps = float(eps)

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        client_ids: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        
        t_start = time.perf_counter()
        n_clients = len(client_updates)
        
        if n_clients == 0:
            return {}, _return_metrics(0.0, mode="feddlad")
            
        if n_clients <= 3:
            return _mean_updates(client_updates), _return_metrics(time.perf_counter() - t_start, mode="feddlad")

        updates = _to_device(client_updates=client_updates, device=self.device)
        keys = _keys(updates)

        V = torch.stack([_flat(u).float() for u in updates], dim=0)
        
        sq_norms = torch.sum(V ** 2, dim=1, keepdim=True)
        dist_matrix_sq = sq_norms + sq_norms.T - 2 * torch.matmul(V, V.T)
        dist_matrix = torch.sqrt(dist_matrix_sq.clamp_min(self.eps))
        
        dist_scores = torch.sum(dist_matrix, dim=1) / (n_clients - 1)
        
        q25 = torch.quantile(dist_scores, 0.25)
        q75 = torch.quantile(dist_scores, 0.75)
        iqr = q75 - q25
        
        upper_bound = q75 + self.iqr_multiplier * iqr
        
        trusted_idx = []
        suspicious_idx = []
        
        for i in range(n_clients):
            if dist_scores[i].item() <= upper_bound.item():
                trusted_idx.append(i)
            else:
                suspicious_idx.append(i)
                
        if len(trusted_idx) == 0:
            trusted_idx = list(range(n_clients))
            suspicious_idx = []

        pardoned_idx = []
        
        if len(suspicious_idx) > 0 and len(trusted_idx) > 0:
            trusted_V = V[trusted_idx]
            trusted_mean_vec = torch.mean(trusted_V, dim=0)
            norm_trusted = torch.norm(trusted_mean_vec).clamp_min(self.eps)
            
            for i in suspicious_idx:
                u_vec = V[i]
                norm_u = torch.norm(u_vec).clamp_min(self.eps)
                
                cos_sim = torch.dot(u_vec, trusted_mean_vec) / (norm_u * norm_trusted)
                
                if cos_sim.item() >= self.pardon_threshold:
                    pardoned_idx.append(i)
                    
        final_admitted_idx = trusted_idx + pardoned_idx
        
        agg = {}
        use_sizes = (
            isinstance(client_sizes, list)
            and len(client_sizes) == n_clients
            and sum(int(s) for s in client_sizes) > 0
        )

        if use_sizes:
            w = np.asarray([max(1, int(client_sizes[i])) for i in final_admitted_idx], dtype=np.float64)
            w_sum = float(w.sum())
            w = w / max(w_sum, self.eps)
            
            for k_name in keys:
                stacked = torch.stack([updates[i][k_name].float() for i in final_admitted_idx], dim=0)
                weights_t = torch.from_numpy(w).to(stacked.device, stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
                agg[k_name] = (stacked * weights_t).sum(dim=0)
            agg_weight_sum = w_sum
        else:
            for k_name in keys:
                stacked = torch.stack([updates[i][k_name].float() for i in final_admitted_idx], dim=0)
                agg[k_name] = stacked.mean(dim=0)
            agg_weight_sum = float(len(final_admitted_idx))

        t_end = time.perf_counter()
        
        metrics = _return_metrics(t_end - t_start, mode="feddlad")
        metrics.update({
            "n_total": n_clients,
            "n_trusted_init": len(trusted_idx),
            "n_suspicious": len(suspicious_idx),
            "n_pardoned": len(pardoned_idx),
            "n_final_admitted": len(final_admitted_idx),
            "iqr_upper_bound": float(upper_bound.item()),
            "size_weighted": use_sizes,
            "agg_weight_sum": float(agg_weight_sum),
        })

        return agg, metrics