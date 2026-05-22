import time
import torch
from typing import Dict, List, Tuple, Optional


class FedAvgAggregator:
    def __init__(self, device):
        self.device = device

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        client_ids: Optional[List[int]] = None,
        
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        if len(client_updates) == 0:
            return {}, {"name": "fedavg", "total_time": 0.0, "total_samples": 0}

        if len(client_updates) != len(client_sizes):
            raise ValueError(
                f"client_updates ({len(client_updates)}) and client_sizes ({len(client_sizes)}) "
                f"must have the same length."
            )

        total_samples = sum(int(n) for n in client_sizes)
        if total_samples <= 0:
            keys = client_updates[0].keys()
            avg = {}
            t0 = time.perf_counter()
            for k in keys:
                stacked = torch.stack([u[k].to(self.device) for u in client_updates], dim=0)
                avg[k] = stacked.mean(dim=0)
            t1 = time.perf_counter()
            return avg, {
                "name": "fedavg",
                "total_time": t1 - t0,
                "total_samples": 0,
                "note": "total_samples <= 0, fall back to simple mean"
            }

        weights = [float(n) / float(total_samples) for n in client_sizes]

        t_start = time.perf_counter()
        keys = client_updates[0].keys()
        agg: Dict[str, torch.Tensor] = {}

        for k in keys:
            first = client_updates[0][k].to(self.device).float()
            agg_k = weights[0] * first

            for i in range(1, len(client_updates)):
                u_i = client_updates[i][k].to(self.device).float()
                agg_k = agg_k + weights[i] * u_i

            agg[k] = agg_k

        t_end = time.perf_counter()
        metrics = {
            "name": "fedavg",
            "total_time": t_end - t_start,
            "total_samples": total_samples,
            "client_sizes": [int(n) for n in client_sizes],
        }
        return agg, metrics
