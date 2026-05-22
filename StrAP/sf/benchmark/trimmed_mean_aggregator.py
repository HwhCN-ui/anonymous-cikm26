import torch
import time
from typing import DefaultDict, Dict,List, Optional
from .__init__ import _to_device, _keys, _return_metrics,device



class TrimmedMeanAggregator:
    def __init__(self, trim_k: int=3, device: torch.device = device):
        self.trim_k = trim_k
        self.device = device

    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]],
                  client_sizes: List[int],
                  client_ids: Optional[List[int]] = None,
                  is_init_benign=False,
                  clean_updates = None
                  ):
        start_time = time.perf_counter()
        client_updates = _to_device(client_updates, self.device)

        outputs = {}
        num_clients = len(client_updates)
        self.trim_k = min(self.trim_k, max((num_clients - 1)//2, 0))
        for key in _keys(client_updates):
            x = torch.stack([u[key] for u in client_updates], dim=0)
            x_sorted, _ = torch.sort(x, dim=0)
            kept = x_sorted[self.trim_k:num_clients-self.trim_k,...] if self.trim_k > 0 else  x_sorted
            outputs[key] = kept.mean(dim=0)
        end_time = time.perf_counter()
        return outputs, _return_metrics(end_time - start_time,"trimmed")

            
            
            

