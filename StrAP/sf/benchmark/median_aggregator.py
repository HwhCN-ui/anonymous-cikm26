import torch
import time
from .__init__ import _to_device, _keys, _return_metrics
from typing import Dict, List, Tuple, Optional


class MedianAggregator:
    def __init__(self,device:torch.device):
        self.device = device
    
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]], 
                client_sizes: List[int],
                client_ids: Optional[List[int]] = None,
                is_init_benign: bool=False, 
                clean_updates: Optional[List[Dict[str, torch.Tensor]]]=None
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        start_time = time.perf_counter()
        client_updates = _to_device(client_updates, self.device)
        
        outputs = {}
        for key in _keys(client_updates):
            stacked = torch.stack([u[key] for u in client_updates], dim=0)
            outputs[key] = torch.median(stacked, dim=0).values
        end_time = time.perf_counter()
        return outputs, _return_metrics(end_time - start_time,"median")