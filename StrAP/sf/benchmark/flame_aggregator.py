import time
import math
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
        return list(updates[0].keys())

    def _flat(u: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = sorted(u.keys())
        return torch.cat([u[k].reshape(-1) for k in keys], dim=0)

    def _mean_updates(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not updates:
            return {}
        out = {}
        for k in _keys(updates):
            out[k] = torch.stack([u[k].float() for u in updates], dim=0).mean(dim=0)
        return out

    def _return_metrics(elapsed: float, mode: str = "flame") -> dict:
        return {"mode": mode, "total_time": float(elapsed)}


class FLAMEAggregator:

    def __init__(
        self,
        device: torch.device = device,
        min_cluster_ratio: float = 0.5,   
        dp_epsilon: float = 3705.0,
        dp_delta: float = 1e-5,
        enable_noising: bool = True,
        eps: float = 1e-12,
        prefer_hdbscan: bool = True,
    ):
        self.device = device
        self.min_cluster_ratio = float(min_cluster_ratio)
        self.dp_epsilon = float(dp_epsilon)
        self.dp_delta = float(dp_delta)
        self.enable_noising = bool(enable_noising)
        self.eps = float(eps)
        self.prefer_hdbscan = bool(prefer_hdbscan)

    def _cosine_distance_matrix(self, vecs: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(vecs, dim=1, keepdim=True).clamp_min(self.eps)
        u = vecs / norms
        sim = torch.mm(u, u.t()).clamp(-1.0, 1.0)
        dist = (1.0 - sim).clamp(0.0, 2.0)
        dist.fill_diagonal_(0.0)
        return dist

    def _connected_components(self, adj: np.ndarray) -> List[List[int]]:
        n = adj.shape[0]
        seen = np.zeros(n, dtype=bool)
        comps: List[List[int]] = []
        for i in range(n):
            if seen[i]:
                continue
            q = [i]
            seen[i] = True
            comp = [i]
            while q:
                v = q.pop()
                neigh = np.where(adj[v])[0]
                for w in neigh:
                    if not seen[w]:
                        seen[w] = True
                        q.append(int(w))
                        comp.append(int(w))
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        return comps

    def _cluster_hdbscan(self, dist_np: np.ndarray, min_cluster_size: int) -> Optional[List[int]]:
        try:
            import hdbscan 
        except Exception:
            return None

        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=int(min_cluster_size),
            min_samples=1,
            allow_single_cluster=True,
        )
        labels = np.asarray(clusterer.fit_predict(dist_np))  

        valid = labels[labels >= 0]
        if valid.size == 0:
            return None

        uniq, cnt = np.unique(valid, return_counts=True)
        best_label = int(uniq[np.argmax(cnt)])
        admitted = np.where(labels == best_label)[0].tolist()

        if len(admitted) < min_cluster_size:
            return None
        return admitted

    def _cluster_fallback_majority_component(self, dist_np: np.ndarray, min_cluster_size: int) -> List[int]:
        n = dist_np.shape[0]
        if n <= 1:
            return list(range(n))

        off = dist_np[np.triu_indices(n, 1)]
        if off.size == 0:
            return list(range(n))

        off = np.sort(off)
        qs = np.linspace(0.05, 0.95, num=25)
        eps_candidates = np.unique(np.quantile(off, qs))
        eps_candidates = np.clip(eps_candidates, 0.0, 2.0)

        best = list(range(n))
        best_len = n

        for eps0 in eps_candidates:
            adj = (dist_np <= eps0)
            np.fill_diagonal(adj, True)
            comps = self._connected_components(adj)
            if comps:
                if len(comps[0]) >= min_cluster_size:
                    return comps[0]

        return list(range(n))

    def _dynamic_filter(self, updates: List[Dict[str, torch.Tensor]]) -> Tuple[List[int], bool]:
        """
        Returns (admitted_indices, used_hdbscan)
        """
        n = len(updates)
        if n == 0:
            return [], False
        if n == 1:
            return [0], False

        min_cluster_size = int(math.floor(n * self.min_cluster_ratio)) + 1
        min_cluster_size = max(2, min(min_cluster_size, n))

        vecs = torch.stack([_flat(u).to(self.device).float() for u in updates], dim=0)
        dist = self._cosine_distance_matrix(vecs)
        dist_np = dist.detach().cpu().numpy()

        admitted: Optional[List[int]] = None
        used_hdbscan = False

        if self.prefer_hdbscan:
            admitted = self._cluster_hdbscan(dist_np, min_cluster_size=min_cluster_size)
            if admitted is not None and len(admitted) >= min_cluster_size:
                used_hdbscan = True

        if admitted is None:
            admitted = self._cluster_fallback_majority_component(dist_np, min_cluster_size=min_cluster_size)

        admitted = sorted(set(int(i) for i in admitted if 0 <= i < n))

        if len(admitted) < min_cluster_size:
            return list(range(n)), used_hdbscan

        return admitted, used_hdbscan
    def _compute_norms(self, updates: List[Dict[str, torch.Tensor]]) -> List[float]:
        norms: List[float] = []
        for u in updates:
            v = _flat(u).to(self.device).float()
            norms.append(float(torch.norm(v).item()))
        return norms

    def _adaptive_clipping_bound(self, norms: List[float]) -> float:
        if not norms:
            return 0.0
        return float(np.median(np.asarray(norms)))

    def _clip_update(self, u: Dict[str, torch.Tensor], St: float, e_i: float) -> Dict[str, torch.Tensor]:
        """
        Clip u by L2 norm to bound St.
        """
        if St <= 0.0:
            return {k: v.to(self.device).float() for k, v in u.items()}

        if e_i <= self.eps:
            scale = 1.0
        else:
            gamma = St / (e_i + self.eps)
            scale = float(min(1.0, gamma))

        return {k: (v.to(self.device).float() * scale) for k, v in u.items()}

    def _dp_lambda(self) -> float:
        if (not self.enable_noising) or self.dp_epsilon <= 0:
            return 0.0
        return float(math.sqrt(2.0 * math.log(1.25 / self.dp_delta)) / self.dp_epsilon)

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],  
        client_ids: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        start = time.perf_counter()

        if len(client_updates) == 0:
            return {}, {"name": "flame", "total_time": 0.0, "note": "empty updates"}

        updates = _to_device(client_updates=client_updates, device=self.device)

        admitted_idx, used_hdbscan = self._dynamic_filter(updates)
        if len(admitted_idx) == 0:
            admitted_idx = list(range(len(updates)))

        admitted_updates = [updates[i] for i in admitted_idx]

        all_norms = self._compute_norms(updates)
        St = self._adaptive_clipping_bound(all_norms)

        clipped: List[Dict[str, torch.Tensor]] = []
        for i in admitted_idx:
            u = updates[i]
            e_i = all_norms[i] if i < len(all_norms) else float(torch.norm(_flat(u)).item())
            clipped.append(self._clip_update(u, St=St, e_i=e_i))

        use_sizes = (
            isinstance(client_sizes, list)
            and len(client_sizes) == len(updates)
            and sum(int(s) for s in client_sizes) > 0
        )

        if len(clipped) == 0:
            agg = _mean_updates(updates)
            agg_weight_sum = float(sum(client_sizes)) if use_sizes else float(len(updates))
        else:
            agg: Dict[str, torch.Tensor] = {}
            if use_sizes:
                w = np.asarray([max(1, int(client_sizes[i])) for i in admitted_idx], dtype=np.float64)
                w_sum = float(w.sum())
                w = w / max(w_sum, 1e-12)
                for k in _keys(clipped):
                    stacked = torch.stack([u[k] for u in clipped], dim=0)  
                    weights = torch.from_numpy(w).to(stacked.device, stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
                    agg[k] = (stacked * weights).sum(dim=0)
                agg_weight_sum = w_sum
            else:
                for k in _keys(clipped):
                    stacked = torch.stack([u[k] for u in clipped], dim=0)
                    agg[k] = stacked.mean(dim=0)
                agg_weight_sum = float(len(clipped))

        sigma = 0.0
        if self.enable_noising and St > 0.0:
            lam = self._dp_lambda()
            sigma = float(lam * St)
            if sigma > 0.0:
                for k in list(agg.keys()):
                    agg[k] = agg[k] + torch.randn_like(agg[k]) * sigma

        elapsed = time.perf_counter() - start
        metrics = _return_metrics(elapsed, mode="flame")
        metrics.update({
            "name": "flame",
            "used_hdbscan": bool(used_hdbscan),
            "n_clients": int(len(updates)),
            "n_admitted": int(len(admitted_idx)),
            "n_rejected": int(len(updates) - len(admitted_idx)),
            "min_cluster_ratio": float(self.min_cluster_ratio),
            "St": float(St),
            "sigma": float(sigma),
            "dp_epsilon": float(self.dp_epsilon),
            "dp_delta": float(self.dp_delta),
            "size_weighted": bool(use_sizes),
            "agg_weight_sum": float(agg_weight_sum),
        })
        return agg, metrics