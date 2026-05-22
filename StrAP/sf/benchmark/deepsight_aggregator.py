import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sf.models import get_model
from sf.data import get_dataset
from sf.utils.lora_utils import apply_lora_update
from sf.config.base_config import federated as fl_cfg

from .__init__ import (
    _to_device,
    _keys,
    _flat,
    device as _default_device,
    _return_metrics,
)


class DeepsightAggregator:
    def __init__(
        self,
        device: torch.device = _default_device,
        model_name: Optional[str] = None,
        data_name: Optional[str] = None,
        federated_data_path: Optional[str] = None,
        global_lora_path: Optional[str] = None,
        probe_batch_size: int = 8,
        max_probe_batches: int = 10,
        feature_neup_eps: float = 1e-12,
        feature_ddif_weight: float = 1.0,
        feature_neup_weight: float = 1.0,
        kmeans_iters: int = 30,
        eps: float = 1e-12,
    ):
        self.device = device
        self.model_name = model_name
        self.data_name = data_name
        self.federated_data_path = federated_data_path
        self.global_lora_path = global_lora_path

        self.probe_batch_size = int(probe_batch_size)
        self.max_probe_batches = int(max_probe_batches)
        self.feature_neup_eps = float(feature_neup_eps)

        self.feature_ddif_weight = float(feature_ddif_weight)
        self.feature_neup_weight = float(feature_neup_weight)

        self.kmeans_iters = int(kmeans_iters)
        self.eps = float(eps)

    def _neup_feature(self, update: Dict[str, torch.Tensor]) -> np.ndarray:
        keys = list(update.keys())
        norms = []
        for k in keys:
            v = update[k].detach()
            if not torch.is_tensor(v):
                norms.append(0.0)
                continue
            vv = v.to(self.device).float().reshape(-1)
            norms.append(float(torch.norm(vv, p=2).item()))

        arr = np.asarray(norms, dtype=np.float64)
        s = float(arr.sum()) + self.feature_neup_eps
        arr = arr / s
        return arr

    @torch.no_grad()
    def _build_probe_loader(self, federated_data_path: str):
        ds = get_dataset(
            federated_data_path,
            model_name=self.model_name,
            client_id=0,
            data_name=self.data_name,
            is_server=True,
        )
        loader = ds.get_loader(batch_size=self.probe_batch_size, shuffle=False)
        return loader

    @staticmethod
    def _extract_logits(output):
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
        if isinstance(output, (tuple, list)) and len(output) > 0:
            return output[0]
        if hasattr(output, "logits"):
            return output.logits
        raise RuntimeError("Cannot extract logits from model output.")

    @torch.no_grad()
    def _compute_ddifs(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        global_lora_path: str,
        federated_data_path: str,
    ) -> List[float]:
        model_base = get_model(model_name=self.model_name, global_lora_path=global_lora_path)
        model_base.to(self.device)
        model_base.eval()

        loader = self._build_probe_loader(federated_data_path)

        base_probs: List[torch.Tensor] = []
        batches = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device)
            out = model_base(pixel_values=pixel_values)
            logits = self._extract_logits(out)
            probs = torch.softmax(logits.float(), dim=-1)
            base_probs.append(probs.detach())
            batches += 1
            if batches >= self.max_probe_batches:
                break

        if len(base_probs) == 0:
            return [0.0 for _ in client_updates]

        ddifs: List[float] = []
        for upd in client_updates:
            model_after = get_model(model_name=self.model_name, global_lora_path=global_lora_path)
            model_after.to(self.device)
            model_after.eval()
            model_after = apply_lora_update(model_after, upd)

            diffs = []
            for i, batch in enumerate(loader):
                pixel_values = batch["pixel_values"].to(self.device)
                out2 = model_after(pixel_values=pixel_values)
                logits2 = self._extract_logits(out2)
                probs2 = torch.softmax(logits2.float(), dim=-1)

                p1 = base_probs[i]
                d = torch.norm((probs2 - p1).reshape(probs2.size(0), -1), p=2, dim=1).mean()
                diffs.append(float(d.item()))

                if (i + 1) >= self.max_probe_batches:
                    break

            ddif = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
            ddifs.append(ddif)

        return ddifs

    @staticmethod
    def _kmeans2(X: np.ndarray, iters: int = 30, seed: int = 0) -> np.ndarray:
        n, d = X.shape
        if n <= 1:
            return np.zeros((n,), dtype=np.int64)

        rng = np.random.default_rng(seed)
        i0 = int(rng.integers(0, n))
        c0 = X[i0]

        dist0 = np.sum((X - c0) ** 2, axis=1)
        i1 = int(np.argmax(dist0))
        c1 = X[i1]

        labels = np.zeros((n,), dtype=np.int64)
        for _ in range(max(1, iters)):
            d0 = np.sum((X - c0) ** 2, axis=1)
            d1 = np.sum((X - c1) ** 2, axis=1)
            new_labels = (d1 < d0).astype(np.int64)

            if np.all(new_labels == labels):
                labels = new_labels
                break
            labels = new_labels

            if np.any(labels == 0):
                c0 = X[labels == 0].mean(axis=0)
            if np.any(labels == 1):
                c1 = X[labels == 1].mean(axis=0)

        return labels

    def _weighted_fedavg(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        if len(updates) == 0:
            return {}

        total = sum(int(s) for s in sizes)
        if total <= 0:
            out = {}
            for k in updates[0].keys():
                stacked = torch.stack([u[k].to(self.device).float() for u in updates], dim=0)
                out[k] = stacked.mean(dim=0)
            return out

        weights = [float(s) / float(total) for s in sizes]
        out = {}
        for k in updates[0].keys():
            acc = None
            for i, u in enumerate(updates):
                v = u[k].to(self.device).float() * weights[i]
                acc = v if acc is None else (acc + v)
            out[k] = acc
        return out

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        client_ids: Optional[List[int]] = None,
        current_round: Optional[int] = None,
        global_lora_path: Optional[str] = None,
        federated_data_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        t0 = time.perf_counter()

        if len(client_updates) == 0:
            return {}, {"name": "deepsight", "total_time": 0.0, "kept": 0, "total": 0}

        if len(client_updates) != len(client_sizes):
            raise ValueError(
                f"client_updates ({len(client_updates)}) and client_sizes ({len(client_sizes)}) must have same length."
            )

        client_updates = _to_device(client_updates=client_updates, device=self.device)

        glp = global_lora_path or self.global_lora_path
        fdp = federated_data_path or self.federated_data_path

        neups = [self._neup_feature(u) for u in client_updates]
        neup_dim = int(neups[0].shape[0]) if len(neups) > 0 else 0

        ddif_used = False
        if glp is not None and fdp is not None and self.model_name is not None and self.data_name is not None:
            try:
                ddifs = self._compute_ddifs(
                    client_updates=client_updates,
                    global_lora_path=glp,
                    federated_data_path=fdp,
                )
                ddif_used = True
            except Exception as e:
                ddifs = [0.0 for _ in client_updates]
                ddif_used = False
                err = repr(e)
        else:
            ddifs = [0.0 for _ in client_updates]
            ddif_used = False
            err = "missing(model_name/data_name/global_lora_path/federated_data_path)"

        ddifs_arr = np.asarray(ddifs, dtype=np.float64).reshape(-1, 1)
        neups_arr = np.stack(neups, axis=0) if len(neups) > 0 else np.zeros((len(client_updates), 0), dtype=np.float64)
        X = np.concatenate(
            [self.feature_ddif_weight * ddifs_arr, self.feature_neup_weight * neups_arr],
            axis=1
        )
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + self.eps
        Xs = (X - mu) / sigma

        if Xs.shape[0] < 2:
            labels = np.zeros((Xs.shape[0],), dtype=np.int64)
        else:
            labels = self._kmeans2(Xs, iters=self.kmeans_iters, seed=int(fl_cfg.get("seed", 0)))

        idx0 = np.where(labels == 0)[0].tolist()
        idx1 = np.where(labels == 1)[0].tolist()

        def _mean_ddif(idxs: List[int]) -> float:
            if len(idxs) == 0:
                return float("inf")
            return float(np.mean([ddifs[i] for i in idxs]))

        c0, c1 = len(idx0), len(idx1)
        if c0 == 0 and c1 == 0:
            kept_idx = list(range(len(client_updates)))
        elif c0 == 0:
            kept_idx = idx1
        elif c1 == 0:
            kept_idx = idx0
        else:
            if c0 > c1:
                kept_idx = idx0
            elif c1 > c0:
                kept_idx = idx1
            else:
                kept_idx = idx0 if _mean_ddif(idx0) <= _mean_ddif(idx1) else idx1

        kept_updates = [client_updates[i] for i in kept_idx]
        kept_sizes = [client_sizes[i] for i in kept_idx]

        agg = self._weighted_fedavg(kept_updates, kept_sizes)

        t1 = time.perf_counter()
        metrics = _return_metrics(t1 - t0, mode="deepsight")
        metrics.update({
            "name": "deepsight",
            "round": int(current_round) if current_round is not None else None,
            "total": int(len(client_updates)),
            "kept": int(len(kept_idx)),
            "kept_ratio": float(len(kept_idx) / max(1, len(client_updates))),
            "cluster0": int(c0),
            "cluster1": int(c1),
            "ddif_used": bool(ddif_used),
            "ddif_mean_all": float(np.mean(ddifs)) if len(ddifs) > 0 else 0.0,
            "ddif_mean_kept": float(np.mean([ddifs[i] for i in kept_idx])) if len(kept_idx) > 0 else 0.0,
            "neup_dim": int(neup_dim),
        })
        if not ddif_used:
            metrics["ddif_note"] = err

        return agg, metrics