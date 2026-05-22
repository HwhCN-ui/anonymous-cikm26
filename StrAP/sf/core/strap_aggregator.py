import time
import math
import random
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from sf.config.strap_config import strap as sf_cfg
from sf.config.base_config import train as tr_cfg
from sf.benchmark.fedavg_aggregator import FedAvgAggregator
from sf.data import get_dataset
from sf.utils.lora_utils import iter_lora_pairs


class StrAPAggregator:

    def __init__(self, device=None, model_name="vit", data_name=None,
                 federated_val_data_path=None,
                 federated_probe_data_path=None,
                 base_aggregator=None,
                 strap_max_batches=None,
                 cce_gamma=None,
                 cce_trust_beta=None,
                 phi=None,
                 server_ref_label_mode: str = "all",
                 server_ref_labels: str = "",
                 server_ref_random_k: int = 0,
                 server_ref_random_seed: int = 42):

        self.device = device or torch.device(tr_cfg.get("device", "cuda:0"))
        self.model_name = model_name
        self.data_name = data_name
        if data_name is None:
            raise ValueError("data_name must be provided")
        
        self.cce_max_batches = strap_max_batches if strap_max_batches is not None else sf_cfg.get("cce_max_batches", 8)

        self.cce_gamma = cce_gamma
        self.cce_trust_beta = cce_trust_beta
        self.phi = phi

        self.val_path = federated_val_data_path
        self.probe_path = federated_probe_data_path or federated_val_data_path
        self.fedavg_aggregator = FedAvgAggregator(device=device)
        self.base_aggregator = base_aggregator
        if base_aggregator is None:
            raise ValueError("base_aggregator must be provided")

        self.freeze_counters = defaultdict(int)

        self.init_state: Dict[str, torch.Tensor] = None
        self.last_metrics: Dict[str, float] = {}

        self.client_trust = defaultdict(lambda: 1.0)
        self.client_ban_score = defaultdict(lambda: 0.0)
        self._current_round = 0

        self._ema_update_norm = None
        self._ema_update_norm_init = None

        self._ema_probe_loss = None
        self._ema_probe_delta = None
        self._prev_ema_probe_loss = None

        self._converged_patience = 0
        self._is_converged = False

        self.server_ref_label_mode = (server_ref_label_mode or "all").lower().strip()
        self.server_ref_labels_raw = server_ref_labels or ""
        self.server_ref_random_k = int(server_ref_random_k or 0)
        self.server_ref_random_seed = int(server_ref_random_seed or 42)
        self._cached_reference_label_set: Optional[Set[int]] = None

    def _to_cpu_tensor(self, t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu().clone()

    def _to_cpu_dict(self, d: Dict) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in d.items():
            if v is None:
                continue
            if torch.is_tensor(v):
                out[k] = self._to_cpu_tensor(v)
        return out

    def _summary_stats(self, values: List[float]):
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        n = float(len(values))
        vmin = min(values)
        vmax = max(values)
        mean = sum(values) / n
        var = sum((x - mean) ** 2 for x in values) / n
        std = math.sqrt(var)
        return float(mean), float(std), float(vmin), float(vmax)

    def _quantile(self, xs: List[float], q: float) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        idx = int(q * (len(xs_sorted) - 1))
        idx = max(0, min(idx, len(xs_sorted) - 1))
        return float(xs_sorted[idx])

    def _extract_labels_from_batch(self, batch: Dict) -> torch.Tensor:
        label = None
        for k in ("labels", "label", "targets", "target", "y"):
            if k in batch and batch[k] is not None:
                label = batch[k]
                break
        if label is None:
            raise KeyError(f"Cannot find labels in batch keys: {list(batch.keys())}")

        t = label if isinstance(label, torch.Tensor) else torch.as_tensor(label)
        if t.dim() > 1 and t.size(-1) == 1:
            t = t.view(-1)
        if t.dtype != torch.long:
            if t.is_floating_point() and t.dim() > 1 and t.size(-1) > 1:
                t = t.argmax(dim=-1)
            t = t.long()
        return t

    def _get_lora_param_tensors(self, model) -> Dict[str, torch.Tensor]:
        return {k: p for k, p in model.lora_model.named_parameters()}

    def _flat_cpu(self, d: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for k in sorted(d.keys()):
            v = d[k]
            if v is None:
                continue
            parts.append(v.detach().float().flatten())
        if not parts:
            return torch.zeros(1)
        return torch.cat(parts, dim=0)

    def _get_client_ids(self, client_updates: List[Dict]) -> List[str]:
        ids = []
        for i, upd in enumerate(client_updates):
            cid = None
            if isinstance(upd, dict):
                for k in ("client_id", "__client_id__", "cid", "id"):
                    if k in upd and upd[k] is not None and not torch.is_tensor(upd[k]):
                        cid = str(upd[k])
                        break
            ids.append(cid if cid is not None else f"idx_{i}")
        return ids

    def _build_hdr_stats(self,
                         client_ids: List[str],
                         weights_final: List[float],
                         benign_client_ids: Optional[List[int]]) -> Dict[str, float]:
        if not client_ids or not weights_final or not benign_client_ids:
            return {}

        benign_set = set(str(x) for x in benign_client_ids)
        benign_weights = []

        for cid, w in zip(client_ids, weights_final):
            if str(cid) in benign_set:
                benign_weights.append(float(w))

        if not benign_weights:
            return {}

        n = len(benign_weights)
        stats = {
            "hdr_benign_count": float(n),
            "hdr_benign_weight_mean": float(sum(benign_weights) / n),
            "hdr_benign_weight_min": float(min(benign_weights)),
            "hdr_benign_weight_max": float(max(benign_weights)),
            "hdr_benign_mean_suppression": float(sum(1.0 - w for w in benign_weights) / n),
        }

        thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for th in thresholds:
            cnt = sum(1 for w in benign_weights if w <= th)
            key = f"{th:.1f}".replace(".", "p")
            stats[f"hdr_benign_le_{key}_count"] = float(cnt)
            stats[f"hdr_benign_le_{key}_ratio"] = float(cnt / n)

        return stats

    def _parse_label_csv(self, s: str) -> List[int]:
        if s is None:
            return []
        items = []
        for part in str(s).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                items.append(int(part))
            except Exception:
                continue
        return items

    def _get_num_classes(self) -> int:
        name = str(self.data_name).lower().strip()
        if name == "tiny-imagenet":
            return 200
        elif name == "cifar-100":
            return 100
        return 0

    def _resolve_reference_label_set(self) -> Optional[Set[int]]:
        if self.server_ref_label_mode == "all":
            return None

        if self._cached_reference_label_set is not None:
            return self._cached_reference_label_set

        explicit = set(self._parse_label_csv(self.server_ref_labels_raw))
        num_classes = self._get_num_classes()

        if self.server_ref_label_mode in ("keep", "drop"):
            if num_classes > 0:
                active = set([x for x in explicit if 0 <= x < num_classes])
            else:
                active = set(explicit)
        elif self.server_ref_label_mode == "random_k":
            if num_classes <= 0:
                active = set()
            else:
                k = int(self.server_ref_random_k)
                if k <= 0 or k >= num_classes:
                    active = set(range(num_classes))
                else:
                    rng = random.Random(self.server_ref_random_seed)
                    active = set(rng.sample(list(range(num_classes)), k))
        else:
            active = set(range(num_classes)) if num_classes > 0 else set()

        self._cached_reference_label_set = active
        return self._cached_reference_label_set

    def _get_ref_filter_mode_id(self) -> float:
        mode = self.server_ref_label_mode
        if mode == "all":
            return 0.0
        if mode == "keep":
            return 1.0
        if mode == "drop":
            return 2.0
        if mode == "random_k":
            return 3.0
        return 0.0

    def _build_reference_stats(self) -> Dict[str, float]:
        mode = self.server_ref_label_mode
        num_classes = self._get_num_classes()
        ref_set = self._resolve_reference_label_set()

        if mode == "all" or ref_set is None:
            active_num_labels = float(num_classes)
        else:
            active_num_labels = float(len(ref_set))

        stats = {
            "ref_active_num_labels": float(active_num_labels),
            "ref_filter_mode_id": float(self._get_ref_filter_mode_id()),
            "ref_probe_kept_ratio": 1.0 if mode == "all" else 0.0,
        }

        if not self.probe_path:
            return stats

        if mode == "all" or ref_set is None:
            stats["ref_probe_kept_ratio"] = 1.0
            return stats

        ds_probe = get_dataset(
            self.probe_path,
            self.model_name,
            client_id=0,
            data_name=self.data_name,
            is_server=True
        )
        probe_loader = ds_probe.get_loader(
            batch_size=int(sf_cfg.get("batch_size", 64)),
            shuffle=False
        )

        total_seen = 0
        total_kept = 0
        cnt = 0

        for batch in probe_loader:
            y = self._extract_labels_from_batch(batch).detach().cpu()
            total_seen += int(y.numel())

            if mode in ("keep", "random_k"):
                kept = sum(1 for v in y.tolist() if int(v) in ref_set)
            elif mode == "drop":
                kept = sum(1 for v in y.tolist() if int(v) not in ref_set)
            else:
                kept = int(y.numel())

            total_kept += int(kept)

            cnt += 1
            if cnt >= self.cce_max_batches:
                break

        stats["ref_probe_kept_ratio"] = float(total_kept / max(1, total_seen))
        return stats

    def _compute_lora_grads_and_probe_loss_cpu(self, model, loader, max_batches: int) -> Tuple[Dict[str, torch.Tensor], float]:
        model.train(False)
        model.to(self.device)

        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.lora_model.named_parameters():
            if ("lora_A" in n) or ("lora_B" in n):
                p.requires_grad = True

        grad_sums: Dict[str, torch.Tensor] = {}
        loss_sum = 0.0
        cnt = 0

        for batch in loader:
            x = batch["pixel_values"].to(self.device)
            y = self._extract_labels_from_batch(batch).to(self.device)

            model.zero_grad(set_to_none=True)
            out = model(pixel_values=x, labels=y, return_loss=True)
            if not isinstance(out, dict) or out.get("loss", None) is None:
                raise TypeError("Model forward must return dict with non-null 'loss' when return_loss=True.")
            loss = out["loss"]
            loss_sum += float(loss.detach().item())
            loss.backward()

            for name, p in model.lora_model.named_parameters():
                if (("lora_A" not in name) and ("lora_B" not in name)) or (p.grad is None):
                    continue
                g = p.grad.detach().float()
                grad_sums[name] = g.clone() if name not in grad_sums else (grad_sums[name] + g)

            cnt += 1
            if cnt >= max_batches:
                break

        print(f"Probe processed {cnt} batches for LoRA grads.")
        if cnt > 0:
            for k in list(grad_sums.keys()):
                grad_sums[k] = (grad_sums[k] / float(cnt)).detach().cpu()

        model.to("cpu")
        probe_loss = (loss_sum / float(cnt)) if cnt > 0 else 0.0
        return grad_sums, float(probe_loss)

    def _update_convergence(self, mean_update_norm: float, probe_loss: Optional[float]) -> Dict[str, float]:
        if self._ema_update_norm is None:
            self._ema_update_norm = float(mean_update_norm)
        else:
            self._ema_update_norm = float(0.80 * self._ema_update_norm + 0.20 * float(mean_update_norm))

        if self._current_round == 3 and self._ema_update_norm_init is None:
            self._ema_update_norm_init = float(self._ema_update_norm)
        if self._ema_update_norm_init is None:
            self._ema_update_norm_init = float(self._ema_update_norm)

        ratio = float(self._ema_update_norm) / max(1e-12, float(self._ema_update_norm_init))

        probe_ok = False
        if probe_loss is not None:
            if self._ema_probe_loss is None:
                self._ema_probe_loss = float(probe_loss)
            else:
                self._ema_probe_loss = float(0.90 * self._ema_probe_loss + 0.10 * float(probe_loss))

            if self._prev_ema_probe_loss is None:
                delta = 0.0
            else:
                delta = float(self._prev_ema_probe_loss) - float(self._ema_probe_loss)
            self._prev_ema_probe_loss = float(self._ema_probe_loss)

            if self._ema_probe_delta is None:
                self._ema_probe_delta = float(delta)
            else:
                self._ema_probe_delta = float(0.90 * self._ema_probe_delta + 0.10 * float(delta))

            probe_ok = (abs(float(self._ema_probe_delta)) <= 5e-4)

        if self._current_round <= 3:
            self._is_converged = False
            self._converged_patience = 0
        else:
            if bool((ratio <= 0.25) or probe_ok):
                self._converged_patience += 1
            else:
                self._converged_patience = 0

            self._is_converged = bool(self._converged_patience >= 4)

        return {
            "conv_ratio": float(ratio),
            "conv_ratio_th": 0.25,
            "conv_probe_loss_ema": float(self._ema_probe_loss) if self._ema_probe_loss is not None else 0.0,
            "conv_probe_delta_ema": float(self._ema_probe_delta) if self._ema_probe_delta is not None else 0.0,
            "conv_is_converged": float(1.0 if self._is_converged else 0.0),
            "conv_patience_cnt": float(self._converged_patience),
        }

    def _cce_progressive_gate(
        self,
        client_updates_cpu: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        g_clean_cpu: Dict[str, torch.Tensor],
        client_ids: List[str],
        benign_client_ids: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, float], List[float], Dict[str, float]]:

        if (not client_updates_cpu) or (not g_clean_cpu):
            ones = [1.0 for _ in client_updates_cpu]
            hdr_stats = self._build_hdr_stats(client_ids, ones, benign_client_ids)
            return client_updates_cpu, {"cce_used": 0.0}, ones, hdr_stats

        g = self._flat_cpu(g_clean_cpu)
        ng = float(g.norm(p=2).item())
        if ng <= 1e-12:
            ones = [1.0 for _ in client_updates_cpu]
            hdr_stats = self._build_hdr_stats(client_ids, ones, benign_client_ids)
            return client_updates_cpu, {"cce_used": 0.0, "cce_ng": float(ng)}, ones, hdr_stats

        effs, norms = [], []
        for upd in client_updates_cpu:
            u = self._flat_cpu(upd)
            nu = float(u.norm(p=2).item())
            norms.append(nu)
            if nu <= 1e-12:
                effs.append(0.0)
                continue
            effs.append(float((-float(torch.dot(g, u).item())) / (nu + 1e-12)))

        effs_sorted = sorted(effs)
        med = effs_sorted[len(effs_sorted) // 2]
        abs_dev = sorted([abs(x - med) for x in effs])
        mad = float(abs_dev[len(abs_dev) // 2]) + 1e-12

        weights_round, weights_final, z_list, trust_list = [], [], [], []
        ban_mult_list = []

        g_val = float(self.cce_gamma if self.cce_gamma is not None else 3.0)
        t_beta = float(self.cce_trust_beta if self.cce_trust_beta is not None else 0.85)

        for cid, e in zip(client_ids, effs):
            z = (med - e) / mad
            z_list.append(float(z))

            if z <= (0.5 if self._is_converged else 0.9):
                w_round = 1.0
            else:
                w_round = math.exp(-(g_val * 4.0 if self._is_converged else g_val) * (z - (0.5 if self._is_converged else 0.9)))
            w_round = max(0.01, min(1.0, float(w_round)))
            weights_round.append(w_round)

            prev_trust = float(self.client_trust[cid])
            new_trust = t_beta * prev_trust + (1.0 - t_beta) * float(w_round)
            new_trust = max(0.0, min(1.0, new_trust))
            self.client_trust[cid] = new_trust
            trust_list.append(new_trust)

            prev_bs = float(self.client_ban_score[cid])
            bs = 0.95 * prev_bs + (1.0 if z >= (0.8 if self._is_converged else 1.8) else 0.0)
            self.client_ban_score[cid] = float(bs)

            ban_mult = math.exp(-(10.0 if self._is_converged else 4.0) * float(bs))
            ban_mult = max(1e-4, min(1.0, float(ban_mult)))
            ban_mult_list.append(ban_mult)

            w_final = max(0.0, min(1.0, float(w_round) * (float(new_trust) ** 1.6) * float(ban_mult)))
            weights_final.append(w_final)

        scaled = []
        for upd, w in zip(client_updates_cpu, weights_final):
            if w >= 0.999:
                scaled.append(upd)
            else:
                scaled.append({k: (v * w).detach() for k, v in upd.items()})

        stats = {
            "cce_used": 1.0,
            "cce_converged": float(1.0 if self._is_converged else 0.0),
            "cce_med": float(med),
            "cce_mad": float(mad),
            "cce_eff_mean": float(sum(effs) / max(1, len(effs))),
            "cce_z_mean": float(sum(z_list) / max(1, len(z_list))),
            "cce_w_round_mean": float(sum(weights_round) / max(1, len(weights_round))),
            "cce_w_final_mean": float(sum(weights_final) / max(1, len(weights_final))),
            "cce_w_final_min": float(min(weights_final) if weights_final else 1.0),
            "cce_trust_mean": float(sum(trust_list) / max(1, len(trust_list))) if trust_list else 1.0,
            "cce_trust_min": float(min(trust_list)) if trust_list else 1.0,
            "cce_ban_mult_mean": float(sum(ban_mult_list) / max(1, len(ban_mult_list))) if ban_mult_list else 1.0,
            "cce_ban_mult_min": float(min(ban_mult_list)) if ban_mult_list else 1.0,
            "cce_mean_update_norm": float(sum(norms) / max(1, len(norms))),
        }

        hdr_stats = self._build_hdr_stats(client_ids, weights_final, benign_client_ids)
        return scaled, stats, weights_final, hdr_stats

    def _clean_gradient_projection(
        self,
        avg_update_cpu: Dict[str, torch.Tensor],
        g_clean_cpu: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:

        if (not avg_update_cpu) or (not g_clean_cpu):
            return avg_update_cpu, {"cgp_used": 0.0}

        dot = 0.0
        gg = 0.0
        keys = set(avg_update_cpu.keys()).intersection(set(g_clean_cpu.keys()))
        if not keys:
            return avg_update_cpu, {"cgp_used": 0.0}

        for k in keys:
            u = avg_update_cpu[k].detach().float()
            g = g_clean_cpu[k].detach().float()
            dot += float((u * g).sum().item())
            gg += float((g * g).sum().item())

        if gg <= 1e-12:
            return avg_update_cpu, {"cgp_used": 0.0, "cgp_gg": float(gg)}

        if dot <= 0.0:
            return avg_update_cpu, {"cgp_used": 1.0, "cgp_dot_before": float(dot), "cgp_alpha": 0.0}

        alpha = max(0.0, min(10.0, dot / (gg + 1e-12)))

        out = {k: v.clone() for k, v in avg_update_cpu.items()}
        for k in keys:
            out[k] = (out[k].detach().float() - alpha * g_clean_cpu[k].detach().float()).to(out[k].dtype)

        dot_after = 0.0
        for k in keys:
            dot_after += float((out[k].detach().float() * g_clean_cpu[k].detach().float()).sum().item())

        return out, {
            "cgp_used": 1.0,
            "cgp_dot_before": float(dot),
            "cgp_dot_after": float(dot_after),
            "cgp_alpha": float(alpha),
        }

    def _score_suspicious_channels_cpu(
        self,
        client_updates_cpu: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
    ) -> Tuple[Dict[Tuple[str, int], float], Dict[str, float]]:

        if not client_updates_cpu:
            return {}, {}

        w = torch.as_tensor([max(1, int(s)) for s in client_sizes], dtype=torch.float32)
        w_sum = float(max(1.0, w.sum().item()))

        susp: Dict[Tuple[str, int], float] = {}
        cons_list, act_list, mad_list, tail_list = [], [], [], []

        ref = client_updates_cpu[0]
        for A_key, B_key in iter_lora_pairs(ref):
            A_list, B_list = [], []
            for u in client_updates_cpu:
                if A_key not in u or B_key not in u:
                    A_list, B_list = [], []
                    break
                A_list.append(u[A_key].float())
                B_list.append(u[B_key].float())
            if not A_list:
                continue

            A = torch.stack(A_list, dim=0)
            B = torch.stack(B_list, dim=0)
            if A.dim() != 3 or B.dim() != 3:
                continue
            C = int(A.shape[0])
            r = int(A.shape[1])
            if r <= 0:
                continue

            normA2 = (A ** 2).sum(dim=2)
            normB2 = (B ** 2).sum(dim=1)
            norm = torch.sqrt(torch.clamp(normA2 + normB2, min=0.0))

            med = norm.median(dim=0).values
            mad = (norm - med.unsqueeze(0)).abs().median(dim=0).values
            mad_safe = mad + 1e-6

            z = (norm - med.unsqueeze(0)) / mad_safe.unsqueeze(0)

            act = (z > 2.0).float().mean(dim=0)
            tail = (z > 4.0).float().mean(dim=0)

            trim_mask = (z.abs() <= 2.5).float()
            w_trim = w.view(C, 1) * trim_mask
            sum_w = torch.clamp(w_trim.sum(dim=0), min=1.0)

            meanA = (w_trim.unsqueeze(2) * A).sum(dim=0) / sum_w.unsqueeze(1)
            meanB = (w_trim.unsqueeze(1) * B).sum(dim=0) / sum_w.unsqueeze(0)

            dotA = (A * meanA.unsqueeze(0)).sum(dim=2)
            dotB = (B * meanB.unsqueeze(0)).sum(dim=1)
            dot = dotA + dotB

            norm_mean = torch.sqrt(torch.clamp((meanA ** 2).sum(dim=1) + (meanB ** 2).sum(dim=0), min=0.0))
            denom = (norm * norm_mean.unsqueeze(0)) + 1e-6
            cos = torch.clamp(dot / denom, min=-1.0, max=1.0)

            cons = (w.view(C, 1) * torch.relu(cos)).sum(dim=0) / w_sum
            cons = torch.clamp(cons, 0.0, 1.0)

            mad_ratio = mad_safe / (med.abs() + 1e-6)

            susp_r = ((mad_ratio + 1e-6) ** 2.0) \
                   * ((tail + 1e-6) ** 2.0) \
                   * ((1.0 / (cons + 1e-6)) ** 2.0) \
                   * ((1.0 / (act + 0.2)) ** 1.0)

            for j in range(r):
                susp[(A_key, j)] = float(susp_r[j].item())

            cons_list.extend(cons.tolist())
            act_list.extend(act.tolist())
            mad_list.extend(mad.tolist())
            tail_list.extend(tail.tolist())

        stats = {}
        if cons_list:
            stats["cons_mean"] = float(sum(float(x) for x in cons_list) / len(cons_list))
            stats["act_mean"] = float(sum(float(x) for x in act_list) / len(act_list))
            stats["mad_mean"] = float(sum(float(x) for x in mad_list) / len(mad_list))
            stats["tail_mean"] = float(sum(float(x) for x in tail_list) / len(tail_list))
        return susp, stats

    def _score_p_clean_drop(self, model, candidate_channels: List[Tuple[str, int]]) -> Dict[Tuple[str, int], float]:
        if not self.val_path or not candidate_channels:
            return {}

        ds = get_dataset(self.val_path, self.model_name, client_id=0,
                         data_name=self.data_name, is_server=True)
        loader = ds.get_loader(batch_size=int(sf_cfg.get("batch_size", 64)), shuffle=False)

        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i + 1 >= 6:
                break
        if not batches:
            return {}

        model.eval()
        model.to(self.device)

        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in batches:
                x = batch["pixel_values"].to(self.device)
                y = self._extract_labels_from_batch(batch).to(self.device)
                out = model(pixel_values=x, labels=y, return_loss=True)
                total_loss += float(out["loss"].detach().item()) * x.size(0)
                total_samples += int(x.size(0))

        if total_samples <= 0:
            return {}
        baseline_loss = total_loss / total_samples

        lora_params = self._get_lora_param_tensors(model)
        p: Dict[Tuple[str, int], float] = {}

        for (A_key, j) in candidate_channels:
            B_key = A_key.replace("lora_A", "lora_B")
            pA = lora_params.get(A_key, None)
            pB = lora_params.get(B_key, None)
            if pA is None or pB is None:
                continue
            if pA.data.dim() != 2 or pB.data.dim() != 2:
                continue
            if j < 0 or j >= pA.data.shape[0] or j >= pB.data.shape[1]:
                continue

            A_backup = pA.data[j, :].detach().clone()
            B_backup = pB.data[:, j].detach().clone()

            pA.data[j, :].zero_()
            pB.data[:, j].zero_()

            total_loss_removed = 0.0
            total_samples_removed = 0
            with torch.no_grad():
                for batch in batches:
                    x = batch["pixel_values"].to(self.device)
                    y = self._extract_labels_from_batch(batch).to(self.device)
                    out = model(pixel_values=x, labels=y, return_loss=True)
                    total_loss_removed += float(out["loss"].detach().item()) * x.size(0)
                    total_samples_removed += int(x.size(0))

            avg_loss_removed = (total_loss_removed / total_samples_removed) if total_samples_removed > 0 else baseline_loss

            pA.data[j, :] = A_backup
            pB.data[:, j] = B_backup

            diff = avg_loss_removed - baseline_loss
            p[(A_key, j)] = float(diff) if diff > 0 else 0.0

        return p

    def aggregate(self,
                  client_updates: List[Dict],
                  client_sizes: List[int],
                  client_ids: Optional[List[str]],
                  global_model,
                  current_round: int = 1,
                  benign_client_ids: Optional[List[int]] = None):

        self._current_round = int(current_round)
        start_time = time.perf_counter()

        client_ids_for_hdr = None
        if client_ids is not None and len(client_ids) == len(client_updates):
            client_ids_for_hdr = [str(x) for x in client_ids]

        client_ids = self._get_client_ids(client_updates)
        client_updates_cpu = [self._to_cpu_dict(u) for u in client_updates]
        global_lora_sd_cpu = {k: v.detach().cpu().clone() for k, v in global_model.lora_model.state_dict().items()}

        g_clean_cpu = {}
        probe_loss = None
        cce_stats = {"cce_used": 0.0}
        hdr_stats = {}
        ref_stats = self._build_reference_stats()
        cce_weights = [1.0 for _ in client_updates_cpu]
        cgp_stats = {"cgp_used": 0.0}

        if self.probe_path:
            ds_probe = get_dataset(self.probe_path, self.model_name, client_id=0,
                                   data_name=self.data_name, is_server=True)
            probe_loader = ds_probe.get_loader(batch_size=int(sf_cfg.get("batch_size", 64)), shuffle=False)

            global_model.lora_model.load_state_dict(global_lora_sd_cpu, strict=False)
            g_clean_cpu, probe_loss = self._compute_lora_grads_and_probe_loss_cpu(global_model, probe_loader, max_batches=self.cce_max_batches)

        norms = []
        for upd in client_updates_cpu:
            u = self._flat_cpu(upd)
            norms.append(float(u.norm(p=2).item()))
        conv_stats = self._update_convergence(float(sum(norms) / max(1, len(norms))), probe_loss)

        if g_clean_cpu:
            scaled_cpu, cce_stats, cce_weights, _ = self._cce_progressive_gate(
                client_updates_cpu, client_sizes, g_clean_cpu, client_ids, benign_client_ids
            )
        else:
            scaled_cpu = client_updates_cpu
            
        hdr_stats = self._build_hdr_stats(
            client_ids_for_hdr if client_ids_for_hdr is not None else client_ids,
            cce_weights,
            benign_client_ids
        )

        avg_update, fedavg_metrics = self.base_aggregator.aggregate(
            client_updates=scaled_cpu,
            client_sizes=client_sizes,
            client_ids=client_ids,
        )
        avg_update_cpu = self._to_cpu_dict(avg_update)
        t_after_fedavg = time.perf_counter()

        if g_clean_cpu:
            avg_update_cpu, cgp_stats = self._clean_gradient_projection(avg_update_cpu, g_clean_cpu)

        agg_sd_cpu = {k: v.clone() for k, v in global_lora_sd_cpu.items()}
        for k, v in avg_update_cpu.items():
            if k in agg_sd_cpu:
                agg_sd_cpu[k] = agg_sd_cpu[k] + v.to(dtype=agg_sd_cpu[k].dtype)

        if self.init_state is None:
            self.init_state = {
                k: v.clone()
                for k, v in global_lora_sd_cpu.items()
                if ("lora_A" in k or "lora_B" in k)
            }

        t_before_scoring = time.perf_counter()
        susp, susp_stats = self._score_suspicious_channels_cpu(scaled_cpu, client_sizes)

        if not susp:
            end_time = time.perf_counter()
            metrics = {
                "total_time": float(end_time - start_time),
                "time_fedavg": float(fedavg_metrics.get("total_time", t_after_fedavg - start_time)),
                "time_scoring": float(end_time - t_before_scoring),
                "time_filtering": 0.0,
                "num_channels": 0,
                "selected_channels": 0,
                **conv_stats,
                **cce_stats,
                **cgp_stats,
                **susp_stats,
                **hdr_stats,
                **ref_stats,
            }
            self.last_metrics = metrics
            return avg_update_cpu, metrics

        susp_items = sorted(susp.items(), key=lambda x: x[1], reverse=True)
        num_channels = len(susp_items)

        cand_k = min(max(1, int(num_channels * 0.20)), num_channels)
        candidates = [k for k, _ in susp_items[:cand_k]]

        if self.val_path:
            global_model.lora_model.load_state_dict(agg_sd_cpu, strict=False)
            p = self._score_p_clean_drop(global_model, candidates)
            global_model.lora_model.load_state_dict(global_lora_sd_cpu, strict=False)
        else:
            p = {}

        p_vals = list(p.values())
        p_missing_default = float(sorted(p_vals)[len(p_vals) // 2]) if p_vals else 0.0
        p_phi = float(self.phi if self.phi is not None else 0.7)

        score = {}
        for ch, sv in susp.items():
            pk = float(p.get(ch, p_missing_default))
            score[ch] = float(sv / ((pk + 1e-6) ** p_phi))

        t_after_scoring = time.perf_counter()
        t_before_filter = time.perf_counter()

        items = sorted(score.items(), key=lambda x: x[1], reverse=True)
        K = len(items)
        all_scores = [sc for _, sc in items]
        score_mean, score_std, score_min, score_max = self._summary_stats(all_scores)

        top_k = min(max(1, int(K * 0.08)), K)
        selected = items[:top_k]

        reset_th = self._quantile(all_scores, 0.985)
        freeze_th = self._quantile(all_scores, 0.93)
        hard_zero_th = self._quantile(all_scores, 0.995)

        denom = (score_max - score_min) + 1e-12
        filtered_sd = {k: v.clone() for k, v in agg_sd_cpu.items()}

        num_reset = 0
        num_scale = 0
        num_hard_zero = 0
        num_frozen_new = 0

        for (A_key, j), sc in selected:
            if self.freeze_counters[(A_key, j)] > 0:
                continue

            B_key = A_key.replace("lora_A", "lora_B")
            if A_key not in filtered_sd or B_key not in filtered_sd:
                continue

            if sc >= hard_zero_th:
                filtered_sd[A_key][j, :].zero_()
                filtered_sd[B_key][:, j].zero_()
                num_hard_zero += 1

            elif sc >= reset_th:
                if self.init_state and A_key in self.init_state and B_key in self.init_state:
                    filtered_sd[A_key][j, :] = self.init_state[A_key][j, :].clone()
                    filtered_sd[B_key][:, j] = self.init_state[B_key][:, j].clone()
                else:
                    filtered_sd[A_key][j, :] *= 0.02
                    filtered_sd[B_key][:, j] *= 0.02
                num_reset += 1

            else:
                scale = max(0.02, 1.0 - 0.90 * float((sc - score_min) / denom))
                filtered_sd[A_key][j, :] *= scale
                filtered_sd[B_key][:, j] *= scale
                num_scale += 1

            if sc >= freeze_th:
                if self.freeze_counters[(A_key, j)] <= 0:
                    self.freeze_counters[(A_key, j)] = 6
                    num_frozen_new += 1

        for (A_key, j), rounds in list(self.freeze_counters.items()):
            if rounds > 0:
                B_key = A_key.replace("lora_A", "lora_B")
                if A_key in filtered_sd and B_key in filtered_sd:
                    if self.init_state and A_key in self.init_state and B_key in self.init_state:
                        filtered_sd[A_key][j, :] = self.init_state[A_key][j, :].clone()
                        filtered_sd[B_key][:, j] = self.init_state[B_key][:, j].clone()
                    else:
                        filtered_sd[A_key][j, :] *= 0.02

                self.freeze_counters[(A_key, j)] = rounds - 1

        num_frozen = sum(1 for v in self.freeze_counters.values() if v > 0)

        post_delta: Dict[str, torch.Tensor] = {}
        for k, v in avg_update_cpu.items():
            if k in filtered_sd and k in global_lora_sd_cpu:
                post_delta[k] = (filtered_sd[k] - global_lora_sd_cpu[k]).detach().clone()
            else:
                post_delta[k] = v.detach().clone()

        end_time = time.perf_counter()

        trust_vals = [float(v) for v in self.client_trust.values()] if len(self.client_trust) > 0 else [1.0]
        ban_vals = [float(v) for v in self.client_ban_score.values()] if len(self.client_ban_score) > 0 else [0.0]

        metrics = {
            "total_time": float(end_time - start_time),
            "time_fedavg": float(fedavg_metrics.get("total_time", t_after_fedavg - start_time)),
            "time_scoring": float(t_after_scoring - t_before_scoring),
            "time_filtering": float(end_time - t_before_filter),

            "num_channels": int(K),
            "candidate_channels_for_p": int(cand_k),
            "selected_channels": int(top_k),
            "top_ratio_effective": float(top_k) / float(K),

            "num_reset": int(num_reset),
            "num_scale": int(num_scale),
            "num_hard_zero": int(num_hard_zero),
            "num_frozen_new": int(num_frozen_new),
            "num_frozen": int(num_frozen),

            "score_mean": float(score_mean),
            "score_std": float(score_std),
            "score_min": float(score_min),
            "score_max": float(score_max),

            "reset_th": float(reset_th),
            "freeze_th": float(freeze_th),
            "hard_zero_th": float(hard_zero_th),

            "p_missing_default": float(p_missing_default),

            "cce_trust_min_global": float(min(trust_vals)),
            "cce_trust_mean_global": float(sum(trust_vals) / max(1, len(trust_vals))),
            "cce_ban_score_mean_global": float(sum(ban_vals) / max(1, len(ban_vals))),
            "cce_ban_score_max_global": float(max(ban_vals)),

            **conv_stats,
            **cce_stats,
            **cgp_stats,
            **susp_stats,
            **hdr_stats,
            **ref_stats,
        }

        self.last_metrics = metrics
        return post_delta, metrics