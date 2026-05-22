import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from sf.config.attack_config import (
    attack as attack_cfg,
    rank_aware as ra_cfg,
    backdoor as bd_cfg,
    alie as alie_cfg,
    minmax as minmax_cfg,
    minsum as minsum_cfg,
    poisonedfl as pf_cfg,
    signflip as sf_cfg,
    scaling as scaling_cfg,
    gaussian as gauss_cfg,

    neurotoxin as nt_cfg,
    cerp as cerp_cfg,
    pfedba as pfedba_cfg,
)


class ByzantineAttacker:
    _GLOBAL_PFEDBA_PATCH = None

    def __init__(self, attack_type: str, device, enable_backdoor: bool = False, trigger_type: str = "none"):
        self.device = device
        self.attack_type = (attack_type or "none").lower().strip()

        self.trigger_type = (trigger_type or "none").lower().strip()
        if self.trigger_type == "none":
            raise ValueError("trigger_type cannot be 'none' when using backdoor attack.")
        
        self.enable_backdoor = bool(enable_backdoor) and (self.trigger_type != "none")

        self.poison_intensity = float(attack_cfg.get("poison_intensity", 10.0))

        self.alie_z = float(alie_cfg.get("z", 2.5))
        self.alie_sign = float(np.sign(alie_cfg.get("sign", -1)))
        self.alie_eps = float(alie_cfg.get("eps", 1e-9))

        self.mm_dir = minmax_cfg.get("direction", "neg_mean")
        self.mm_gamma_cap = float(minmax_cfg.get("gamma_cap", 1e9))
        self.mm_scale = float(minmax_cfg.get("gamma_scale", 1.0))
        self.mm_nonneg = bool(minmax_cfg.get("nonneg", True))
        self.mm_eps = float(minmax_cfg.get("eps", 1e-12))
        self.mm_align_mix = float(minmax_cfg.get("align_mix", 0.0))
        self.mm_cos_min = float(minmax_cfg.get("cos_min", 0.0))
        self.mm_tau_norm = minmax_cfg.get("tau_norm", None)

        self.ms_cfg = minsum_cfg

        self.pf_s: Optional[torch.Tensor] = None
        self.pf_prev_k: Optional[torch.Tensor] = None
        self.pf_prev_global: Optional[torch.Tensor] = None
        self.poison_c: float = float(pf_cfg.get("c_init", 8.0))
        self.poison_c_min: float = float(pf_cfg.get("c_min", 0.5))
        self.poison_decay: float = float(pf_cfg.get("decay", 0.7))
        self.poison_eps: float = float(pf_cfg.get("eps", 1e-12))
        self.consistency_cos_min: float = float(pf_cfg.get("cos_min", 0.0))
        self.round_idx: int = 0

        self.sf_gamma = float(sf_cfg.get("gamma") or self.poison_intensity)
        self.scaling_lambda = float(scaling_cfg.get("lambda") or self.poison_intensity)

        self.gauss_std_scale = float(gauss_cfg.get("std_scale", 1.0))
        self.gauss_eps = float(gauss_cfg.get("eps", 1e-12))

        self.target_dir: Dict[str, torch.Tensor] = {}

        self.benign_mu: Dict[str, torch.Tensor] = {}
        self.benign_sigma: Dict[str, torch.Tensor] = {}
        self._order: Optional[List[str]] = None
        self._shapes: Dict[str, torch.Size] = {}
        self._dtypes: Dict[str, torch.dtype] = {}

        self._benign_vecs: Optional[torch.Tensor] = None
        self._benign_center: Optional[torch.Tensor] = None
        self._benign_dmax: Optional[float] = None
        self._benign_sum_max: Optional[float] = None

        self._pfedba_patch: Optional[torch.Tensor] = None
        self._pfedba_optim: Optional[torch.optim.Optimizer] = None

        self._tact_patch: Optional[torch.Tensor] = None
        self._tact_patch_channels: Optional[int] = None
        self._tact_patch_size: Optional[int] = None

        self._blended_pattern: Optional[torch.Tensor] = None
        self._blended_pattern_channels: Optional[int] = None

    @property
    def target_label(self):
        return int(bd_cfg.get("target_label", 0))

    def _get_trigger_position(self, h: int, w: int, size: int) -> Tuple[int, int]:
        pos = bd_cfg.get("trigger_position", "bottom_right")
        if self.attack_type == "pfedba" and bool(pfedba_cfg.get("random_position", False)):
            pos = "random"

        if pos == "random":
            y0 = int(torch.randint(low=0, high=max(1, h - size + 1), size=(1,)).item())
            x0 = int(torch.randint(low=0, high=max(1, w - size + 1), size=(1,)).item())
            return y0, x0
        if pos == "top_left":
            return 0, 0
        return h - size, w - size

    def add_badnets_trigger(self, pixel_values: torch.Tensor):
        size = int(bd_cfg.get("trigger_size", 4))
        val = float(bd_cfg.get("trigger_value", 1.0))

        b, c, h, w = pixel_values.shape
        size = max(1, min(size, h, w))
        y0, x0 = self._get_trigger_position(h, w, size)

        x = pixel_values.clone()
        x[:, :, y0:y0 + size, x0:x0 + size] = val
        return x

    def _init_tact_patch(self, channels: int, size: int):
        if self._tact_patch is not None and self._tact_patch_channels == channels and self._tact_patch_size == size:
            return

        seed = int(bd_cfg.get("tact_seed", 0))
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        patch = torch.rand((1, channels, size, size), generator=gen, device=self.device, dtype=torch.float32)
        self._tact_patch = patch.detach()
        self._tact_patch_channels = channels
        self._tact_patch_size = size

    def add_tact_trigger(self, pixel_values: torch.Tensor):
        size = int(bd_cfg.get("trigger_size", 4))

        b, c, h, w = pixel_values.shape
        size = max(1, min(size, h, w))
        y0, x0 = self._get_trigger_position(h, w, size)

        self._init_tact_patch(channels=c, size=size)
        assert self._tact_patch is not None

        patch = self._tact_patch
        if patch.shape[-1] != size:
            patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)

        x = pixel_values.clone()
        x[:, :, y0:y0 + size, x0:x0 + size] = patch.expand(b, -1, -1, -1)
        return x

    def _init_blended_pattern(self, channels: int, h: int, w: int):
        if self._blended_pattern is not None and self._blended_pattern_channels == channels:
            return

        seed = int(bd_cfg.get("blended_seed", 0))
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        pat = torch.rand((1, channels, h, w), generator=gen, device=self.device, dtype=torch.float32)
        self._blended_pattern = pat.detach()
        self._blended_pattern_channels = channels

    def add_blended_trigger(self, pixel_values: torch.Tensor):
        alpha = float(bd_cfg.get("blended_alpha", 0.2))
        alpha = max(0.0, min(1.0, alpha))

        b, c, h, w = pixel_values.shape
        self._init_blended_pattern(channels=c, h=h, w=w)
        assert self._blended_pattern is not None

        pat = self._blended_pattern
        if pat.shape[-2:] != (h, w):
            pat = F.interpolate(pat, size=(h, w), mode="bilinear", align_corners=False)

        pat = pat.expand(b, -1, -1, -1)
        x = (1.0 - alpha) * pixel_values + alpha * pat
        return x

    def _init_pfedba_patch(self, channels: int):
        if ByzantineAttacker._GLOBAL_PFEDBA_PATCH is None:
            size = int(bd_cfg.get("trigger_size", 4))
            patch = torch.rand((1, channels, size, size), device=self.device, dtype=torch.float32)
            patch.requires_grad_(True)
            ByzantineAttacker._GLOBAL_PFEDBA_PATCH = patch

        self._pfedba_patch = ByzantineAttacker._GLOBAL_PFEDBA_PATCH
        
        if self._pfedba_optim is None:
            lr = float(pfedba_cfg.get("trigger_lr", 0.1))
            self._pfedba_optim = torch.optim.SGD([self._pfedba_patch], lr=lr)

    def _tv_loss(self, patch: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]).mean()
        dw = torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]).mean()
        return dh + dw

    def add_pfedba_trigger(self, pixel_values: torch.Tensor):
        assert self._pfedba_patch is not None
        alpha = float(bd_cfg.get("pfedba_alpha", 0.2))

        b, c, h, w = pixel_values.shape
        size = int(self._pfedba_patch.shape[-1])
        size = max(1, min(size, h, w))
        y0, x0 = self._get_trigger_position(h, w, size)

        patch = self._pfedba_patch
        if patch.shape[-1] != size:
            patch = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)

        x = pixel_values.clone()
        region = x[:, :, y0:y0 + size, x0:x0 + size]
        x[:, :, y0:y0 + size, x0:x0 + size] = (1.0 - alpha) * region + alpha * patch
        return x

    def _pfedba_update_patch(self, model, pixel_values: torch.Tensor, labels: torch.Tensor):
        if model is None or self._pfedba_patch is None or self._pfedba_optim is None:
            return
        if not bool(pfedba_cfg.get("trigger_opt", True)):
            return

        steps = int(pfedba_cfg.get("trigger_steps", 1))
        if steps <= 0:
            return

        tv_w = float(pfedba_cfg.get("tv_weight", 1e-4))
        l2_w = float(pfedba_cfg.get("l2_weight", 1e-4))

        reqs = []
        for p in model.parameters():
            reqs.append(p.requires_grad)
            p.requires_grad_(False)

        try:
            for _ in range(steps):
                self._pfedba_optim.zero_grad(set_to_none=True)

                x_trig = self.add_pfedba_trigger(pixel_values)
                y_tgt = torch.full_like(labels, fill_value=self.target_label)

                out = model(pixel_values=x_trig, labels=y_tgt, return_loss=True)
                loss = out["loss"] if isinstance(out, dict) and "loss" in out else out[0]

                reg_tv = self._tv_loss(self._pfedba_patch)
                reg_l2 = torch.mean(self._pfedba_patch * self._pfedba_patch)

                total = loss + tv_w * reg_tv + l2_w * reg_l2
                total.backward()

                self._pfedba_optim.step()
                with torch.no_grad():
                    self._pfedba_patch.data.clamp_(0.0, 1.0)
        finally:
            for p, r in zip(model.parameters(), reqs):
                p.requires_grad_(r)


    def _apply_trigger(self, pixel_values: torch.Tensor, model=None, labels=None):
        if self.attack_type == "pfedba":
            b, c, h, w = pixel_values.shape
            self._init_pfedba_patch(channels=c)
            if labels is not None:
                self._pfedba_update_patch(model, pixel_values, labels)
            return self.add_pfedba_trigger(pixel_values)

        t = (self.trigger_type or bd_cfg.get("trigger_type", "badnets")).lower().strip()
        if t == "blended":
            return self.add_blended_trigger(pixel_values)
        if t == "tact":
            return self.add_tact_trigger(pixel_values)
        return self.add_badnets_trigger(pixel_values)

    def poison_batch(self, pixel_values: torch.Tensor, labels: torch.Tensor, model=None):
        if not self.enable_backdoor:
            return pixel_values, labels

        poison_rate = float(bd_cfg.get("poison_ratio", 1.0))
        if poison_rate <= 0.0:
            return pixel_values, labels

        tgt = self.target_label
        ttype = (self.trigger_type).lower().strip()

        b, c, h, w = pixel_values.shape

        def _label_transform(sel_mask: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if ttype == "tact" and bool(bd_cfg.get("tact_enable_cover", True)):
                out = y.clone()
                to_flip = sel_mask & (y != tgt)
                out[to_flip] = tgt
                return out
            out = y.clone()
            out[sel_mask] = tgt
            return out

        if poison_rate >= 1.0:
            x = self._apply_trigger(pixel_values, model=model, labels=labels)
            y = _label_transform(torch.ones((b,), device=labels.device, dtype=torch.bool), labels)
            return x, y

        mask = torch.rand(b, device=pixel_values.device) < poison_rate
        if not mask.any():
            return pixel_values, labels

        x = pixel_values.clone()
        x[mask] = self._apply_trigger(pixel_values[mask], model=model, labels=labels[mask])

        y = _label_transform(mask, labels)
        return x, y

    def _flat(self, update: dict) -> torch.Tensor:
        if self._order is None:
            self._order = list(update.keys())
            for k, v in update.items():
                self._shapes[k] = v.shape
                self._dtypes[k] = v.dtype

        vecs = [update[k].reshape(-1) for k in self._order]
        return torch.cat(vecs).float().to(self.device)

    def _unflat(self, vec: torch.Tensor) -> dict:
        outputs = {}
        offset = 0
        for k in self._order:
            vec_len = int(np.prod(self._shapes[k]))
            chunk = vec[offset:offset + vec_len].reshape(self._shapes[k])
            outputs[k] = chunk.to(self._dtypes[k]).detach().cpu()
            offset += vec_len
        return outputs

    @torch.no_grad()
    def set_benign_stats(self, benign_updates_list):
        if len(benign_updates_list) == 0:
            self._benign_vecs = None
            self._benign_center = None
            self._benign_dmax = None
            self._benign_sum_max = None
            return

        keys = benign_updates_list[0].keys()
        for k in keys:
            stacked = torch.stack([u[k].to(self.device) for u in benign_updates_list], dim=0)
            self.benign_mu[k] = stacked.mean(dim=0)
            self.benign_sigma[k] = stacked.std(dim=0, unbiased=False)

        vecs = torch.stack([self._flat(u) for u in benign_updates_list], dim=0)
        self._benign_vecs = vecs
        self._benign_center = vecs.mean(dim=0)

        if vecs.size(0) >= 2:
            dists = torch.cdist(vecs, vecs, p=2)
            self._benign_dmax = float(dists.max().item())
            dists_sq = dists * dists
            sums = dists_sq.sum(dim=1)
            self._benign_sum_max = float(sums.max().item())
        else:
            self._benign_dmax = 0.0
            self._benign_sum_max = 0.0

    def get_target_singular_dir(self, benign_update, key):
        if key not in self.target_dir:
            X = benign_update[key].detach().to(self.device).float()
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            self.target_dir[key] = U[:, 0:1].detach()
        return self.target_dir[key]

    def _choose_attack_direction(self, clean_vec, f_avg, benign_vecs):
        if self.mm_dir == "clean":
            v_base = -clean_vec.clone()
        elif self.mm_dir == "principal" and benign_vecs is not None and benign_vecs.size(0) >= 2:
            X = benign_vecs - f_avg.unsqueeze(0)
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            v_base = -Vh[0]
        else:
            v_base = -f_avg.clone()

        v = v_base / (torch.norm(v_base) + self.mm_eps)

        if self.mm_align_mix > 0.0:
            v_align = clean_vec / (torch.norm(clean_vec) + self.mm_eps)
            v = (1.0 - self.mm_align_mix) * v + self.mm_align_mix * v_align
            v = v / (torch.norm(v) + self.mm_eps)

        if torch.norm(v) < 1e-12:
            v = torch.randn_like(v_base)
            v = v / (torch.norm(v) + self.mm_eps)
        return v

    def _solve_gamma_max(self, f_avg, v, benign_vecs, D_max):
        a = torch.dot(v, v) + self.mm_eps
        delta = f_avg.unsqueeze(0) - benign_vecs
        b = 2.0 * (delta @ v)
        c = (delta * delta).sum(dim=1) - (D_max ** 2)

        disc = torch.clamp(b * b - 4.0 * a * c, min=0.0)
        sqrt_disc = torch.sqrt(disc + self.mm_eps)

        gamma_candidates = (-b + sqrt_disc) / (2.0 * a)
        if self.mm_nonneg:
            gamma_candidates = torch.clamp(gamma_candidates, min=0.0)

        gamma_max = float(gamma_candidates.min().item())
        gamma_max = gamma_max * self.mm_scale
        gamma_max = min(gamma_max, self.mm_gamma_cap)
        gamma_max = max(0.0, gamma_max)
        return gamma_max

    def _minmax_poison_vector(self, clean_vec, benign_avg_update):
        if self._benign_center is None:
            f_avg = self._flat(benign_avg_update) if benign_avg_update is not None else torch.zeros_like(clean_vec)
            D_max = 0.0
            benign_vecs = None
        else:
            f_avg = self._benign_center
            D_max = self._benign_dmax or 0.0
            benign_vecs = self._benign_vecs

        v = self._choose_attack_direction(clean_vec, f_avg, benign_vecs)

        if benign_vecs is None or D_max <= 0.0:
            gamma_max = 1.0 * self.mm_scale
        else:
            gamma_max = self._solve_gamma_max(f_avg, v, benign_vecs, D_max)

        g_m = f_avg + gamma_max * v

        if self.mm_tau_norm is not None:
            tau = float(self.mm_tau_norm)
            ref_norm = torch.norm(clean_vec).item() + self.mm_eps
            g_norm = torch.norm(g_m).item() + self.mm_eps
            if g_norm > tau * ref_norm:
                g_m = g_m * ((tau * ref_norm) / g_norm)

        return g_m

    def _minsum_poison_vector(self, clean_vec, benign_avg_update):
        if self._benign_center is None or self._benign_vecs is None:
            f_avg = self._flat(benign_avg_update) if benign_avg_update is not None else torch.zeros_like(clean_vec)
            benign_vecs = None
            S_max = 0.0
        else:
            f_avg = self._benign_center
            benign_vecs = self._benign_vecs
            S_max = self._benign_sum_max or 0.0

        v = self._choose_attack_direction(clean_vec, f_avg, benign_vecs)

        if benign_vecs is None or S_max <= 0.0:
            gamma_max = 1.0 * self.mm_scale
        else:
            m = benign_vecs.size(0)
            delta = f_avg.unsqueeze(0) - benign_vecs
            A = float(m) * float(torch.dot(v, v).item() + self.mm_eps)
            B = float(2.0 * torch.matmul(delta, v).sum().item())
            C0 = float((delta * delta).sum().item() - S_max)

            disc = B * B - 4.0 * A * C0
            if disc <= 0.0 or A <= 0:
                gamma_max = 0.0
            else:
                sqrt_disc = math.sqrt(disc)
                r2 = (-B + sqrt_disc) / (2.0 * A)
                gamma_max = max(0.0, r2) if self.mm_nonneg else r2

            gamma_max = gamma_max * self.mm_scale
            gamma_max = min(gamma_max, self.mm_gamma_cap)

        g_m = f_avg + gamma_max * v

        if self.mm_tau_norm is not None:
            tau = float(self.mm_tau_norm)
            ref_norm = torch.norm(clean_vec).item() + self.mm_eps
            g_norm = torch.norm(g_m).item() + self.mm_eps
            if g_norm > tau * ref_norm:
                g_m = g_m * ((tau * ref_norm) / g_norm)

        return g_m

    def _init_poisonedfl_state(self, dim_d):
        if self.pf_s is None:
            random_bits = torch.randint(0, 2, (dim_d,), device=self.device)
            self.pf_s = (random_bits.float() * 2.0 - 1.0)
            if torch.all(self.pf_s > 0) or torch.all(self.pf_s < 0):
                self.pf_s[0] = -self.pf_s[0]

    def _estimate_vt(self):
        if self.pf_prev_global is None or self.pf_prev_k is None:
            v = torch.ones_like(self.pf_s, dtype=torch.float32, device=self.device)
            return v / (torch.norm(v) + self.poison_eps)

        g_prev = self.pf_prev_global
        atk_prev = self.pf_prev_k * self.pf_s

        atk_prev_norm = torch.norm(atk_prev) + self.poison_eps
        g_prev_norm = torch.norm(g_prev) + self.poison_eps

        atk_prev_scale = atk_prev * (g_prev_norm / atk_prev_norm)
        benign_est = g_prev - atk_prev_scale
        abs_benign_est = torch.abs(benign_est)
        v = abs_benign_est / (torch.norm(abs_benign_est) + self.poison_eps)
        return v

    def _calc_poisonedfl_alpha_t(self):
        if self.pf_prev_global is None:
            return float(pf_cfg.get("warmup_alpha", 1e-3))
        g_prev_norm = torch.norm(self.pf_prev_global).item()
        return float(g_prev_norm * self.poison_c)

    def _check_consistency_and_decay(self):
        if self.pf_prev_global is None or self.pf_s is None:
            return
        cos_dir = torch.nn.functional.cosine_similarity(
            self.pf_prev_global.unsqueeze(0),
            self.pf_s.unsqueeze(0),
            dim=1
        ).item()
        if cos_dir < self.consistency_cos_min:
            self.poison_c = max(self.poison_c * self.poison_decay, self.poison_c_min)

    def update_round_info(self, global_prev_vec: Optional[torch.Tensor]):
        self.round_idx += 1
        if global_prev_vec is not None:
            self.pf_prev_global = global_prev_vec.detach().to(self.device).float()
        self._check_consistency_and_decay()

    def _get_benign_center_and_dmax(self, benign_avg_update, like_vec: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self._benign_center is not None:
            return self._benign_center, float(self._benign_dmax or 0.0)
        if benign_avg_update is not None:
            center = self._flat({k: v.to(self.device) for k, v in benign_avg_update.items()})
            return center, 0.0
        return torch.zeros_like(like_vec), 0.0

    def _project_to_ball(self, vec: torch.Tensor, center: torch.Tensor, radius: float, eps: float) -> torch.Tensor:
        if radius is None or radius <= 0:
            return vec
        diff = vec - center
        dist = torch.norm(diff).item()
        if dist <= radius:
            return vec
        return center + diff * (radius / (dist + eps))

    def _cerp_postprocess(self, vec: torch.Tensor, benign_avg_update) -> torch.Tensor:
        eps = float(cerp_cfg.get("eps", 1e-12))
        center, dmax = self._get_benign_center_and_dmax(benign_avg_update, vec)

        radius_scale = float(cerp_cfg.get("radius_scale", 1.0))
        radius = radius_scale * dmax
        if radius <= 0:
            radius = 0.1 * (torch.norm(center).item() + eps)

        out = self._project_to_ball(vec, center, radius, eps)

        mix = float(cerp_cfg.get("mix_with_benign", 0.0))
        if mix > 0:
            out = (1.0 - mix) * out + mix * center

        clip_norm = cerp_cfg.get("clip_norm", None)
        if clip_norm is not None:
            clip_norm = float(clip_norm)
            n = torch.norm(out).item() + eps
            if n > clip_norm:
                out = out * (clip_norm / n)
        return out

    def _pfedba_postprocess(self, vec: torch.Tensor, benign_avg_update) -> torch.Tensor:
        eps = float(pfedba_cfg.get("eps", 1e-12))
        center, dmax = self._get_benign_center_and_dmax(benign_avg_update, vec)

        scale = float(pfedba_cfg.get("update_scale", 1.0))
        out = center + scale * (vec - center)

        if bool(pfedba_cfg.get("stealth_projection", True)):
            radius_scale = float(pfedba_cfg.get("radius_scale", 1.0))
            radius = radius_scale * dmax
            
            survival_rate = 0.8
            original_dist = torch.norm(vec - center).item()
            min_radius = survival_rate * original_dist + eps 
            
            if radius < min_radius:
                radius = min_radius
                
            out = self._project_to_ball(out, center, radius, eps)

        clip_norm = pfedba_cfg.get("clip_norm", None)
        if clip_norm is not None:
            clip_norm = float(clip_norm)
            n = torch.norm(out).item() + eps
            if n > clip_norm:
                out = out * (clip_norm / n)
        return out

    def _apply_neurotoxin(self, vec: torch.Tensor, benign_avg_update, clean_vec_for_score: Optional[torch.Tensor]) -> torch.Tensor:
        eps = float(nt_cfg.get("eps", 1e-12))
        keep_ratio = float(nt_cfg.get("keep_ratio", 0.2))
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        if keep_ratio >= 1.0:
            return vec

        score_source = str(nt_cfg.get("score_source", "benign")).lower()
        if score_source == "clean" and clean_vec_for_score is not None:
            score = torch.abs(clean_vec_for_score.detach())
        else:
            if self._benign_center is not None:
                score = torch.abs(self._benign_center.detach())
            elif benign_avg_update is not None:
                score = torch.abs(self._flat({k: v.to(self.device) for k, v in benign_avg_update.items()}).detach())
            else:
                score = torch.abs(vec.detach())

        d = vec.numel()
        k = max(1, int(round(keep_ratio * d)))

        idx = torch.topk(score, k=k, largest=False).indices
        mask = torch.zeros_like(vec)
        mask[idx] = 1.0

        out = vec * mask

        if bool(nt_cfg.get("rescale", True)):
            n0 = torch.norm(vec).item()
            n1 = torch.norm(out).item()
            if n1 > 0:
                out = out * (n0 / (n1 + eps))
        return out

    def poison_update(self, clean_update, benign_avg_update=None):
        t = self.attack_type

        if t in ["none", "backdoor"]:
            return clean_update

        if t == "signflip":
            return {k: (-self.sf_gamma * v).detach().cpu() for k, v in clean_update.items()}

        if t == "scaling":
            return {k: (self.scaling_lambda * v).detach().cpu() for k, v in clean_update.items()}

        if t == "gaussian":
            out = {}
            for k, v in clean_update.items():
                mu = benign_avg_update[k].to(self.device) if benign_avg_update and k in benign_avg_update else v.to(self.device)
                sigma = torch.abs(mu)
                sigma = torch.maximum(sigma, torch.tensor(self.gauss_eps, device=self.device))
                noise = torch.randn_like(mu) * sigma * self.gauss_std_scale
                out[k] = (mu + noise).type_as(v).detach().cpu()
            return out

        if t == "alie":
            out = {}
            for k, v in clean_update.items():
                if k in self.benign_mu and k in self.benign_sigma:
                    mu = self.benign_mu[k]
                    sigma = self.benign_sigma[k]
                else:
                    mu = benign_avg_update[k].to(self.device) if benign_avg_update and k in benign_avg_update else torch.zeros_like(v, device=self.device)
                    sigma = torch.zeros_like(v, device=self.device)

                sigma = torch.maximum(sigma, torch.tensor(self.alie_eps, device=self.device))
                adv = mu + self.alie_sign * self.alie_z * sigma
                out[k] = adv.type_as(v).detach().cpu()
            return out

        if t == "rank_aware":
            out = {}
            for k, v in clean_update.items():
                if benign_avg_update is None or k not in benign_avg_update:
                    out[k] = v.detach().cpu()
                    continue
                target_dir = self.get_target_singular_dir(benign_avg_update, k)
                X = v.detach().to(self.device).float()
                proj = (target_dir @ target_dir.T) @ X
                strength = float(ra_cfg.get("proj_strength", 2.0))
                out[k] = (X + strength * proj).type_as(v).detach().cpu()
            return out

        if t == "minmax":
            clean_vec = self._flat({k: v.to(self.device) for k, v in clean_update.items()})
            g_m = self._minmax_poison_vector(clean_vec, benign_avg_update)
            return self._unflat(g_m)

        if t == "minsum":
            clean_vec = self._flat({k: v.to(self.device) for k, v in clean_update.items()})
            g_m = self._minsum_poison_vector(clean_vec, benign_avg_update)
            return self._unflat(g_m)

        if t == "poisonedfl":
            clean_vec = self._flat({k: v.to(self.device) for k, v in clean_update.items()})
            self._init_poisonedfl_state(clean_vec.shape[0])
            v_t = self._estimate_vt()
            alpha_t = self._calc_poisonedfl_alpha_t()
            k_t = alpha_t * v_t
            g_att = k_t * self.pf_s
            self.pf_prev_k = k_t.detach()
            return self._unflat(g_att)

        if t == "cerp":
            vec = self._flat({k: v.to(self.device) for k, v in clean_update.items()})
            vec2 = self._cerp_postprocess(vec, benign_avg_update)
            return self._unflat(vec2)

        if t == "pfedba":
            vec = self._flat({k: v.to(self.device) for k, v in clean_update.items()})
            vec2 = self._pfedba_postprocess(vec, benign_avg_update)
            return self._unflat(vec2)

        if t == "neurotoxin":
            vec = self._flat({k: v.to(self.device) for k, v in clean_update.items()})
            vec_score = vec
            vec2 = self._apply_neurotoxin(vec, benign_avg_update, clean_vec_for_score=vec_score)
            return self._unflat(vec2)

        return clean_update