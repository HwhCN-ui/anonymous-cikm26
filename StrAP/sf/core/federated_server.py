import os
import time
import math
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from sf.benchmark.fedavg_aggregator import FedAvgAggregator
from sf.utils.lora_utils import apply_lora_update, compute_lora_norm
from sf.core.federated_client import FederatedClient
from sf.core.strap_aggregator import StrAPAggregator
from sf.benchmark.deepsight_aggregator import DeepsightAggregator
from sf.benchmark.flame_aggregator import FLAMEAggregator
from sf.benchmark.lasa_aggregator import LASAAggregator
from sf.benchmark.feddlad_aggregator import FedDLADAggregator
from sf.benchmark.mesas_aggregator import MESASAggregator

from sf.benchmark.trimmed_mean_aggregator import TrimmedMeanAggregator
from sf.benchmark.median_aggregator import MedianAggregator
from sf.benchmark.fltrust_aggregator import FLTrustAggregator
from sf.benchmark.flshield_aggregator import FLShieldAggregator
from sf.utils.logger import init_logger
from sf.utils.tensorboard_utils import TBLogger
from sf.utils.evaluator_vit import evaluate_cls, evaluate_backdoor_asr
from sf.utils.seed import set_seed

from sf.config.base_config import (
    federated as fl_cfg, tensorboard as tb_cfg, log as log_cfg, train as tr_cfg, lora as lora_cfg
)
from sf.config.attack_config import backdoor as bd_cfg

from sf.models import get_model


class FederatedServer:
    def __init__(self,
                 federated_train_data_path,
                 federated_val_data_path,
                 data_name,
                 aggregator, attack,
                 enable_backdoor: bool,
                 trigger_type: str,
                 model_name,
                 byzantine_ratio,
                 lorasf_base_aggregator="fedavg",
                 lorasf_max_batches: int = 8,
                 cce_gamma: float = None,
                 cce_trust_beta: float = None,
                 phi: float = None,
                 server_ref_label_mode: str = "all",
                 server_ref_labels: str = "",
                 server_ref_random_k: int = 0,
                 server_ref_random_seed: int = 42):
        self.model_name = model_name
        self.data_name = data_name
        self.task = "vit"

        set_seed(fl_cfg.get("seed", 42))

        self.global_model = get_model(model_name)
        self.global_lora_path = lora_cfg["global_lora_path"]
        os.makedirs(os.path.dirname(self.global_lora_path), exist_ok=True)
        self.global_model.save_lora(self.global_lora_path)

        self.device = torch.device(tr_cfg["device"])

        self.fedavg_aggregator = FedAvgAggregator(device=self.device)
        self.median_aggregator = MedianAggregator(device=self.device)
        self.trimmed_aggregator = TrimmedMeanAggregator(trim_k=3, device=self.device)
        self.krum_aggregator = KrumAggregator(f=5, m=10, device=self.device)

        self.fltrust_aggregator = FLTrustAggregator(
            device=self.device,
            global_lora_path=self.global_lora_path,
            federated_data_path=federated_val_data_path,
            model_name=self.model_name,
            data_name=self.data_name,
            root_steps=50,
            root_lr=tr_cfg["lr"],
            root_weight_decay=tr_cfg["weight_decay"],
            root_batch_size=8,
            clip_coef=1.0,
            recompute_root_every=1,
            eps=1e-12
        )
        self.flshield_aggregator = FLShieldAggregator(
            device=self.device,
            global_lora_path=self.global_lora_path,
            federated_data_path=federated_val_data_path,
            model_name=self.model_name,
            data_name=self.data_name,
            keep_ratio=0.5,
            clip_coef=1.0,
            val_batch_size=8,
            max_val_batches=10,
            eps=1e-12,
        )
        self.deepsight_aggregator = DeepsightAggregator(
            device=self.device,
            model_name=self.model_name,
            data_name=self.data_name,
            federated_data_path=federated_val_data_path,
            global_lora_path=self.global_lora_path,
            probe_batch_size=8,
            max_probe_batches=10,
        )
        self.flame_aggregator = FLAMEAggregator(
            device=self.device,
            min_cluster_ratio=0.5,
            dp_epsilon=3705.0,
            dp_delta=1e-5,
            enable_noising=True,
            prefer_hdbscan=True,
        )

        self.foolsgold_aggregator = FoolsGoldAggregator(device=self.device, kappa=1.0, feature_top_ratio=0.2)

        self.mesas_aggregator = MESASAggregator(
            device=self.device,
            p_threshold=0.05,
            metric_order=["COS", "EUCL", "COUNT", "VAR", "MIN", "MAX"],
            analyze_per_key_as_layer=True,
            enable_sigma_rule=True,
            min_clients_for_tests=6,
            min_cluster_prune=1,
            max_prune_rounds=50,
            random_state=fl_cfg.get("seed", 0),
        )
        self.feddlad_aggregator = FedDLADAggregator(
            device=self.device,
            iqr_multiplier=1.5,
            pardon_threshold=0.8
        )
        
        self.lasa_aggregator = LASAAggregator(
            device=self.device,
            lambda_m=1.0,
            lambda_d=1.0,
            keep_ratio=0.7,
            eps=1e-12
        )

        if lorasf_base_aggregator == "fltrust":
            self.lorasf_base_aggregator = self.fltrust_aggregator
        elif lorasf_base_aggregator == "flshield":
            self.lorasf_base_aggregator = self.flshield_aggregator
        elif lorasf_base_aggregator == "median":
            self.lorasf_base_aggregator = self.median_aggregator
        elif lorasf_base_aggregator == "trimmed":
            self.lorasf_base_aggregator = self.trimmed_aggregator
        elif lorasf_base_aggregator == "deepsight":
            self.lorasf_base_aggregator = self.deepsight_aggregator
        elif lorasf_base_aggregator == "flame":
            self.lorasf_base_aggregator = self.flame_aggregator
        elif lorasf_base_aggregator == "mesas":
            self.lorasf_base_aggregator = self.mesas_aggregator
        elif lorasf_base_aggregator == "feddlad":
            self.lorasf_base_aggregator = self.feddlad_aggregator
        elif lorasf_base_aggregator == "lasa":
            self.lorasf_base_aggregator = self.lasa_aggregator
        else:
            self.lorasf_base_aggregator = self.fedavg_aggregator

        self.byzantine_ratio = byzantine_ratio

        self.aggregator = aggregator
        self.attack = attack

        self.trigger_type = (trigger_type or "none").lower().strip()
        self.enable_backdoor = bool(enable_backdoor) and (self.trigger_type != "none")

        bd_cfg["enabled"] = self.enable_backdoor
        bd_cfg["trigger_type"] = self.trigger_type

        self.federated_train_data_path = federated_train_data_path
        self.federated_val_data_path = federated_val_data_path

        self.lorasf_aggregator = StrAPAggregator(
            device=self.device,
            model_name=self.model_name,
            data_name=data_name,
            federated_val_data_path=federated_val_data_path,
            federated_probe_data_path=None,
            base_aggregator=self.lorasf_base_aggregator,
            lorasf_max_batches=lorasf_max_batches,
            cce_gamma=cce_gamma,
            cce_trust_beta=cce_trust_beta,
            phi=phi,
            server_ref_label_mode=server_ref_label_mode,
            server_ref_labels=server_ref_labels,
            server_ref_random_k=server_ref_random_k,
            server_ref_random_seed=server_ref_random_seed,
        )

        self.logger = init_logger(log_cfg["log_dir"], log_cfg["log_name"])
        self.tb_logger = TBLogger(tb_cfg["log_dir"])

        self.local_epochs = fl_cfg["local_epochs"]

        self.round_times = []
        self.total_times = []

        self.prev_global_step_vec = None

    def select_clients(self):
        all_clients = list(range(1, fl_cfg["num_clients"] + 1))
        selected = np.random.choice(all_clients, fl_cfg["clients_per_round"], replace=False)

        num_byzantine = int(self.byzantine_ratio * len(selected))
        if num_byzantine >= len(selected):
            num_byzantine = max(0, len(selected) - 1)

        byzantine_ids = np.random.choice(selected, num_byzantine, replace=False) if num_byzantine > 0 else []
        benign_ids = [c for c in selected if c not in byzantine_ids]

        self.logger.info(f"Round {self.current_round}: Selected clients {selected}, Byzantine {byzantine_ids}")
        return list(selected), list(byzantine_ids), benign_ids

    def _init_avg_update(self, benign_updates):
        if not benign_updates:
            return None
        avg = {}
        for k in benign_updates[0].keys():
            avg[k] = torch.mean(torch.stack([u[k].to(self.device) for u in benign_updates], dim=0), dim=0)
        return avg

    def collect_updates(self, selected_ids, byzantine_ids):
        client_updates = []
        client_losses = []
        benign_updates = []
        client_sizes = []

        selected_ids = [c for c in selected_ids if c not in byzantine_ids] + [c for c in selected_ids if c in byzantine_ids]

        for client_id in selected_ids:
            is_byzantine = client_id in byzantine_ids

            client = FederatedClient(
                client_id=client_id,
                model_name=self.model_name,
                data_name=self.data_name,
                attack_type=self.attack,
                enable_backdoor=self.enable_backdoor,
                trigger_type=self.trigger_type,
                global_lora_path=self.global_lora_path,
                federated_data_path=self.federated_train_data_path,
                is_byzantine=is_byzantine,
                local_epochs=self.local_epochs
            )
            client_sizes.append(int(getattr(client, "num_samples", 0)))

            _, local_loss = client.local_train()
            client_losses.append(local_loss)

            if not is_byzantine:
                update = client.get_update()
                client_updates.append(update)
                benign_updates.append(update)
            else:
                if hasattr(client, "attacker") and client.attacker is not None:
                    client.attacker.set_benign_stats(benign_updates)
                    client.attacker.update_round_info(self.prev_global_step_vec)

                benign_avg = self._init_avg_update(benign_updates)
                update = client.get_update(benign_avg_update=benign_avg)
                client_updates.append(update)

        avg_loss = float(np.mean(client_losses)) if client_losses else 0.0
        self.logger.info(f"Round {self.current_round}: Avg local loss {avg_loss:.4f}")
        return selected_ids, client_updates, client_sizes, avg_loss, len(benign_updates), len(byzantine_ids)

    def select_aggregator(self, client_updates, client_sizes, selected_ids, benign_client_ids=None):
        if self.aggregator == "strap":
            return self.lorasf_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes,
                client_ids=selected_ids,
                benign_client_ids=benign_client_ids,
                global_model=self.global_model,
                current_round=self.current_round,
            )
        elif self.aggregator == "median":
            return self.median_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        elif self.aggregator == "trimmed":
            return self.trimmed_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        elif self.aggregator == "fltrust":
            return self.fltrust_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        elif self.aggregator == "flshield":
            return self.flshield_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        elif self.aggregator == "fedavg":
            return self.fedavg_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        elif self.aggregator == "deepsight":
            return self.deepsight_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes,
                current_round=self.current_round,
                global_lora_path=self.global_lora_path,
                federated_data_path=self.federated_val_data_path,
            )
        elif self.aggregator == "flame":
            return self.flame_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        elif self.aggregator == "mesas":
            return self.mesas_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        elif self.aggregator == "feddlad":
            return self.feddlad_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes,
                client_ids=selected_ids
            )
        elif self.aggregator == "lasa":
            return self.lasa_aggregator.aggregate(
                client_updates=client_updates,
                client_sizes=client_sizes
            )
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator!r}.")

    def update_global_model(self, aggregated_update):
        has_nan = False
        for k, v in aggregated_update.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                has_nan = True
                aggregated_update[k] = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)

        if has_nan:
            self.logger.warning(f"[警告] Round {self.current_round}: 聚合更新中检测到 NaN/Inf！攻击力度过大或聚合器未能成功防御。")

        self.global_model = get_model(self.model_name, global_lora_path=self.global_lora_path)
        self.global_model = apply_lora_update(self.global_model, aggregated_update)

        parent_path = Path(self.global_lora_path).parent
        new_lora_path = os.path.join(parent_path, f"round_{self.current_round}")
        os.makedirs(os.path.dirname(new_lora_path), exist_ok=True)

        self.global_model.save_lora(new_lora_path)
        self.global_lora_path = new_lora_path

        lora_norm = compute_lora_norm(self.global_model)
        self.tb_logger.log_lora(self.current_round, lora_norm)
        self.logger.info(f"Round {self.current_round}: Global model updated, LoRA norm {lora_norm:.4f}")

    def evaluate_model(self):
        accs = evaluate_cls(
            model=self.global_model,
            data_name=self.data_name,
            eval_data_path=self.federated_val_data_path,
            device=self.device,
            topk=(1,)
        )
        top1 = accs.get("top1", 0.0)

        asr = None
        if self.enable_backdoor:
            asr = evaluate_backdoor_asr(
                model=self.global_model,
                data_name=self.data_name,
                trigger_type=self.trigger_type,
                eval_data_path=self.federated_val_data_path,
                device=self.device
            )

        self.logger.info(
            f"Round {self.current_round}: ACC={top1:.4f}" +
            (f", ASR={asr:.4f}" if asr is not None else "")
        )
        self.tb_logger.log_eval(self.current_round, top1, asr)
        return top1, asr

    def run(self):
        self.logger.info("=== Start ViT-LoRA-SF Experiment ===")

        for round in tqdm(range(fl_cfg["num_rounds"]), desc="Federated Rounds"):
            start_time = time.perf_counter()
            self.current_round = round + 1

            selected_ids, byzantine_ids, benign_ids = self.select_clients()
            selected_ids, client_updates, client_sizes, avg_loss, benign_cnt, byzantine_cnt = self.collect_updates(selected_ids, byzantine_ids)

            aggregated_update, metrics = self.select_aggregator(
                client_updates=client_updates,
                client_sizes=client_sizes,
                selected_ids=selected_ids,
                benign_client_ids=benign_ids,
            )
            self.update_global_model(aggregated_update)

            with torch.no_grad():
                concat_list = [v.reshape(-1).to(self.device).float() for v in aggregated_update.values()]
                step_vec = torch.cat(concat_list, dim=0) if concat_list else torch.zeros(1, device=self.device)
            self.prev_global_step_vec = step_vec.clone()

            if self.current_round % tb_cfg["log_freq"] == 0:
                self.tb_logger.log_train(self.current_round, avg_loss, benign_cnt, byzantine_cnt)
                if self.aggregator == "strap" and isinstance(metrics, dict):
                    self.tb_logger.log_lorasf(self.current_round, metrics)

                self.logger.info(f"Round {self.current_round}: select aggregator is {self.aggregator}")
                self.logger.info(f"Round {self.current_round}: update-attack={self.attack}, backdoor={self.enable_backdoor}, trigger={self.trigger_type}")

                if self.aggregator == "strap" and isinstance(metrics, dict):
                    hdr_keys = [
                        "hdr_benign_count",
                        "hdr_benign_weight_mean",
                        "hdr_benign_weight_min",
                        "hdr_benign_weight_max",
                        "hdr_benign_mean_suppression",
                        "hdr_benign_le_0p9_ratio",
                        "hdr_benign_le_0p8_ratio",
                        "hdr_benign_le_0p7_ratio",
                        "hdr_benign_le_0p6_ratio",
                        "hdr_benign_le_0p5_ratio",
                        "hdr_benign_le_0p4_ratio",
                        "hdr_benign_le_0p3_ratio",
                        "hdr_benign_le_0p2_ratio",
                        "hdr_benign_le_0p1_ratio",
                        "ref_active_num_labels",
                        "ref_filter_mode_id",
                        "ref_probe_kept_ratio",
                    ]
                    hdr_msg = []
                    for k in hdr_keys:
                        if k in metrics:
                            hdr_msg.append(f"{k}={metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k}={metrics[k]}")
                    if hdr_msg:
                        self.logger.info(f"Round {self.current_round}: fairness/HDR => " + ", ".join(hdr_msg))

                self.evaluate_model()

            end_time = time.perf_counter()
            self.round_times.append(end_time - start_time)
            self.total_times.append(metrics.get("total_time", 0.0) if isinstance(metrics, dict) else 0.0)

            torch.cuda.empty_cache()

        avg_time = sum(self.round_times) / max(1, len(self.round_times))
        std_time = math.sqrt(sum((t - avg_time) ** 2 for t in self.round_times) / max(1, len(self.round_times)))
        self.logger.info(f"Average time per round: {avg_time:.4f} seconds")
        self.logger.info(f"Standard deviation of round times: {std_time:.4f} seconds")

        self.tb_logger.close()
        self.logger.info("=== Experiment Finished ===")
        self.logger.info(f"TensorBoard: tensorboard --logdir {tb_cfg['log_dir']}")