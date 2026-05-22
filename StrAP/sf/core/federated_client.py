import torch
from tqdm import tqdm

from sf.models import get_model
from sf.utils.lora_utils import extract_lora_update
from sf.data import get_dataset
from sf.core.byzantine_attack import ByzantineAttacker
from sf.config.base_config import train as train_cfg, federated as fl_cfg


class FederatedClient:
    def __init__(self, client_id: int,
                 model_name: str,
                 data_name: str,
                 attack_type: str,
                 enable_backdoor: bool,
                 trigger_type: str,
                 global_lora_path: str,
                 federated_data_path: str,
                 is_byzantine: bool,
                 local_epochs: int):

        self.client_id = client_id
        self.is_byzantine = is_byzantine
        self.data_name = data_name
        self.device = torch.device(train_cfg.get("device", "cuda:0"))

        self.global_model = get_model(model_name, global_lora_path=global_lora_path)
        self.local_model = get_model(model_name, global_lora_path=global_lora_path)

        self.local_model.to(self.device)
        self.local_model.train()

        self.local_epochs = int(local_epochs)

        self.optimizer = torch.optim.AdamW(
            self.local_model.lora_model.parameters(),
            lr=float(train_cfg.get("lr", 5e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )

        self.data_loader = get_dataset(
            federated_data_path,
            model_name=model_name,
            client_id=client_id,
            data_name=data_name,
            is_server=False,
        ).get_loader(batch_size=int(fl_cfg.get("batch_size", 128)))

        self.num_samples = len(self.data_loader.dataset)
        print(f"Client {self.client_id} has {self.num_samples} samples.")

        if is_byzantine:
            self.attacker = ByzantineAttacker(
                attack_type=attack_type,
                device=self.device,
                enable_backdoor=enable_backdoor,
                trigger_type=trigger_type,
            )
        else:
            self.attacker = None

    def local_train(self):
        total_loss = 0.0
        accumulation_steps = int(fl_cfg.get("accumulation_steps", 1))

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(tqdm(self.data_loader, desc=f"Client {self.client_id} Ep {epoch+1}")):
                pixel = batch["pixel_values"].to(self.device)
                labels = torch.tensor(batch["labels"]).to(self.device)

                if self.is_byzantine and self.attacker is not None:
                    pixel, labels = self.attacker.poison_batch(pixel, labels, model=self.local_model)

                out = self.local_model(pixel_values=pixel, labels=labels, return_loss=True)
                logits = out["logits"] if isinstance(out, dict) and "logits" in out else out[0]
                C = logits.shape[-1]

                min_y = int(labels.min().item())
                max_y = int(labels.max().item())
                if min_y < 0 or max_y >= C:
                    raise ValueError(f"Label out of range: min={min_y}, max={max_y}, num_classes={C}")

                loss = out["loss"]
                epoch_loss += loss.item()

                (loss / accumulation_steps).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.local_model.lora_model.parameters(),
                        float(train_cfg.get("max_grad_norm", 1.0))
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            if (len(self.data_loader) % accumulation_steps) != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.local_model.lora_model.parameters(),
                    float(train_cfg.get("max_grad_norm", 1.0))
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            avg_epoch_loss = epoch_loss / max(1, len(self.data_loader))
            total_loss += avg_epoch_loss
            print(f"Client {self.client_id} Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")

        local_lora_path = f"./tmp/client_{self.client_id}_lora"
        avg_total_loss = total_loss / max(1, self.local_epochs)
        return local_lora_path, avg_total_loss

    def get_update(self, benign_avg_update=None):
        clean_update = extract_lora_update(self.global_model, self.local_model, device=self.device)

        if self.is_byzantine and self.attacker is not None:
            print(f"current client is byzantine, update-attack is {self.attacker.attack_type}, backdoor={self.attacker.enable_backdoor}, trigger={self.attacker.trigger_type}")

            poisoned = self.attacker.poison_update(clean_update, benign_avg_update)

            def _flat(u):
                return torch.cat([v.flatten() for v in u.values()]).float().to(self.device)

            a, b = _flat(clean_update), _flat(poisoned)
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            print(f"[DBG] C{self.client_id} cos(clean,poison)={cos:.3f}  ||clean||={a.norm():.2e} ||poison||={b.norm():.2e}")
            return poisoned

        return clean_update
